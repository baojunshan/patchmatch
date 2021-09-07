from functools import wraps
from abc import abstractmethod
import time
import json
import random
import numpy as np


def timer(pre_str=None):
    def timed(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            start_time = time.time()
            res = func(*args, **kwargs)
            print_srt = pre_str or func.__name__
            print(f"| {print_srt} cost {time.time() - start_time:.3f} seconds.")
            return res

        return wrapped

    return timed


class PMSConfig:
    def __init__(self, path):
        self.path = path
        with open(path, "r") as f:
            self.config = json.load(f)
        for k, v in self.config.items():
            setattr(self, k, v)

    def clone(self):
        return PMSConfig(self.path)


class PVector3f:
    def __init__(self, x: float, y: float, z: float):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def normalize(self):
        sqf = (self.x ** 2 + self.y ** 2 + self.z ** 2) ** 0.5
        self.x /= sqf
        self.y /= sqf
        self.z /= sqf
        return self

    def __mul__(self, other):
        if not isinstance(other, PVector3f):
            raise TypeError(f"{type(self)} and {type(other)} could not multiply.")
        return self.x * other.x + self.y * other.y + self.z * other.z

    def __add__(self, other):
        if not isinstance(other, PVector3f):
            raise TypeError(f"{type(self)} and {type(other)} could not add.")
        return PVector3f(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        if not isinstance(other, PVector3f):
            raise TypeError(f"{type(self)} and {type(other)} could not sub.")
        return PVector3f(self.x - other.x, self.y - other.y, self.z - other.z)

    def __invert__(self):
        return PVector3f(-self.x, -self.y, -self.z)

    def __eq__(self, other):
        if not isinstance(other, PVector3f):
            raise TypeError(f"{type(self)} and {type(other)} could not compare.")
        return self.x == other.x and self.y == other.y and self.z == other.z


class DisparityPlane:
    def __init__(self, x: int = 0, y: int = 0, d: int = 0, n: PVector3f = None, p: PVector3f = None):
        if p is None:
            x, y, z = -n.x / n.z, -n.y / n.z, (n.x * x + n.y * y + n.z * d) / n.z
            self.p = PVector3f(x, y, z)
        else:
            self.p = PVector3f(p.x, p.y, p.z)

    def to_disparity(self, x: int, y: int):
        return self.p * PVector3f(x, y, 1)

    def to_norm(self):
        return PVector3f(self.p.x, self.p.y, -1).normalize()

    def to_another_view(self, x: int, y: int):
        d = self.to_disparity(x, y)
        return DisparityPlane(p=PVector3f(-self.p.x, -self.p.y, -self.p.z - self.p.x * d))

    def __eq__(self, other):
        if not isinstance(other, DisparityPlane):
            raise TypeError(f"{type(self)} and {type(other)} could not compare.")
        return self.p == other.p


class CostComputer:
    def __init__(self, image_left, image_right, width, height, patch_size, min_disparity, max_disparity):
        self.image_left = image_left
        self.image_right = image_right
        self.width = width
        self.height = height
        self.patch_size = patch_size
        self.min_disparity = min_disparity
        self.max_disparity = max_disparity

    @staticmethod
    def fast_exp(v: float):
        v = 1 + v / 1024
        for _ in range(10):
            v *= v
        return v

    @abstractmethod
    def compute(self, x, y, d, *args, **kwargs):
        raise NotImplementedError("Compute should be implement.")


class CostComputerPMS(CostComputer):
    def __init__(self, image_left, image_right, grad_left, grad_right, width, height, patch_size, min_disparity,
                 max_disparity, gamma, alpha, tau_col, tau_grad):
        super(CostComputerPMS, self).__init__(image_left=image_left, image_right=image_right,
                                              width=width, height=height, patch_size=patch_size,
                                              min_disparity=min_disparity, max_disparity=max_disparity)
        self.grad_left = grad_left
        self.grad_right = grad_right
        self.gamma = gamma
        self.alpha = alpha
        self.tau_col = tau_col
        self.tau_grad = tau_grad

    def compute(self, x=0.0, y=0, d=0.0, col_p=None, grad_p=None):
        xr = x - d
        if xr < 0 or xr > self.width:
            return (1 - self.alpha) * self.tau_col + self.alpha * self.tau_grad

        col_p = col_p if col_p is not None else self.get_color(self.image_left, x=x, y=y)
        col_q = self.get_color(self.image_right, x=xr, y=y)
        dc = sum([abs(float(col_p[i]) - float(col_q[i])) for i in range(3)])
        dc = min(dc, self.tau_col)

        grad_p = grad_p if grad_p is not None else self.get_gradient(self.grad_left, x=x, y=y)
        grad_q = self.get_gradient(self.grad_right, x=xr, y=y)
        dg = abs(grad_p[0] - grad_q[0]) + abs(grad_p[1] - grad_q[1])
        dg = min(dg, self.tau_grad)

        return (1 - self.alpha) * dc + self.alpha * dg

    def compute_agg(self, x, y, p: DisparityPlane):
        pat = self.patch_size // 2
        col_p = self.image_left[y, x, :]
        cost = 0
        for r in range(-pat, pat, 1):
            y_ = y + r
            for c in range(-pat, pat, 1):
                x_ = x + c
                if y_ < 0 or y_ > self.height - 1 or x_ < 0 or x_ > self.width - 1:
                    continue
                d = p.to_disparity(x=x, y=y)
                if d < self.min_disparity or d > self.max_disparity:
                    cost += 120.0
                    continue
                col_q = self.image_left[y_, x_, :]
                dc = sum([abs(float(col_p[i]) - float(col_q[i])) for i in range(3)])
                # w = np.exp(-dc / self.gamma)
                w = self.fast_exp(-dc / self.gamma)
                grad_q = self.grad_left[y_, x_]
                cost += w * self.compute(x=x_, y=y_, d=d, col_p=col_q, grad_p=grad_q)
        return cost

    def get_color(self, image, x: float, y: int):
        x1 = int(np.floor(x))
        x2 = int(np.ceil(x))
        ofs = x - x1
        color = list()
        for i in range(3):
            g1 = image[y, x1, i]
            g2 = image[y, x2, i] if x2 < self.width else g1
            color.append((1 - ofs) * g1 + ofs * g2)
        return color

    def get_gradient(self, gradient, x: float, y: int):
        x1 = int(np.floor(x))
        x2 = int(np.ceil(x))
        ofs = x - x1
        g1 = gradient[y, x1]
        g2 = gradient[y, x2] if x2 < self.width else g1
        x_ = (1 - ofs) * g1[0] + ofs * g2[0]
        y_ = (1 - ofs) * g1[1] + ofs * g2[1]
        return [x_, y_]


class PropagationPMS:
    def __init__(self, image_left, image_right, width, height, grad_left, grad_right,
                 plane_left, plane_right, config, cost_left, cost_right, disparity_map):
        self.image_left = image_left
        self.image_right = image_right
        self.width = width
        self.height = height
        self.grad_left = grad_left
        self.grad_right = grad_right
        self.plane_left = plane_left
        self.plane_right = plane_right
        self.config = config
        self.cost_left = cost_left
        self.cost_right = cost_right
        self.disparity_map = disparity_map
        self.cost_cpt_left = CostComputerPMS(image_left, image_right, grad_left, grad_right, width, height,
                                             config.patch_size, config.min_disparity, config.max_disparity,
                                             config.gamma, config.alpha, config.tau_col, config.tau_grad)
        self.cost_cpt_right = CostComputerPMS(image_right, image_left, grad_right, grad_left, width, height,
                                             config.patch_size, -config.max_disparity, -config.min_disparity,
                                             config.gamma, config.alpha, config.tau_col, config.tau_grad)
        self.compute_cost_data()

    def do_propagation(self, curr_iter):
        direction = 1 if curr_iter % 2 == 0 else -1
        y = 0 if curr_iter % 2 == 0 else self.height - 1
        count_ = 0
        times_ = 0
        print(f"\r| Propagation iter {curr_iter}: 0", end="")
        for i in range(self.height):
            x = 0 if curr_iter % 2 == 0 else self.width - 1
            for j in range(self.width):
                start_ = time.time()
                self.spatial_propagation(x=x, y=y, direction=direction)
                if not self.config.is_force_fpw:
                    self.plane_refine(x=x, y=y)
                self.view_propagation(x=x, y=y)
                x += direction
                times_ += time.time() - start_
                count_ += 1
                print(f"\r| Propagation iter {curr_iter}: [{y * self.width + x + 1} / {self.height * self.width}] {times_:.0f}s/{times_ / count_ * (self.width * self.height - count_):.0f}s, {count_/times_:.3f} it/s", end="")
            y += direction
        print(f"\r| Propagation iter {curr_iter} cost {times_:.3f} seconds.")

    def compute_cost_data(self):
        print(f"\r| Init cost {0} / {self.height * self.width}", end="")
        count_ = 0
        times_ = 0
        for y in range(self.height):
            for x in range(self.width):
                start_ = time.time()
                p = self.plane_left[y, x]
                self.cost_left[y, x] = self.cost_cpt_left.compute_agg(x=x, y=y, p=p)
                times_ += time.time() - start_
                count_ += 1
                print(f"\r| Initialize cost [{y * self.width + x + 1} / {self.height * self.width}] {times_:.0f}s/{times_ / count_ * (self.width * self.height - count_):.0f}s, {count_/times_:.3f} it/s", end="")
        print(f"\r| Initialize cost {times_:.3f} seconds.")

    def spatial_propagation(self, x, y, direction):
        plane_p = self.plane_left[y, x]
        cost_p = self.cost_left[y, x]

        xd = x - direction
        if 0 <= xd < self.width:
            plane = self.plane_left[y, xd]
            if plane != plane_p:
                cost = self.cost_cpt_left.compute_agg(x=x, y=y, p=plane)
                if cost < cost_p:
                    plane_p = plane
                    cost_p = cost

        yd = y - direction
        if 0 <= yd < self.height:
            plane = self.plane_left[yd, x]
            if plane != plane_p:
                cost = self.cost_cpt_left.compute_agg(x=x, y=y, p=plane)
                if cost < cost_p:
                    plane_p = plane
                    cost_p = cost

        self.plane_left[y, x] = plane_p
        self.cost_left[y, x] = cost_p

    def view_propagation(self, x, y):
        plane_p = self.plane_left[y, x]
        d_p = plane_p.to_disparity(x=x, y=y)

        xr = int(x - d_p)
        if xr < 0 or xr > self.width - 1:
            return
        plane_q = self.plane_right[y, xr]
        cost_q = self.cost_right[y, xr]

        plane_p2q = plane_p.to_another_view(x=x, y=y)
        d_q = plane_p2q.to_disparity(x=xr, y=y)
        cost = self.cost_cpt_right.compute_agg(x=xr, y=y, p=plane_p2q)
        if cost < cost_q:
            plane_q = plane_p2q
            cost_q = cost
        self.plane_right[y, xr] = plane_q
        self.cost_right[y, xr] = cost_q

    def plane_refine(self, x, y):
        min_disp = self.config.min_disparity
        max_disp = self.config.max_disparity

        plane_p = self.plane_left[y, x]
        cost_p = self.cost_left[y, x]
        d_p = plane_p.to_disparity(x=x, y=y)
        norm_p = plane_p.to_norm()

        disp_update = (max_disp - min_disp) / 2.0
        norm_update = 1.0
        stop_thres = 0.1

        while disp_update > stop_thres:
            disp_rd = np.random.uniform(-1.0, 1.0) * disp_update
            if self.config.is_integer_disparity:
                disp_rd = int(disp_rd)
            d_p_new = d_p + disp_rd
            if d_p_new < min_disp or d_p_new > max_disp:
                disp_update /= 2
                norm_update /= 2
                continue

            if not self.config.is_force_fpw:
                norm_rd = PVector3f(
                    x=np.random.uniform(-1.0, 1.0) * norm_update,
                    y=np.random.uniform(-1.0, 1.0) * norm_update,
                    z=np.random.uniform(-1.0, 1.0) * norm_update,
                )
                while norm_rd.z == 0.0:
                    norm_rd.z = np.random.uniform(-1.0, 1.0)
            else:
                norm_rd = PVector3f(x=0.0, y=0.0, z=0.0)

            norm_p_new = norm_p + norm_rd
            norm_p_new.normalize()

            plane_new = DisparityPlane(x=x, y=y, d=d_p_new, n=norm_p_new)

            if plane_new != plane_p:
                cost = self.cost_cpt_left.compute_agg(x=x, y=y, p=plane_new)
                if cost < cost_p:
                    plane_p = plane_new
                    cost_p = cost
                    d_p = d_p_new
                    norm_p = norm_p_new
                    self.plane_left[y, x] = plane_p
                    self.cost_left[y, x] = cost_p
            disp_update /= 2.0
            norm_update /= 2.0


class PatchMatchStereo:
    def __init__(self, width, height, config, random_seed=2021):
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        self.image_left = None
        self.image_right = None
        self.width = width
        self.height = height
        self.config = config
        self.disparity_range = config.max_disparity - config.min_disparity
        self.gray_left = None
        self.gray_right = None
        self.grad_left = None
        self.grad_right = None
        self.cost_left = None
        self.cost_right = None
        self.disparity_left = None
        self.disparity_right = None
        self.plane_left = None
        self.plane_right = None
        self.mistakes_left = None
        self.mistakes_right = None
        self.invalid_disparity = 1024.0
        self.init(h=height, w=width)

    @timer("Initialize memory")
    def init(self, h, w):
        self.width = w
        self.height = h

        self.disparity_range = self.config.max_disparity - self.config.min_disparity
        self.gray_left = np.zeros([self.height, self.width], dtype=int)
        self.gray_right = np.zeros([self.height, self.width], dtype=int)
        self.grad_left = np.zeros([self.height, self.width, 2], dtype=float)
        self.grad_right = np.zeros([self.height, self.width, 2], dtype=float)
        self.cost_left = np.zeros([self.height, self.width], dtype=float)
        self.cost_right = np.zeros([self.height, self.width], dtype=float)
        self.disparity_left = np.zeros([self.height, self.width], dtype=float)
        self.disparity_right = np.zeros([self.height, self.width], dtype=float)
        self.plane_left = np.zeros([self.height, self.width], dtype=object)
        self.plane_right = np.zeros([self.height, self.width], dtype=object)
        self.mistakes_left = list()
        self.mistakes_right = list()

    @timer("Initialize parameters")
    def random_init(self):
        for y in range(self.height):
            for x in range(self.width):
                # random disparity
                disp_l = np.random.uniform(float(self.config.min_disparity), float(self.config.max_disparity))
                disp_r = np.random.uniform(float(self.config.min_disparity), float(self.config.max_disparity))
                if self.config.is_integer_disparity:
                    disp_l, disp_r = int(disp_l), int(disp_r)
                self.disparity_left[y, x], self.disparity_right[y, x] = disp_l, disp_r

                # random normal vector
                norm_l, norm_r = PVector3f(x=0.0, y=0.0, z=1.0), PVector3f(x=0.0, y=0.0, z=1.0)
                if not self.config.is_force_fpw:
                    norm_l = PVector3f(
                        x=np.random.uniform(-1.0, 1.0),
                        y=np.random.uniform(-1.0, 1.0),
                        z=np.random.uniform(-1.0, 1.0),
                    )
                    norm_r = PVector3f(
                        x=np.random.uniform(-1.0, 1.0),
                        y=np.random.uniform(-1.0, 1.0),
                        z=np.random.uniform(-1.0, 1.0),
                    )
                    while norm_l.z == 0.0:
                        norm_l.z = np.random.uniform(-1.0, 1.0)
                    while norm_r.z == 0.0:
                        norm_r.z = np.random.uniform(-1.0, 1.0)
                    norm_l, norm_r = norm_l.normalize(), norm_r.normalize()

                # random disparity plane
                self.plane_left[y, x] = DisparityPlane(x=x, y=y, d=disp_l, n=norm_l)
                self.plane_right[y, x] = DisparityPlane(x=x, y=y, d=disp_r, n=norm_r)

    @timer("Plane to disparity")
    def plane_to_disparity(self):
        for y in range(self.height):
            for x in range(self.width):
                self.disparity_left[y, x] = self.plane_left[y, x].to_disparity(x=x, y=y)
                self.disparity_right[y, x] = self.plane_right[y, x].to_disparity(x=x, y=y)

    @timer("Total match")
    def match(self, image_left, image_right):
        self.image_left, self.image_right = image_left, image_right
        self.random_init()
        self.compute_gray()
        self.compute_gradient()
        self.propagation()
        self.plane_to_disparity()

        if self.config.is_check_lr:
            self.lr_check()
        if self.config.is_fill_holes:
            self.fill_holes_in_disparity_map()

    @timer("Propagation")
    def propagation(self):
        config_left = self.config.clone()
        config_right = self.config.clone()
        config_right.min_disparity = -config_left.max_disparity
        config_right.max_disparity = -config_left.min_disparity
        propa_left = PropagationPMS(self.image_left, self.image_right, self.width, self.height,
                                    self.grad_left, self.grad_right, self.plane_left, self.plane_right,
                                    config_left, self.cost_left, self.cost_right, self.disparity_left)
        propa_right = PropagationPMS(self.image_right, self.image_left, self.width, self.height,
                                    self.grad_right, self.grad_left, self.plane_right, self.plane_left,
                                    config_right, self.cost_right, self.cost_left, self.disparity_right)
        for it in range(self.config.n_iter):
            propa_left.do_propagation(curr_iter=it)
            propa_right.do_propagation(curr_iter=it)

    @timer("Initialize gray")
    def compute_gray(self):
        for y in range(self.height):
            for x in range(self.width):
                b, g, r = self.image_left[y, x]
                self.gray_left[y, x] = int(r * 0.299 + g * 0.587 + b * 0.114)
                b, g, r = self.image_right[y, x]
                self.gray_right[y, x] = int(r * 0.299 + g * 0.587 + b * 0.114)

    @timer("Initialize gradient")
    def compute_gradient(self):
        for y in range(1, self.height - 1, 1):
            for x in range(1, self.width - 1, 1):
                grad_x = self.gray_left[y - 1, x + 1] - self.gray_left[y - 1, x - 1] \
                         + 2 * self.gray_left[y, x + 1] - 2 * self.gray_left[y, x - 1] \
                         + self.gray_left[y + 1, x + 1] - self.gray_left[y + 1, x - 1]
                grad_y = self.gray_left[y + 1, x - 1] - self.gray_left[y - 1, x - 1] \
                         + 2 * self.gray_left[y + 1, x] - 2 * self.gray_left[y - 1, x] \
                         + self.gray_left[y + 1, x + 1] - self.gray_left[y - 1, x + 1]
                grad_y, grad_x = grad_y / 8, grad_x / 8
                self.grad_left[y, x, 0] = grad_x
                self.grad_left[y, x, 1] = grad_y

                grad_x = self.gray_right[y - 1, x + 1] - self.gray_right[y - 1, x - 1] \
                         + 2 * self.gray_right[y, x + 1] - 2 * self.gray_right[y, x - 1] \
                         + self.gray_right[y + 1, x + 1] - self.gray_right[y + 1, x - 1]
                grad_y = self.gray_right[y + 1, x - 1] - self.gray_right[y - 1, x - 1] \
                         + 2 * self.gray_right[y + 1, x] - 2 * self.gray_right[y - 1, x] \
                         + self.gray_right[y + 1, x + 1] - self.gray_right[y - 1, x + 1]
                grad_y, grad_x = grad_y / 8, grad_x / 8
                self.grad_right[y, x, 0] = grad_x
                self.grad_right[y, x, 1] = grad_y

    @timer("LR check")
    def lr_check(self):
        for y in range(self.height):
            for x in range(self.width):
                disp = self.disparity_left[y, x]
                if disp == self.invalid_disparity:
                    self.mistakes_left.append([x, y])
                    continue
                col_right = round(x - disp)
                if 0 <= col_right < self.width:
                    disp_r = self.disparity_right[y, col_right]
                    if abs(disp + disp_r) > self.config.lr_check_threshold:
                        self.disparity_left[y, x] = self.invalid_disparity
                        self.mistakes_left.append([x, y])
                else:
                    self.disparity_left[y, x] = self.invalid_disparity
                    self.mistakes_left.append([x, y])

        for y in range(self.height):
            for x in range(self.width):
                disp = self.disparity_right[y, x]
                if disp == self.invalid_disparity:
                    self.mistakes_right.append([x, y])
                    continue
                col_right = round(x - disp)
                if 0 <= col_right < self.width:
                    disp_r = self.disparity_left[y, col_right]
                    if abs(disp + disp_r) > self.config.lr_check_threshold:
                        self.disparity_right[y, x] = self.invalid_disparity
                        self.mistakes_right.append([x, y])
                else:
                    self.disparity_right[y, x] = self.invalid_disparity
                    self.mistakes_right.append([x, y])

    @timer("Fill holes")
    def fill_holes_in_disparity_map(self):
        for i in range(len(self.mistakes_left)):
            left_planes = list()
            x, y = self.mistakes_left[i]
            xs = x + 1
            while xs < self.width:
                if self.disparity_left[y, xs] != self.invalid_disparity:
                    left_planes.append(self.plane_left[y, xs])
                    break
                xs += 1
            xs = x - 1
            while xs >= 0:
                if self.disparity_left[y, xs] != self.invalid_disparity:
                    left_planes.append(self.plane_left[y, xs])
                    break
                xs -= 1
            if len(left_planes) == 1:
                self.disparity_left[y, x] = left_planes[0].to_disparity(x=x, y=y)
            elif len(left_planes) > 1:
                d0 = left_planes[0].to_disparity(x=x, y=y)
                d1 = left_planes[1].to_disparity(x=x, y=y)
                self.disparity_left[y, x] = min(abs(d0), abs(d1))

        for i in range(len(self.mistakes_right)):
            right_planes = list()
            x, y = self.mistakes_right[i]
            xs = x + 1
            while xs < self.width:
                if self.disparity_right[y, xs] != self.invalid_disparity:
                    right_planes.append(self.plane_right[y, xs])
                    break
                xs += 1
            xs = x - 1
            while xs >= 0:
                if self.disparity_right[y, xs] != self.invalid_disparity:
                    right_planes.append(self.plane_right[y, xs])
                    break
                xs -= 1
            if len(right_planes) == 1:
                self.disparity_right[y, x] = right_planes[0].to_disparity(x=x, y=y)
            elif len(right_planes) > 1:
                d0 = right_planes[0].to_disparity(x=x, y=y)
                d1 = right_planes[1].to_disparity(x=x, y=y)
                self.disparity_right[y, x] = min(abs(d0), abs(d1))

    @timer("Get disparity map")
    def get_disparity_map(self, view=0, norm=False):
        return self._get_disparity_map(view=view, norm=norm)

    def _get_disparity_map(self, view=0, norm=False):
        if view == 0:
            disparity = self.disparity_left.copy()
        else:
            disparity = self.disparity_right.copy()
        if norm:
            disparity = np.clip(disparity, self.config.min_disparity, self.config.max_disparity)
            disparity = disparity / (self.config.max_disparity - self.config.min_disparity) * 255
        return disparity

    @timer("Get disparity cloud")
    def get_disparity_cloud(self, baseline, focal_length, principal_point_left, principal_point_right):
        b = baseline
        f = focal_length
        l_x, l_y = principal_point_left
        r_x, r_y = principal_point_right
        cloud = list()
        for y in range(self.height):
            for x in range(self.width):
                disp = np.abs(self._get_disparity_map(view=0)[y, x])
                z_ = b * f / (disp + (r_x - l_x))
                x_ = z_ * (x - l_x) / f
                y_ = z_ * (y - l_y) / f
                cloud.append([x_, y_, z_])
        return cloud


if __name__ == "__main__":
    import cv2

    left = cv2.imread("images/pms_0_left.png")
    right = cv2.imread("images/pms_0_right.png")
    config_ = PMSConfig("config.json")

    height_, width_ = left.shape[0], left.shape[1]
    p = PatchMatchStereo(height=height_, width=width_, config=config_)
    p.match(image_left=left, image_right=right)
    disparity_ = p.get_disparity_map(view=0, norm=True)
    cv2.imwrite("./images/pms_0_disparity.png", disparity_)

    cloud = p.get_disparity_cloud(
        baseline=193.001,
        focal_length=999.421,
        principal_point_left=(294.182, 252.932),
        principal_point_right=(326.95975, 252.932)
    )
    with open("./images/pms_0_clouds.txt", "w") as f:
        for c in cloud:
            f.write(" ".join([str(i) for i in c]) + "\n")
