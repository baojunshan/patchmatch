from functools import wraps
import time
import json


def timer(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        start_time = time.time()
        res = func(*args, **kwargs)
        print(f"| {func.__name__} cost {time.time() - start_time:.3f} seconds.")
        return res

    return wrapped


class PMSConfig:
    def __init__(self, path):
        with open(path, "r") as f:
            self.config = json.load(f)
        for k, v in self.config.items():
            setattr(self, k, v)


class PGraient:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y


class PVector3f:
    def __init__(self, x, y, z):
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

    def get_disparity(self, x: int, y: int):
        return self.p * PVector3f(x, y, 1)

    def get_normal(self):
        return PVector3f(self.p.x, self.p.y, -1).normalize()

    def get_another_view(self, x: int, y: int):
        d = self.get_disparity(x, y)
        return DisparityPlane(p=PVector3f(-self.p.x, -self.p.y, -self.p.z - self.p.x * d))

    def __eq__(self, other):
        if not isinstance(other, DisparityPlane):
            raise TypeError(f"{type(self)} and {type(other)} could not compare.")
        return self.p == other.p


class PatchMatchStereo:
    def __init__(self):
        self.width = 0
        self.height = 0
        self.config = config

    def init(self, h, w, config):
        # ··· 赋值
        # 影像尺寸
        self.width = w;
        self.height = h;
        # PMS参数
        self.config = config;

        if (width <= 0 || height <= 0) {
        return false;
        }

        //··· 开辟内存空间
        const sint32 img_size = width * height;
        const sint32 disp_range = option.max_disparity - option.min_disparity;
        // 灰度数据
        gray_left_ = new uint8[img_size];
        gray_right_ = new uint8[img_size];
        // 梯度数据
        grad_left_ = new PGradient[img_size]();
        grad_right_ = new PGradient[img_size]();
        // 代价数据
        cost_left_ = new float32[img_size];
        cost_right_ = new float32[img_size];
        // 视差图
        disp_left_ = new float32[img_size];
        disp_right_ = new float32[img_size];
        // 平面集
        plane_left_ = new DisparityPlane[img_size];
        plane_right_ = new DisparityPlane[img_size];

        is_initialized_ = grad_left_ && grad_right_ && disp_left_ && disp_right_  && plane_left_ && plane_right_;

        return is_initialized_;

    @timer
    def random_init(self, w, h):
        pass

    def train(self, x, y):
        self.random_init(x, y)


if __name__ == "__main__":
    p = PatchMatchStereo()

    p.train(x=1, y=2)

    config = PMSConfig("config.json")
    print(config.n_iter)
