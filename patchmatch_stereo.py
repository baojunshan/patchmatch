from functools import wraps
import time


def timer(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        start_time = time.time()
        res = func(*args, **kwargs)
        print(f"| {func.__name__} cost {time.time() - start_time:.3f} seconds.")
        return res

    return wrapped


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
        pass

    @timer
    def init(self):
        time.sleep(2)

    def train(self, x, y):
        self.init()


if __name__ == "__main__":
    p = PatchMatchStereo()

    p.train(x=1, y=2)
