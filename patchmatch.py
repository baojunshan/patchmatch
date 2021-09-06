import numpy as np
import time


class PatchMatch:
    def __init__(self, patch_size, alpha=0.5, radius=10):
        self.patch_size = patch_size
        self.alpha = alpha
        self.radius = radius

    def calc_dist(self, x_c, y_c, x, y):
        # square distance
        p = self.patch_size // 2
        x_p = x[x_c[0]: x_c[0] + self.patch_size, x_c[1]: x_c[1] + self.patch_size, :]
        y_p = y[y_c[0] - p: y_c[0] + p + 1, y_c[1] - p: y_c[1] + p + 1, :]
        diff = y_p - x_p
        dist = np.sum(np.square(np.nan_to_num(diff))) / np.sum(~np.isnan(diff))
        return dist

    @staticmethod
    def reconstruction(x, y, f):
        x_h, x_w, _ = x.shape
        x_new = np.zeros_like(x)
        for i in range(x_h):
            for j in range(x_w):
                x_new[i, j, :] = y[f[i, j][0], f[i, j][1], :]
        return x_new

    def init(self, x, y):
        x_h, x_w, x_c = x.shape
        y_h, y_w, y_c = y.shape

        p = self.patch_size // 2

        x_padding = np.ones([x_h + 2 * p, x_w + 2 * p, x_c]) * np.nan
        x_padding[p: x_h + p, p: x_w + p, :] = x

        f = np.zeros([x_h, x_w], dtype=object)  # x pixel project to y
        d = np.zeros([x_h, x_w])  # distance of x pixel and projected y pixel

        for i in range(x_h):
            for j in range(x_w):
                # the center point of patch
                x_p_c = np.array([i, j])
                y_p_c = np.array([np.random.randint(p, y_h - p), np.random.randint(p, y_w - p)])
                f[i, j] = y_p_c
                d[i, j] = self.calc_dist(x_c=x_p_c, y_c=y_p_c, x=x_padding, y=y)
        return f, d, x_padding

    def propagation(self, x, y, curr_p, f, d, odd=False):
        x_h = x.shape[0] - self.patch_size + 1
        x_w = x.shape[1] - self.patch_size + 1
        curr_h, curr_w = curr_p
        if not odd:
            d_left = d[curr_h, max(curr_w - 1, 0)]
            d_up = d[max(curr_h - 1, 0), curr_w]
            d_curr = d[curr_h, curr_w]
            d_min = min([d_left, d_up, d_curr])
            if d_min == d_curr:
                return
            elif d_min == d_up:
                f[curr_h, curr_w] = f[max(curr_h - 1, 0), curr_w]
            else:
                f[curr_h, curr_w] = f[curr_h, max(curr_w - 1, 0)]
        else:
            d_right = d[curr_h, min(curr_w + 1, x_w - 1)]
            d_down = d[min(curr_h + 1, x_h - 1), curr_w]
            d_curr = d[curr_h, curr_w]
            d_min = min([d_right, d_down, d_curr])
            if d_min == d_curr:
                return
            elif d_min == d_down:
                f[curr_h, curr_w] = f[min(curr_h + 1, x_h - 1), curr_w]
            else:
                f[curr_h, curr_w] = f[curr_h, min(curr_w + 1, x_w - 1)]
        d[curr_h, curr_w] = self.calc_dist(x_c=curr_p, y_c=f[curr_h, curr_w], x=x, y=y)

    def random_search(self, curr_p, x, y, f, d):
        curr_h, curr_w = curr_p
        y_h, y_w, _ = y.shape
        p = self.patch_size // 2

        # radius = h * alpha ** i
        i = int(np.log(self.radius / y_h) / np.log(self.alpha))
        search_h = y_h * self.alpha ** i
        search_w = y_w * self.alpha ** i

        y_curr_h = f[curr_h, curr_w][0]
        y_curr_w = f[curr_h, curr_w][1]

        while search_h > 1 and search_w > 1:
            random_h = np.random.randint(
                max(y_curr_h - search_h, p),
                min(y_curr_h + search_h, y_h - p)
            )
            random_w = np.random.randint(
                max(y_curr_w - search_w, p),
                min(y_curr_w + search_w, y_w - p)
            )
            random_p = np.array([random_h, random_w])
            random_d = self.calc_dist(x_c=curr_p, y_c=random_p, x=x, y=y)
            if random_d < d[curr_h, curr_w]:
                d[curr_h, curr_w] = random_d
                f[curr_h, curr_w] = random_p

            i += 1
            search_h = y_h * self.alpha ** i
            search_w = y_w * self.alpha ** i

    def train(self, x, y, n_iter=10):
        start_time = time.time()
        f, d, x_padding = self.init(x, y)
        x_h, x_w, _ = x.shape
        print(f"| Init cost {time.time() - start_time:.3f} seconds.")

        for itr in range(1, n_iter + 1):
            curr_time = time.time()
            if itr % 2 != 0:
                for i in range(x_h):
                    for j in range(x_w):
                        curr_p = np.array([i, j])
                        self.propagation(x=x_padding, y=y, curr_p=curr_p, f=f, d=d, odd=False)
                        self.random_search(curr_p=curr_p, x=x_padding, y=y, f=f, d=d)
            else:
                for i in range(x_h - 1, -1, -1):
                    for j in range(x_w - 1, -1, -1):
                        curr_p = np.array([i, j])
                        self.propagation(x=x_padding, y=y, curr_p=curr_p, f=f, d=d, odd=True)
                        self.random_search(curr_p=curr_p, x=x_padding, y=y, f=f, d=d)
            print(f"| Finish {itr}/{n_iter} iteration, cost {time.time() - curr_time:.3} seconds.")
        return self.reconstruction(x=x, y=y, f=f)


if __name__ == "__main__":
    import cv2

    p_size = 3
    n_iter = 10

    a = cv2.imread("images/pm_0_left.jpg")
    b = cv2.imread("images/pm_0_right.jpg")
    print(a.shape)

    patchmatch = PatchMatch(patch_size=p_size)
    c = patchmatch.train(x=a, y=b, n_iter=10)
    cv2.imwrite("images/pm_0_res.jpg", c)



