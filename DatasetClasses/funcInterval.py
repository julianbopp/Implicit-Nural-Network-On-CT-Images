from functools import total_ordering


@total_ordering
class FuncInterval:
    def __init__(
        self, dim, start, end, x_sign=None, y_sign=None, x_dist=None, y_dist=None
    ):
        self.x_sign = x_sign
        self.y_sign = y_sign
        self.x_dist = x_dist
        self.y_dist = y_dist
        self.dim = dim
        self.start = start
        self.end = end

    def contains(self, z):
        if self.start < z < self.end:
            return True
        else:
            return False

    def split(self, z, a_x_sign=None, a_y_sign=None, b_x_sign=None, b_y_sign=None):
        if z <= self.start or z >= self.end:
            return self, self
        else:
            interval_a = FuncInterval(
                self.dim,
                self.start,
                z,
                x_sign=a_x_sign,
                y_sign=a_y_sign,
                x_dist=self.x_dist,
                y_dist=self.y_dist,
            )
            interval_b = FuncInterval(
                self.dim,
                z,
                self.end,
                x_sign=b_x_sign,
                y_sign=b_y_sign,
                x_dist=self.x_dist,
                y_dist=self.y_dist,
            )

            return interval_a, interval_b

    def join(self, other):
        if self.__eq__(other) and self.dim != other.dim:
            if self.dim == "x":
                joint_interval = FuncInterval(
                    "xy",
                    self.start,
                    self.end,
                    x_sign=self.x_sign,
                    y_sign=other.y_sign,
                    x_dist=self.x_dist,
                    y_dist=other.y_dist,
                )
            elif self.dim == "y":
                joint_interval = FuncInterval(
                    "xy",
                    self.start,
                    self.end,
                    x_sign=other.x_sign,
                    y_sign=self.y_sign,
                    x_dist=other.x_dist,
                    y_dist=self.y_dist,
                )
            else:
                raise ValueError("dim not set")
        else:
            raise ValueError("Intervals can not be combined")

        return joint_interval

    def modify_length(self, new_start, new_end):
        return FuncInterval(
            self.dim,
            new_start,
            new_end,
            x_sign=self.x_sign,
            y_sign=self.y_sign,
            x_dist=self.x_dist,
            y_dist=self.y_dist,
        )

    def __eq__(self, other):
        return self.start == other.start and self.end == other.end

    def __lt__(self, other):
        return self.start < other.start

    def __str__(self):
        return f"({self.start,self.end}), dim: {self.dim}, x_sign: {self.x_sign}, y_sign, {self.y_sign}, x_dist: {self.x_dist}, y_dist: {self.y_dist}"
