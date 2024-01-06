class FuncInterval:
    def __int__(self, dim, func, start, end):
        self.dim = dim
        if dim != 0 and dim != 1:
            raise ValueError(f"dim: {dim} has to be 0 or 1")
        self.start = start
        if start < 0 or start > 1:
            raise ValueError(f"start: {start} out of bounds (<0 or >1)")
        self.end = end
        if end < 0 or end > 1:
            raise ValueError(f"start: {end} out of bounds (<0 or >1)")

        # func should be a number between 1 and 16
        self.func = func

    def contains(self, x):
        if x > self.start and x < self.end:
            return True
        else:
            return False

    def split(self, x, func_a, func_b):
        interval_a = FuncInterval(self.dim, func_a, self.start, x)
        interval_b = FuncInterval(self.dim, func_b, x, self.end)
