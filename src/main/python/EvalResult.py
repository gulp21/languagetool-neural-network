class EvalResult():
    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

    def add_tp(self):
        self.tp += 1

    def add_fp(self):
        self.fp += 1

    def add_tn(self):
        self.tn += 1

    def add_fn(self):
        self.fn += 1

    def recall(self):
        return self.tp / (self.tp + self.fn)

    def precision(self):
        if (self.tp + self.fp) > 0:
            return self.tp / (self.tp + self.fp)
        return 0

    def __str__(self):
        return "<tp: %d, fp: %d, tn: %d, fn: %d, p: %3.2f, r: %3.2f>\n" % (self.tp, self.fp, self.tn, self.fn, self.precision(), self.recall())

    def __repr__(self):
        return self.__str__()
