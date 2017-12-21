class EvalResult():
    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

    def add_tp(self):
        self.tp = self.tp + 1

    def add_fp(self):
        self.fp = self.fp + 1

    def add_tn(self):
        self.tn = self.tn + 1

    def add_fn(self):
        self.fn = self.fn + 1

    def __str__(self):
        return str({"tp": self.tp, "fp": self.fp, "tn": self.tn, "fn": self.fn})
