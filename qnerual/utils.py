class AvgMeter():
    sum: float
    count: int
    
    def __init__(self):
        self.reset()
        self.sum = 0.
        self.count = 0

    def reset(self):
        self.sum = 0.
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.sum += val
        self.count += n

    @property
    def avg(self):
        return self.sum/self.count
    

def get_class_name(obj: object) -> str:
    return obj.__class__.__name__