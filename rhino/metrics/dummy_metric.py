import torch

class DummyMetric:
    def __init__(self):
        pass

    def __call__(self, x, y):

        return torch.tensor(1.0)
