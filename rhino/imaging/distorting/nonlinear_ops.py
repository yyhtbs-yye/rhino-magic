
# Non-linear operator base class
class NonLinearOperator(nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, data, **kwargs):
        pass

    def project(self, data, measurement, **kwargs):
        return data + measurement - self.forward(data, **kwargs)
    

# Non-linear operator implementations
class PhaseRetrievalOperator(NonLinearOperator):
    def __init__(self, oversample = 1.5):
        super().__init__()
        self.pad = int((oversample / 8.0) * 256)

    @classmethod
    def from_config(cls, config) -> 'PhaseRetrievalOperator':
        params = config.get('params', {})
        return cls(oversample=params.get('oversample', 1.5)
        )

    def forward(self, data, **kwargs):
        padded = F.pad(data, (self.pad,) * 4)
        return torch.fft.fft2(padded).abs()

class NonlinearBlurOperator(NonLinearOperator):
    def __init__(self, model_path: str = None):
        super().__init__()
        self.model_path = model_path
        self.model = torch.nn.Conv2d(3, 3, 3, padding=1).to() if model_path is None else None

    @classmethod
    def from_config(cls, config) -> 'NonlinearBlurOperator':
        params = config.get('params', {})
        return cls(model_path=params.get('model_path')
        )

    def forward(self, data, **kwargs):
        data = (data + 1.0) / 2.0  # [-1, 1] -> [0, 1]
        blurred = self.model(data) if self.model is not None else F.avg_pool2d(data, 3, stride=1, padding=1)
        return (blurred * 2.0 - 1.0).clamp(-1, 1)