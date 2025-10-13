import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import _calculate_correct_fan

class EqualizedLR:
    def __init__(self, name='weight', gain=2**0.5, mode='fan_in', lr_mul=1.0):
        self.name = name
        self.mode = mode
        self.gain = gain
        self.lr_mul = lr_mul

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        if weight.ndim == 5:
            fan = _calculate_correct_fan(weight[0], self.mode)
        else:
            assert weight.ndim <= 4
            fan = _calculate_correct_fan(weight, self.mode)
        weight = weight * torch.tensor(self.gain, device=weight.device) \
                 * torch.sqrt(torch.tensor(1. / fan, device=weight.device)) \
                 * self.lr_mul
        return weight

    def __call__(self, module, inputs):
        setattr(module, self.name, self.compute_weight(module))

    @staticmethod
    def apply(module, name, gain=2**0.5, mode='fan_in', lr_mul=1.):
        for _, hook in module._forward_pre_hooks.items():
            if isinstance(hook, EqualizedLR):
                raise RuntimeError(
                    'Cannot register two equalized_lr hooks on the same '
                    f'parameter {name} in {module} module.'
                )
        fn = EqualizedLR(name, gain=gain, mode=mode, lr_mul=lr_mul)
        weight = module._parameters[name]
        delattr(module, name)
        module.register_parameter(name + '_orig', weight)
        setattr(module, name, weight.data)  # keep attribute for init code paths
        module.register_forward_pre_hook(fn)
        return fn

def equalized_lr(module, name='weight', gain=2**0.5, mode='fan_in', lr_mul=1.):
    EqualizedLR.apply(module, name, gain=gain, mode=mode, lr_mul=lr_mul)
    return module