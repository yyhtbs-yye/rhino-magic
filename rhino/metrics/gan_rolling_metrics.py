# pip install torchmetrics==1.*  clean-fid==0.1.*
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance

class RollingFrechetInceptionDistance:
    def __init__(self):
        self.fid = None

    @torch.no_grad()
    def add_real(self, imgs: torch.Tensor):
        # imgs in [0,1]; convert to uint8 expected by torchmetrics (0..255)
        imgs_uint8 = (imgs.clamp(0,1) * 255).byte()
        self.fid.update(imgs_uint8, real=True)

    @torch.no_grad()
    def add_fake(self, imgs: torch.Tensor):
        imgs_uint8 = (imgs.clamp(0,1) * 255).byte()
        self.fid.update(imgs_uint8, real=False)

    @torch.no_grad()
    def compute(self):            
        fid_val = self.fid.compute().item()
        return fid_val

    def __call__(self, fakes, reals):
        if self.fid is None:
            self.fid = FrechetInceptionDistance(feature=2048).to(fakes.device)

        self.add_real(reals)
        self.add_fake(fakes)

        return self.compute()

    def reset(self):
        if self.fid is not None:
            self.fid.reset()

class RollingKernelInceptionDistance:
    def __init__(self):
        self.kid = None

    @torch.no_grad()
    def add_real(self, imgs: torch.Tensor):
        # imgs in [0,1]; convert to uint8 expected by torchmetrics (0..255)
        imgs_uint8 = (imgs.clamp(0,1) * 255).byte()
        self.kid.update(imgs_uint8, real=True)
        self.device = imgs.device

    @torch.no_grad()
    def add_fake(self, imgs: torch.Tensor):
        imgs_uint8 = (imgs.clamp(0,1) * 255).byte()
        self.kid.update(imgs_uint8, real=False)
        self.device = imgs.device

    @torch.no_grad()
    def compute(self):
        if self.kid is None:
            self.kid = KernelInceptionDistance(subset_size=1000).to(self.device)
            
        kid_mean, kid_std = self.kid.compute()
        return {"kid_mean": kid_mean.item(), "kid_std": kid_std.item()}

    def reset(self):
        self.kid.reset()
