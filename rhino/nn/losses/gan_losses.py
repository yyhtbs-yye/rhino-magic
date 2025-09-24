import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod

# -----------------------------
# Base interface + reduction
# -----------------------------

class _ReduceMixin:
    def __init__(self, reduction: str = "mean"):
        if reduction not in ("mean", "sum", "none"):
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")
        self.reduction = reduction

    def _reduce(self, x: torch.Tensor) -> torch.Tensor:
        if self.reduction == "mean":
            return x.mean()
        elif self.reduction == "sum":
            return x.sum()
        return x

class GANLossBase(nn.Module, ABC, _ReduceMixin):
    """
    Interface:
      D step: forward(d_real, d_fake)
      G step: forward(d_fake, None)
    """
    def __init__(self, reduction: str = "mean"):
        nn.Module.__init__(self)
        _ReduceMixin.__init__(self, reduction=reduction)

    @abstractmethod
    def forward(self, a: torch.Tensor, b: torch.Tensor | None) -> torch.Tensor:
        ...


# -----------------------------
# Hinge loss (spectral-norm D common)
# D: E[relu(1 - D(real))] + E[relu(1 + D(fake))]
# G: -E[D(fake)]
# -----------------------------

class HingeGANLoss(GANLossBase):
    def forward(self, a: torch.Tensor, b: torch.Tensor | None) -> torch.Tensor:
        if b is None:  # generator
            return self._reduce(-a)
        # discriminator
        d_real, d_fake = a, b
        loss_real = F.relu(1.0 - d_real)
        loss_fake = F.relu(1.0 + d_fake)
        return self._reduce(loss_real + loss_fake)


# -----------------------------
# Non-saturating logistic (BCE-with-logits)
# D: softplus(-D(real)) + softplus(D(fake))
# G: softplus(-D(fake))
# Optional label smoothing via real_label/fake_label.
# -----------------------------

class BCEGANLoss(GANLossBase):
    def __init__(self, real_label: float = 1.0, fake_label: float = 0.0, reduction: str = "mean"):
        super().__init__(reduction=reduction)
        self.real_label = real_label
        self.fake_label = fake_label
        self._fast_labels = (real_label == 1.0 and fake_label == 0.0)

    def forward(self, a: torch.Tensor, b: torch.Tensor | None) -> torch.Tensor:
        if b is None:  # generator
            if self._fast_labels:
                return self._reduce(F.softplus(-a))  # -log(sigmoid(a))
            target = torch.full_like(a, self.real_label)
            return self._reduce(F.binary_cross_entropy_with_logits(a, target, reduction='none'))
        # discriminator
        d_real, d_fake = a, b
        if self._fast_labels:
            loss_real = F.softplus(-d_real)  # -log(sigmoid(d_real))
            loss_fake = F.softplus(d_fake)   # -log(1 - sigmoid(d_fake))
        else:
            t_real = torch.full_like(d_real, self.real_label)
            t_fake = torch.full_like(d_fake, self.fake_label)
            loss_real = F.binary_cross_entropy_with_logits(d_real, t_real, reduction='none')
            loss_fake = F.binary_cross_entropy_with_logits(d_fake, t_fake, reduction='none')
        return self._reduce(loss_real + loss_fake)


class LogisticGANLoss(GANLossBase):
    """StyleGAN2 'logistic' (non-saturating) loss.
    - Generator:     E[ softplus(-D(G(z))) ]
    - Discriminator: E[ softplus( D(G(z)) ) + softplus( -D(x) ) ]
    """
    def __init__(self, real_label: float = 1.0, fake_label: float = 0.0, reduction: str = "mean"):
        super().__init__(reduction=reduction)
        self.real_label = real_label
        self.fake_label = fake_label
        self._fast_labels = (real_label == 1.0 and fake_label == 0.0)

    def forward(self, a: torch.Tensor, b: torch.Tensor | None) -> torch.Tensor:
        if b is None:  # generator step, 'a' are D logits on fakes
            if self._fast_labels:
                return self._reduce(F.softplus(-a))  # -log(sigmoid(a))
            target = torch.full_like(a, self.real_label)
            return self._reduce(F.binary_cross_entropy_with_logits(a, target, reduction="none"))

        # discriminator step, 'a' are logits on real, 'b' on fake
        d_real, d_fake = a, b
        if self._fast_labels:
            loss_real = F.softplus(-d_real)  # -log(sigmoid(d_real))
            loss_fake = F.softplus(d_fake)   # -log(1 - sigmoid(d_fake))
        else:
            t_real = torch.full_like(d_real, self.real_label)
            t_fake = torch.full_like(d_fake, self.fake_label)
            loss_real = F.binary_cross_entropy_with_logits(d_real, t_real, reduction="none")
            loss_fake = F.binary_cross_entropy_with_logits(d_fake, t_fake, reduction="none")
        return self._reduce(loss_real + loss_fake)


# -----------------------------
# WGAN (use GP/R1/clipping externally)
# D: E[D(fake)] - E[D(real)]
# G: -E[D(fake)]
# -----------------------------

class WGANLoss(GANLossBase):
    def forward(self, a: torch.Tensor, b: torch.Tensor | None) -> torch.Tensor:
        if b is None:  # generator
            return self._reduce(-a)
        d_real, d_fake = a, b
        return self._reduce(d_fake - d_real)


# -----------------------------
# LSGAN (least squares)
# D: 0.5*((D(real)-y_real)^2 + (D(fake)-y_fake)^2)
# G: 0.5*((D(fake)-y_real)^2)
# -----------------------------

class LSGANLoss(GANLossBase):
    def __init__(self, real_label: float = 1.0, fake_label: float = 0.0, reduction: str = "mean"):
        super().__init__(reduction=reduction)
        self.real_label = real_label
        self.fake_label = fake_label

    def forward(self, a: torch.Tensor, b: torch.Tensor | None) -> torch.Tensor:
        if b is None:  # generator
            target = torch.full_like(a, self.real_label)
            return self._reduce(0.5 * (a - target).pow(2))
        d_real, d_fake = a, b
        t_real = torch.full_like(d_real, self.real_label)
        t_fake = torch.full_like(d_fake, self.fake_label)
        loss = 0.5 * ((d_real - t_real).pow(2) + (d_fake - t_fake).pow(2))
        return self._reduce(loss)
