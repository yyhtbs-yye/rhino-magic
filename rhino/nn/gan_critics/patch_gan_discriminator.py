import torch, torch.nn as nn

def init_weights(m, kind="normal", gain=0.02):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        if   kind == "normal":   nn.init.normal_(m.weight, 0.0, gain)
        elif kind == "xavier":   nn.init.xavier_normal_(m.weight, gain=gain)
        elif kind == "kaiming":  nn.init.kaiming_normal_(m.weight, a=0.2, mode="fan_in", nonlinearity="leaky_relu")
        elif kind == "orth":     nn.init.orthogonal_(m.weight, gain)
        if m.bias is not None: nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
        if getattr(m, "weight", None) is not None: nn.init.ones_(m.weight)
        if getattr(m, "bias",   None) is not None: nn.init.zeros_(m.bias)

class PatchGanDiscriminator(nn.Module):
    """
    N-layer PatchGAN (pix2pix/CycleGAN style).
    Example: n_layers=3 -> C64(s2)->C128(s2)->C256(s2)->C512(s1)->C1(s1)
    """
    def __init__(self, in_ch=3, base_ch=64, n_layers=3, norm="instance",
                 spectral=False, init="normal", gain=0.02,
                 reduce='mean'):
        super().__init__()
        Norm = {"instance": lambda c: nn.InstanceNorm2d(c, affine=True, track_running_stats=False),
                "batch":    lambda c: nn.BatchNorm2d(c),
                None:       lambda c: nn.Identity()}[norm if not spectral else None]  # typically no norm with SN

        def conv(c_in, c_out, stride, use_norm=True):
            bias = not use_norm
            conv = nn.Conv2d(c_in, c_out, 4, stride, 1, bias=bias)
            if spectral: conv = nn.utils.spectral_norm(conv)
            layers = [conv]
            if use_norm: layers.append(Norm(c_out))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        # first block (no norm)
        layers += conv(in_ch, base_ch, 2, use_norm=False)
        nf = base_ch
        # pyramid
        for i in range(1, n_layers):
            nf_prev, nf = nf, min(base_ch * 2**i, 512)
            layers += conv(nf_prev, nf, 2, use_norm=True)
        # one conv at stride 1
        nf_prev, nf = nf, min(nf * 2, 512)
        layers += conv(nf_prev, nf, 1, use_norm=True)
        # final 1-channel logit map (no norm/act)
        last = nn.Conv2d(nf, 1, 4, 1, 1)
        if spectral: last = nn.utils.spectral_norm(last)
        layers.append(last)

        self.net = nn.Sequential(*layers)
        self.apply(lambda m: init_weights(m, init, gain))

        self.reduce = reduce

    def forward(self, x):  # -> N×1×H'×W'

        y = self.net(x)

        if self.reduce is None or self.reduce == 'none':
            return y
        elif self.reduce == 'mean':
            return y.mean(dim=(1, 2, 3)) 
        elif self.reduce == 'instance':
            return y.mean(dim=(2,3)).squeeze(1)  # -> (N,)

# --- quick smoke test ---
if __name__ == "__main__":
    D = PatchGanDiscriminator(in_ch=3, n_layers=3, spectral=True, norm=None)
    x = torch.randn(2, 3, 256, 256)
    y = D(x).mean(dim=(1, 2, 3))
    print(y.shape)  # e.g., torch.Size([2, 1, 30, 30])
