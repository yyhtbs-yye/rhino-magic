import torch
import torch.nn as nn

# ---------------- Discriminator (real/fake) head ----------------

class BaseDiscHead(nn.Module):
    """
    Input:  (B, C, H, W)
    Output: (B, 1) real/fake logit (no sigmoid)
    """
    def __init__(self, in_channels, hidden_dim=None):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)  # -> (B, C, 1, 1)

        if hidden_dim is None:
            self.mlp = nn.Linear(in_channels, 1)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(in_channels, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 1),
            )

    def forward(self, x):
        x = self.pool(x).flatten(1)  # (B, C)
        return self.mlp(x)           # (B, 1)
