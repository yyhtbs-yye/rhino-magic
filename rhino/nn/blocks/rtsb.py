import torch
import torch.nn as nn

class RTSB(nn.Module):
    """
    Residual Swin-style Transformer Block (single stage).
    Args:
      block_cls   : SwinTransformerBlockND subclass.
      embed_dim   : feature dimension C.
      depth       : number of blocks in this stage.
      num_heads   : heads in this stage.
      window_size : size of each attention window.
      mlp_ratio    : hidden/embedding ratio in the MLP.
      qkv_bias     : whether to use bias in QKV projections.
    """
    def __init__(self,
                 block_cls,
                 embed_dim,
                 depth,
                 num_heads,
                 window_size,
                 mlp_ratio,
                 qkv_bias):
        super().__init__()
        self.blocks = nn.ModuleList()
        shift_size = tuple(ws // 2 for ws in window_size)
        for i in range(depth):
            self.blocks.append(
                block_cls(
                    dim         = embed_dim,
                    num_heads   = num_heads,
                    window_size = window_size,
                    shift_size  = shift_size,
                    mlp_ratio   = mlp_ratio,
                    qkv_bias    = qkv_bias,
                    shifted     = (i % 2 == 1)
                )
            )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        x: (B, *T, H, W, C)
        """
        res = x
        for blk in self.blocks:
            x = blk(x)
        x = x + res
        x = self.norm(x)
        return x
