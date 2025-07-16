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
                 **args):
        super().__init__()
        self.blocks = nn.ModuleList()

        depth = args.pop('depth', 1)  # Default depth is 1 if not specified
        triggers = args.pop('triggers', None)  # Optional triggers for each block

        if triggers is not None:
            assert len(triggers) == depth, "len(triggers) must equal depth"

        for i in range(depth):
            self.blocks.append(
                block_cls(
                    trigger = triggers[i] if triggers is not None else None,  # Use trigger if provided
                    **args,  # Pass all other arguments to the block
                )
            )
        self.norm = nn.LayerNorm(args['embed_dim'])

    def forward(self, x, cond=None):
        """
        x: (B, *T, H, W, C)
        """
        shortcut = x
        for blk in self.blocks:
            x = blk(x, cond)
        
        x = self.norm(x)

        x = x + shortcut
        return x
