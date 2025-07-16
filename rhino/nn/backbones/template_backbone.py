import torch.nn as nn
from timm.layers import trunc_normal_
from einops import rearrange

from rhino.nn.minimals.sinusoidal_time_embedder import SinusoidalTimeEmbedder

class TemplateBackbone(nn.Module):
    def __init__(self, **args):
        super().__init__()

        if args['enable_condition']:
            self.enable_condition = args['enable_condition']
            cond_dim = args['cond_dim'] or args['embed_dim'] * 4
            self.time_embed = nn.Sequential(
                SinusoidalTimeEmbedder(args['embed_dim']),
                nn.Linear(args['embed_dim'], cond_dim * 4),
                nn.SiLU(),
                nn.Linear(cond_dim * 4, cond_dim),
            ) if args['enable_condition'] else None

        self.pos_drop = nn.Dropout(p=args.get('drop_rate', 0.))

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            trunc_normal_(m.weight, std=0.02)  # ViT-style
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward(self, x, t=None):

        if t is not None and self.enable_condition:
            cond = self.time_embed(t)
        else:
            cond = None 
        x = rearrange(x, 'b c h w -> b h w c')

        x = self.pos_drop(x)

        for stage in self.stages:
            x = stage(x, cond=cond)

        x = rearrange(x, 'b h w c -> b c h w')
        return x
    