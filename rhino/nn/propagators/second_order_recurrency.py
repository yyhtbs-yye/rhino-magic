import torch
import torch.nn as nn

class SecondOrderRecurrency(nn.Module):
    
    def __init__(self, channels=64, extractor=None, 
                 pre_align=None, post_align=None, 
                 is_reversed=False):

        super().__init__()

        self.channels = channels

        self.is_reversed = is_reversed

        self.extractor = extractor

        self.pre_align = pre_align
        
        if self.pre_align is not None:
            self.pre_align_fc = nn.Conv2d(3 * channels, channels, kernel_size=1)

        self.post_align = post_align

    def forward(self, curr_feats, flows):

        n, t, _, h, w = curr_feats.size()

        feat_indices = list(range(-1, -t - 1, -1)) \
                                if self.is_reversed \
                                    else list(range(t))

        history_feats = [curr_feats[:, feat_indices[0], ...], curr_feats[:, feat_indices[0], ...]]
        history_flows = [flows.new_zeros(n, 2, h, w), flows.new_zeros(n, 2, h, w)]

        out_feats = []

        for i in range(0, t):
            
            x = curr_feats[:, feat_indices[i], ...]
            y2, y1 = history_feats
            f2, f1 = history_flows

            if self.pre_align is not None:
                a1 = self.pre_align(y1, f1.permute(0, 2, 3, 1), x)
                f2_c = f1 + self.pre_align(f2, f1.permute(0, 2, 3, 1))
                a2 = self.pre_align(y2, f2_c.permute(0, 2, 3, 1), x)
                comb = torch.cat([a2, a1, x], 1) # Concat on Channel Dim

                in_feat = self.pre_align_fc(comb)
            else:
                in_feat = x
            
            out_feat = self.extractor(in_feat) + x

            if self.post_align is not None:
                out_feat = self.post_align(x, [y1, y2, out_feat], [f1, f2])

            out_feats.append(out_feat.clone())

            if i < t - 1:  # Update history only if not the last iteration
                history_feats = [history_feats[1], out_feat]
                history_flows = [history_flows[1], flows[:, feat_indices[i], ...]]

        if self.is_reversed:
            out_feats = out_feats[::-1]

        return torch.stack(out_feats, dim=1)