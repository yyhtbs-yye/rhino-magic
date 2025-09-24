import torch
import torch.nn as nn
import torch.nn.functional as F

class VQVAELoss(nn.Module):
    def __init__(self):
        super(VQVAELoss, self).__init__()

    def forward(self, prediction, target):
        # Compute element-wise squared error
        recon_loss = F.mse_loss(prediction['sample'], target)

        return recon_loss + prediction['vq_loss']
