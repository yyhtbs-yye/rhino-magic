import torch
import torch.nn as nn

class WeightedMSELoss(nn.Module):
    def __init__(self):
        super(WeightedMSELoss, self).__init__()

    def forward(self, prediction, target, weights):
        # Compute element-wise squared error
        error = prediction - target
        squared_error = error ** 2

        # Mean over dimensions 1, 2, 3 (e.g., channel, height, width)
        loss_per_sample = torch.mean(squared_error, dim=[1, 2, 3])

        # Weight each sample and compute overall mean
        weighted_loss = torch.mean(loss_per_sample * weights)

        return weighted_loss
