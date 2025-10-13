import torch
import torch.nn as nn
import torch.nn.functional as F

class MiniBatchStddevLayer(nn.Module):
    """Minibatch standard deviation.

    Args:
        group_size (int, optional): The size of groups in batch dimension.
            Defaults to 4.
        eps (float, optional):  Epsilon value to avoid computation error.
            Defaults to 1e-8.
    """

    def __init__(self, group_size=4, eps=1e-8):
        super().__init__()
        self.group_size = group_size
        self.eps = eps

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        # batch size should be smaller than or equal to group size. Otherwise,
        # batch size should be divisible by the group size.
        assert x.shape[
            0] <= self.group_size or x.shape[0] % self.group_size == 0, (
                'Batch size be smaller than or equal '
                'to group size. Otherwise,'
                ' batch size should be divisible by the group size.'
                f'But got batch size {x.shape[0]},'
                f' group size {self.group_size}')
        n, c, h, w = x.shape
        group_size = min(n, self.group_size)
        # [G, M, C, H, W]
        y = torch.reshape(x, (group_size, -1, c, h, w))
        # [G, M, C, H, W]
        y = y - y.mean(dim=0, keepdim=True)
        # In pt>=1.7, you can just use `.square()` function.
        # [M, C, H, W]
        y = y.pow(2).mean(dim=0, keepdim=False)
        y = torch.sqrt(y + self.eps)
        # [M, 1, 1, 1]
        y = y.mean(dim=(1, 2, 3), keepdim=True)
        y = y.repeat(group_size, 1, h, w)
        return torch.cat([x, y], dim=1)