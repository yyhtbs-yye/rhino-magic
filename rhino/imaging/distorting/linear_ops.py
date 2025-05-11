import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Linear operator base class
class LinearOperator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, data):
        return data

    def transpose(self, data):
        return data

    def ortho_project(self, data):
        return data - self.transpose(self.forward(data))

    def project(self, data, measurement):
        return self.ortho_project(measurement) - self.forward(data)
    
    @classmethod
    def from_config(cls, config):
        params = config.get('params', {})
        return cls(**params)

class SuperResolutionOperator(LinearOperator):
    def __init__(self, scale_factor=4.0, mode='bicubic', ):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, data):
        return F.interpolate(data, scale_factor=1/self.scale_factor, mode=self.mode)

    def transpose(self, data):
        return F.interpolate(data, scale_factor=self.scale_factor, mode=self.mode)

class GaussianBlurOperator(LinearOperator):
    def __init__(self, kernel_size = 9, intensity = 1.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = intensity
        self.kernel = self._gaussian_kernel(kernel_size, intensity).to()

    def _gaussian_kernel(self, size, sigma):
        x = torch.arange(-size // 2 + 1, size // 2 + 1)
        x, y = torch.meshgrid(x, x, indexing='ij')
        kernel = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()  # Normalize
        return kernel.view(1, 1, size, size)

    def forward(self, data):
        return F.conv2d(data, self.kernel, padding=self.kernel_size // 2, groups=data.shape[1])

    def transpose(self, data):
        return data

class MotionBlurOperator(LinearOperator):
    def __init__(self, kernel_size=9, angle=0.0, intensity=1.0, device='cpu'):
        """Initialize MotionBlurOperator.
        
        Args:
            kernel_size (int): Size of the square convolution kernel (odd number).
            angle (float): Angle of motion in degrees (0=horizontal, 90=vertical).
            intensity (float): Strength of the blur effect (scales kernel weights).
            device (str): Device to store kernel ('cpu' or 'cuda').
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.angle = angle
        self.intensity = intensity
        self.device = device
        self.kernel = self._generate_motion_kernel().to(device)

    def _generate_motion_kernel(self):
        """Generate a motion blur kernel based on size, angle, and intensity."""
        size = self.kernel_size
        center = size // 2
        kernel = torch.zeros(size, size)

        # Convert angle to radians
        angle_rad = math.radians(self.angle)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        # Calculate the endpoints of the motion line
        length = size // 2
        x0 = center - length * cos_a
        y0 = center - length * sin_a
        x1 = center + length * cos_a
        y1 = center + length * sin_a

        # Draw a line in the kernel using Bresenham's line algorithm
        points = self._bresenham_line(x0, y0, x1, y1, size)
        for x, y in points:
            if 0 <= x < size and 0 <= y < size:
                kernel[x, y] = 1.0

        # Normalize kernel and apply intensity
        kernel_sum = kernel.sum()
        if kernel_sum > 0:
            kernel = kernel * (self.intensity / kernel_sum)
        
        return kernel.view(1, 1, size, size)

    def _bresenham_line(self, x0, y0, x1, y1, size):
        points = []
        x0, y0, x1, y1 = map(round, [x0, y0, x1, y1])
        
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            points.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        
        return points

    def forward(self, data):

        kernel = self.kernel.repeat(data.shape[1], 1, 1, 1)
        return F.conv2d(data, kernel, padding=self.kernel_size // 2, groups=data.shape[1])

    def transpose(self, data):
        # Transpose of convolution is convolution with 180-degree rotated kernel
        flipped_kernel = torch.flip(self.kernel, dims=[2, 3])
        return F.conv2d(data, flipped_kernel, padding=self.kernel_size // 2, groups=data.shape[1])

class InpaintingOperator(LinearOperator):
    def __init__(self, bbox):
        super().__init__()
        self.bbox = bbox  # [x1, y1, x2, y2]

    def forward(self, data, mask):
        if mask is None and self.bbox is not None:
            mask = torch.ones_like(data)
            x1, y1, x2, y2 = self.bbox
            mask[:, :, y1:y2, x1:x2] = 0  # Mask out the bounding box
        if mask is None:
            raise ValueError("Inpainting requires a mask or bbox.")
        return data * mask

    def transpose(self, data):
        return data
