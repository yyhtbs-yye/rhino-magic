import torch
import torch.nn.functional as F

class RasterScanner:
    def __init__(self, dist_type='multinomial'):
        if dist_type == 'multinomial':
            self.sampler = multinomial_sampling
        elif dist_type == 'gaussian':
            self.sampler = gaussian_sampling
        else:
            raise ValueError(f"Unsupported distribution type: {dist_type}")

    def solve(self, network, zeros, seed=None):
        if seed is not None:
            torch.manual_seed(seed)

        _, _, H, W = zeros.shape

        for i in range(H):
            for j in range(W):
                # Get network outputs for current position: [B, C, N, H, W]
                outputs = network(zeros)
                if hasattr(outputs, 'sample'):
                    outputs = outputs.sample

                # outputs[:, :, i, j, :].shape = (B, C, N)
                zeros[:, :, i, j] = self.sampler(outputs[:, :, i, j, :]).float()
                print(i*W + j)

        return zeros
    
def multinomial_sampling(pixel_outputs):
    B, C, n_classes = pixel_outputs.shape
    # probs.shape = [B, C]
    probs = F.softmax(pixel_outputs, dim=2)
    # Sample from multinomial for all channels at once: [B, C]
    return torch.multinomial(probs.view(-1, n_classes), 1).view(B, C) / (n_classes - 1.0)

def gaussian_sampling(pixel_outputs):
    mean = pixel_outputs[:, :, 0]  # [B, C]
    log_var = pixel_outputs[:, :, 1]  # [B, C]
    std = torch.exp(0.5 * log_var)
    # Sample from Gaussian distribution: [B, C]
    epsilon = torch.randn_like(mean)
    return mean + epsilon * std

if __name__ == "__main__":
    # Example usage
    class MockNetwork:

        def __init__(self, B, C, H, W):
            self.t = torch.randn(B, C, 128, H, W)
        def __call__(self, x):
            # Mock output: [B, C, N, H, W] for multinomial
            self.t = self.t + 1
            return self.t

    # Initialize scanner and mock data
    scanner = RasterScanner(dist_type='multinomial')
    zeros = torch.zeros(2, 3, 128, 128)  # [B, C, H, W]
    network = MockNetwork(2, 3, 128, 128)

    # Run solver
    result = scanner.solve(network, zeros, seed=42)
    print("Result:", result)
    print("Result shape:", result.shape)
