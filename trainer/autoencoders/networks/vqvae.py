import torch
import torch.nn as nn
import torch.nn.functional as F

# Vector Quantizer Layer
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, beta):
        super(VectorQuantizer, self).__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta
        
        # Initialize embeddings
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)
    
    def forward(self, z):
        # z shape: (batch, embedding_dim, height, width)
        # Reshape z for easier calculations
        z_flattened = z.permute(0, 2, 3, 1).contiguous()  # (batch, height, width, embedding_dim)
        flat_z = z_flattened.view(-1, self.embedding_dim)  # (batch*height*width, embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_z**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_z, self.embedding.weight.t()))
                    
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)  # (batch*height*width, 1)
        
        # Convert encodings to one-hot
        encodings = torch.zeros(encoding_indices.size(0), self.num_embeddings, device=z.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize the latents
        quantized = torch.matmul(encodings, self.embedding.weight).view(z_flattened.size())
        
        # Compute the VQ Losses
        # Commitment loss
        q_latent_loss = F.mse_loss(quantized.detach(), z_flattened)
        # Codebook loss
        e_latent_loss = F.mse_loss(quantized, z_flattened.detach())
        # Total loss
        loss = q_latent_loss + self.beta * e_latent_loss
        
        # Preserve gradients
        quantized = z_flattened + (quantized - z_flattened).detach()
        
        # Return values
        return quantized.permute(0, 3, 1, 2).contiguous(), loss, encoding_indices.view(z.size(0), -1)

# VQVAE Model
class VQVAE(nn.Module):
    def __init__(self, embedding_dim, num_embeddings, beta):
        super(VQVAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, embedding_dim, kernel_size=1, stride=1)
        )
        
        # Vector Quantizer
        self.vq = VectorQuantizer(num_embeddings, embedding_dim, beta)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        z = self.encoder(x)
        return z
    
    def quantize(self, z):
        quantized, vq_loss, indices = self.vq(z)
        return quantized, vq_loss, indices
    
    def decode(self, quantized):
        x_recon = self.decoder(quantized)
        return x_recon
    
    def forward(self, x):
        z = self.encode(x)
        quantized, vq_loss, indices = self.quantize(z)
        x_recon = self.decode(quantized)
        return {'sample': x_recon, 
                'vq_loss': vq_loss, 
                'indices': indices}
    
    def encode_indices(self, x):
        """
        Encode input images to discrete indices
        """
        z = self.encode(x)
        _, _, indices = self.quantize(z)
        return indices
    
    def decode_indices(self, indices, h=7, w=7):
        """
        Decode from discrete indices to images
        """
        batch_size = indices.size(0)
        
        # Convert indices to one-hot encodings
        encodings = torch.zeros(batch_size*h*w, self.vq.num_embeddings, device=indices.device)
        encodings.scatter_(1, indices.reshape(-1, 1), 1)
        
        # Convert one-hot to embeddings
        quantized = torch.matmul(encodings, self.vq.embedding.weight)
        quantized = quantized.view(batch_size, h, w, self.vq.embedding_dim)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        
        # Decode
        x_recon = self.decode(quantized)
        return x_recon
