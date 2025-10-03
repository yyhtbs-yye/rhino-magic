import torch

from rhino.boats.gan.base_gan_boat import BaseGanBoat

class GanGradientPenaltyBoat(BaseGanBoat):
    def __init__(self, config={}):
        super().__init__(config=config)
        self.lambda_gp = 10

    def d_step_calc_losses(self, batch):

        gt = batch['gt']
        batch_size = gt.size(0)
        
        # Initialize random noise in latent space
        noise = self.noise_generator.next(batch_size, device=self.device)

        # Generate fake samples with current G (no grad to G)
        with torch.no_grad():
            x_fake = self.models['net'](noise)

        # Forward D on real & fake (under autocast if enabled)
        d_real, d_fake = self.models['critic'](gt), self.models['critic'](x_fake)
        # GPT suggest remove "* self.adversarial_weight"
        d_loss = self.losses['critic'](d_real, d_fake) 

        gp = self._gradient_penalty(gt, x_fake)
        d_loss = d_loss + self.lambda_gp * gp

        return d_loss

    def _gradient_penalty(self, real, fake):
        """
        real, fake: [B, ...] tensors on the same device.
        Computes WGAN-GP with ||grad||_2 → 1 penalty.
        """
        device = real.device
        bsz = real.size(0)
        # shape like [B, 1, 1, ...] to broadcast over non-batch dims
        eps = torch.rand(bsz, *([1] * (real.dim() - 1)), device=device)
        # use detached fake; we don't want grads to G
        interpolates = eps * real + (1 - eps) * fake.detach()
        interpolates.requires_grad_(True)

        # do GP in fp32 to avoid half precision issues
        with torch.cuda.amp.autocast(enabled=False):
            d_inter = self.models['critic'](interpolates.float())
            # reduce to per-sample scalar if needed
            if d_inter.dim() > 1:
                d_inter = d_inter.view(d_inter.size(0), -1).mean(dim=1)

            grads = torch.autograd.grad(
                outputs=d_inter.sum(),
                inputs=interpolates,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]

            grads = grads.view(grads.size(0), -1)
            gp = ((grads.norm(2, dim=1) - 1.0) ** 2).mean()

        return gp
