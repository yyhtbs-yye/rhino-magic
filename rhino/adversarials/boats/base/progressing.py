from copy import deepcopy
import torch
from trainer.boats.image_gen_boat import ImageGenerationBoat
from trainer.utils.build_components import build_module, build_optimizer, build_lr_scheduer, build_modules

class PGGANBoat(ImageGenerationBoat):
    """
    Progressive Growing GAN boat using epoch-based stage progression.
    Expects boat_config to contain:
      - generator: config for the generator module
      - discriminator: config for the discriminator module
      - loss: {generator: ..., discriminator: ...}
      - noise_dim: latent dimension for z
      - progressive: {stages: [...], epochs_per_stage: int}
    """
    def __init__(self, boat_config=None, optimization_config=None, validation_config=None):
        super().__init__(boat_config, optimization_config, validation_config)
        # Build generator and discriminator
        self.models['generator'] = build_module(boat_config['generator'])
        self.models['discriminator'] = build_module(boat_config['discriminator'])

        # Loss modules
        self.losses['generator'] = build_module(boat_config.get('loss', {}).get('generator', None))
        self.losses['discriminator'] = build_module(boat_config.get('loss', {}).get('discriminator', None))

        # Progressive training schedule (epoch-based)
        prog_cfg = boat_config.get('progressive', {})
        self.stages = prog_cfg.get('stages', [4,8,16,32,64,128,256,512,1024])
        self.epochs_per_stage = prog_cfg.get('epochs_per_stage', 10)
        self.current_stage = 0

    def forward(self, z):
        """
        Generate images from latent z at the current resolution stage.
        """
        gen = self.models['generator']
        return gen(z, stage=self.stages[self.current_stage])

    def training_step(self, batch, batch_idx):
        real = batch['gt']
        bs = real.size(0)
        device = real.device

        # Sample latent vectors
        z = torch.randn(bs, self.boat_config['noise_dim'], device=device)
        # Generate fake images
        fake = self.forward(z)

        # Discriminator step
        real_pred = self.models['discriminator'](real, stage=self.stages[self.current_stage])
        fake_pred = self.models['discriminator'](fake.detach(), stage=self.stages[self.current_stage])
        d_loss = self.losses['discriminator'](real_pred, fake_pred)
        self._step(d_loss)
        self._log_metric(d_loss, name='d_loss', prefix='train')

        # Generator step
        fake_pred_g = self.models['discriminator'](fake, stage=self.stages[self.current_stage])
        g_loss = self.losses['generator'](fake_pred_g)
        self._step(g_loss)
        self._log_metric(g_loss, name='g_loss', prefix='train')

        # EMA update for generator
        if self.use_ema and self.get_global_step() > self.ema_start:
            self._update_ema()

        return {'d_loss': d_loss, 'g_loss': g_loss}

    def training_epoch_end(self, epoch):
        # Progress to next stage at epoch boundaries
        if epoch > 0 and epoch % self.epochs_per_stage == 0:
            self.current_stage = min(self.current_stage + 1, len(self.stages) - 1)

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            bs = batch['gt'].size(0)
            # Sample latents and generate
            z = torch.randn(bs, self.boat_config['noise_dim'], device=batch['gt'].device)
            # Use EMA generator if available
            gen_net = self.models.get('model_ema', None) or self.models['generator']
            fake = gen_net(z, stage=self.stages[self.current_stage])

            # Visualization
            named_imgs = {'groundtruth': batch['gt'], 'generated': fake}
            self._visualize_validation(named_imgs, batch_idx)

        return None

    def configure_optimizers(self):
        # Override to set optimizers for both G and D
        self.optimizers['generator'] = build_optimizer(
            self.models['generator'].parameters(), self.optimization_config['generator'])
        self.optimizers['discriminator'] = build_optimizer(
            self.models['discriminator'].parameters(), self.optimization_config['discriminator'])

        # LR schedulers if any
        if 'generator' in self.optimization_config and 'lr_scheduler' in self.optimization_config['generator']:
            self.lr_schedulers['generator'] = build_lr_scheduer(
                self.optimizers['generator'], self.optimization_config['generator']['lr_scheduler'])
        if 'discriminator' in self.optimization_config and 'lr_scheduler' in self.optimization_config['discriminator']:
            self.lr_schedulers['discriminator'] = build_lr_scheduer(
                self.optimizers['discriminator'], self.optimization_config['discriminator']['lr_scheduler'])
