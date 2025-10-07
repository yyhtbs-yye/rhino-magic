import torch
import torch.nn.functional as F

from rhino.boats.gan.base_gan_boat import BaseGanBoat

class InfoGanBoat(BaseGanBoat):

    def __init__(self, config={}):
        super().__init__(config=config)

        info_cfg = (boat_config or {}).get('infogan', {})
        self.cat_dim = int(info_cfg.get('cat_dim', 0))     # number of categorical codes
        self.cont_dim = int(info_cfg.get('cont_dim', 0))   # number of continuous codes
        self.cont_range = float(info_cfg.get('cont_range', 1.0))  # sample U(-range, range)

        # Q loss weights
        self.q_cat_weight = float(info_cfg.get('q_cat_weight', 1.0))
        self.q_cont_weight = float(info_cfg.get('q_cont_weight', 1.0))

    # -----------------------------
    # Code sampling (simple + tidy)
    # -----------------------------
    def _sample_codes(self, batch_size: int, device: torch.device):
        """
        Returns:
          c_cat_idx:  (N,) long or None
          c_cat_1h:   (N, cat_dim) or None
          c_cont:     (N, cont_dim) or None
        """
        c_cat_idx = None
        c_cat_1h = None
        c_cont = None

        if self.cat_dim > 0:
            c_cat_idx = torch.randint(self.cat_dim, (batch_size,), device=device)    # (N,)
            c_cat_1h = F.one_hot(c_cat_idx, num_classes=self.cat_dim).float()        # (N, cat_dim)

        if self.cont_dim > 0:
            c_cont = (torch.rand(batch_size, self.cont_dim, device=device) * 2.0 - 1.0) * self.cont_range

        return c_cat_idx, c_cat_1h, c_cont

    # -----------------------------
    # Inference path (EMA-aware)
    # -----------------------------
    def predict(self, noise: torch.Tensor) -> torch.Tensor:
        net = self.models['net_ema'] if self.use_ema and 'net_ema' in self.models else self.models['net']
        bsz, device = noise.size(0), noise.device
        _, c_cat, c_cont = self._sample_codes(bsz, device)
        return net(noise, c_cat=c_cat, c_cont=c_cont)

    # -----------------------------
    # Discriminator step losses
    # -----------------------------
    def d_step_calc_losses(self, batch):
        gt = batch['gt']
        bsz, device = gt.size(0), self.device

        # Sample latent + codes
        z = self.noise_generator.next(bsz, device=device)
        c_idx, c_1h, c_cont = self._sample_codes(bsz, device)

        # Generate fakes (stop grad to G)
        with torch.no_grad():
            x_fake = self.models['net'](z, c_cat=c_1h, c_cont=c_cont)

        # Critic on real and fake
        d_real_out = self.models['critic'](gt)
        d_fake_out = self.models['critic'](x_fake)

        d_real = d_real_out['adv']
        d_fake = d_fake_out['adv']

        # Adversarial loss (same interface as BaseGanBoat)
        d_loss = self.losses['critic'](d_real, d_fake)

        # ---- Q losses on fakes (update Q/D) ----
        if self.cat_dim > 0:
            q_cat_loss = self._q_cat_loss(d_fake_out['q_cat_logits'], c_idx)
            d_loss = d_loss + self.q_cat_weight * q_cat_loss

        if self.cont_dim > 0:
            q_cont_loss = self._q_cont_loss(d_fake_out['q_cont_mu'], d_fake_out['q_cont_logvar'], c_cont)
            d_loss = d_loss + self.q_cont_weight * q_cont_loss

        return d_loss
    
    # -----------------------------
    # Generator step losses
    # -----------------------------
    def g_step_calc_losses(self, batch):
        gt = batch['gt']
        bsz, device = gt.size(0), self.device

        # Sample latent + codes
        z = self.noise_generator.next(bsz, device=device)
        c_idx, c_1h, c_cont = self._sample_codes(bsz, device)

        # Generate fakes (G needs grad)
        x_fake = self.models['net'](z, c_cat=c_1h, c_cont=c_cont)

        # Critic feedback on fakes (critic params are frozen by BaseGanBoat.g_step)
        d_fake_out = self.models['critic'](x_fake)
        d_fake_for_g = d_fake_out['adv']

        # Adversarial G loss (scaled as in BaseGanBoat)
        g_loss = self.losses['critic'](d_fake_for_g, None) * self.adversarial_weight

        # ---- Q losses on fakes (encourage G to encode codes) ----
        if self.cat_dim > 0:
            q_cat_loss_g = self._q_cat_loss(d_fake_out['q_cat_logits'], c_idx)
            g_loss = g_loss + self.q_cat_weight * q_cat_loss_g

        if self.cont_dim > 0:
            q_cont_loss_g = self._q_cont_loss(d_fake_out['q_cont_mu'], d_fake_out['q_cont_logvar'], c_cont)
            g_loss = g_loss + self.q_cont_weight * q_cont_loss_g

        return g_loss

    @staticmethod
    def _q_cat_loss(logits: torch.Tensor, target_idx: torch.Tensor) -> torch.Tensor:
        """Categorical code loss: cross-entropy between predicted q(c|x) and true indices."""
        return F.cross_entropy(logits, target_idx)

    @staticmethod
    def _q_cont_loss(mu: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Continuous code loss: Gaussian negative log-likelihood (up to a constant).
        0.5 * [ (x - μ)^2 * exp(-logσ^2) + logσ^2 ]
        """
        return 0.5 * ((target - mu).pow(2) * torch.exp(-logvar) + logvar).mean()
