import torch
import torch.nn.functional as F

from rhino.boats.base.base_boat import BaseBoat
from trainer.utils.build_components import build_module
from trainer.utils.ddp_utils import move_to_device

class BaseAutoregressionBoat(BaseBoat):
    """
    Base class for *spatial autoregressive* generation over discrete VQ latents.
    - Assumes a pretrained latent encoder/decoder (VAE without adversarial loss).
    - Assumes a VQ module that maps continuous latents <-> discrete code indices.
    - The AR `net` models a raster-ordered sequence of code indices.
    """

    def __init__(self, config={}):
        super().__init__(config=config)

        assert config is not None, "main config must be provided"

        # Core components
        self.models['net'] = build_module(boat_config['net'])                  # AR transformer/PixelCNN prior over codes
        self.models['latent_encoder'] = build_module(boat_config['latent_encoder']) if 'latent_encoder' in boat_config else None  # pretrained VAE (encoder/decoder)
        self.models['vq'] = build_module(boat_config['vq']) if 'vq' in boat_config else None  # vector-quantizer module (codebook)

        # make sure VQ and latent encoder are not trained
        if self.models['latent_encoder'] is not None:
            for param in self.models['latent_encoder'].parameters():
                param.requires_grad = False
        if self.models['vq'] is not None:
            for param in self.models['vq'].parameters():
                param.requires_grad = False

        self.boat_config = boat_config or {}
        self.optimization_config = optimization_config or {}
        self.validation_config = validation_config or {}

        self.use_ema = self.optimization_config.get('use_ema', False)
        self.use_reference = self.validation_config.get('use_reference', False)

        # Sampling hyperparams
        samp_cfg = self.boat_config.get('sampling', {})
        self.temperature = float(samp_cfg.get('temperature', 1.0))
        self.top_k = int(samp_cfg.get('top_k', 0))
        self.top_p = float(samp_cfg.get('top_p', 1.0))
        self.sample_method = samp_cfg.get('method', 'categorical')  # 'categorical' or 'argmax'

        # Token ids
        self.bos_token = int(self.boat_config.get('bos_token', 0))
        self.pad_token = int(self.boat_config.get('pad_token', -100))  # used only for loss masking / placeholder

        if self.use_ema:
            self._setup_ema()
            self.ema_start = self.optimization_config.get('ema_start', 0)

    # ------------------------------- Training -------------------------------

    def training_calc_losses(self, batch):
        """
        Teacher-forced AR training on raster-ordered code indices.
        Loss is cross-entropy over the codebook vocabulary.
        """
        gt = batch['gt']  # (B, C, H, W)
        with torch.no_grad():
            z = self.encode_images(gt)                    # (B, C_l, H_l, W_l)
            codes = self.latent_to_codes(z)               # (B, Ht, Wt) long

        B, Ht, Wt = codes.shape
        N = Ht * Wt
        targets = codes.view(B, N)                        # (B, N)

        # Shifted inputs: [BOS, x_0, x_1, ..., x_{N-2}]
        bos = torch.full((B, 1), self.bos_token, dtype=torch.long, device=targets.device)
        inputs = torch.cat([bos, targets[:, :-1]], dim=1)  # (B, N)

        # Forward AR net
        out = self.models['net'](inputs)
        logits = out.get('logits', out.get('sample', out)) if isinstance(out, dict) else out  # (B, N, V)

        # Delegate to configured loss (expects keys: 'preds', 'targets', optional 'weights')
        train_output = {'preds': logits, 'targets': targets, **batch}
        net_loss = self.losses['net'](train_output)

        losses = {'total_loss': net_loss, 'net': net_loss}
        return losses

    # ------------------------------- Inference -------------------------------

    @torch.no_grad()
    def generate_codes(self, batch_size, grid_hw, cond=None):
        """
        Autoregressively sample a (Ht, Wt) grid of code indices.
        Args:
            batch_size: int
            grid_hw: tuple (Ht, Wt)
            cond: optional conditioning dict passed through to net if it accepts it
        Returns:
            codes: LongTensor (B, Ht, Wt)
        """
        Ht, Wt = grid_hw
        N = Ht * Wt

        net = self.models['net_ema'] if self.use_ema and 'net_ema' in self.models else self.models['net']

        # Tokens buffer
        device = next(net.parameters()).device
        tokens = torch.empty((batch_size, N), dtype=torch.long, device=device)

        # First input position uses BOS
        input_tokens = torch.full((batch_size, N), self.pad_token, dtype=torch.long, device=device)
        input_tokens[:, 0] = self.bos_token

        for i in range(N):
            if i > 0:
                input_tokens[:, 1:i+1] = tokens[:, :i]

            out = net(input_tokens) if cond is None else net(input_tokens, cond=cond)
            logits = out.get('logits', out.get('sample', out)) if isinstance(out, dict) else out  # (B, N, V)
            step_logits = logits[:, i, :] / max(self.temperature, 1e-8)                           # (B, V)

            # Top-k / top-p filtering
            step_logits = self._top_k_top_p_filtering(step_logits, top_k=self.top_k, top_p=self.top_p)

            if self.sample_method == 'argmax':
                next_tok = step_logits.argmax(dim=-1)
            else:
                probs = step_logits.softmax(dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1).squeeze(-1)                    # (B,)

            tokens[:, i] = next_tok

        return tokens.view(batch_size, Ht, Wt)

    def predict(self, reference_batch):
        """
        Generate images with the same latent grid size as the reference batch's latents.
        `reference_batch` can be:
            - a tensor of images (B, C, H, W) -> we'll infer (Ht, Wt) by encoding once
            - or a dict containing 'gt' images
        """
        if isinstance(reference_batch, dict) and 'gt' in reference_batch:
            imgs = reference_batch['gt']
        else:
            imgs = reference_batch

        with torch.no_grad():
            z = self.encode_images(imgs)
            # Infer code grid size without building the actual codes (cheaper than VQ sometimes)
            # But to be robust, just go through VQ once.
            codes_ref = self.latent_to_codes(z)
            Ht, Wt = codes_ref.shape[-2:]

            codes = self.generate_codes(imgs.size(0), (Ht, Wt))
            z_q = self.codes_to_latent(codes)
            x_hat = self.decode_latents(z_q)
            return torch.clamp(x_hat, -1, 1)

    # ------------------------------- Validation -------------------------------

    def validation_step(self, batch, batch_idx):
        batch = move_to_device(batch, self.device)
        gt = batch['gt']

        with torch.no_grad():
            # Generate with matching latent grid size
            x_hat = self.predict({'gt': gt})

            valid_output = {'preds': x_hat, 'targets': gt}
            metrics = self._calc_metrics(valid_output)
            named_imgs = {'groundtruth': gt, 'generated': x_hat}

        return metrics, named_imgs

    # ------------------------------- Utils -------------------------------

    @staticmethod
    def _top_k_top_p_filtering(logits, top_k=0, top_p=1.0):
        """
        Filter a distribution of logits using top-k and/or nucleus (top-p) filtering.
        """
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            vals, _ = torch.topk(logits, top_k)
            cut = vals[..., -1, None]
            logits = torch.where(logits < cut, torch.full_like(logits, float('-inf')), logits)

        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
            probs = sorted_logits.softmax(dim=-1)
            cumprobs = probs.cumsum(dim=-1)

            # mask tokens with cumulative prob above threshold
            mask = cumprobs > top_p
            # ensure at least one token is kept
            mask[..., 0] = False
            # map back to original indices
            mask_scatter = torch.zeros_like(mask, dtype=torch.bool)
            mask_scatter.scatter_(dim=-1, index=sorted_idx, src=mask)
            logits = torch.where(mask_scatter, torch.full_like(logits, float('-inf')), logits)

        return logits


    # ------------------------------- Required encode/decode helpers -------------------------------

    def encode_images(self, x):
        """
        Encode images to continuous latent space using the pretrained VAE encoder.
        """
        if self.models.get('latent_encoder') is None:
            return x
        return self.models['latent_encoder'].encode(x)

    def decode_latents(self, z_q):
        """
        Decode continuous latents back to pixel space using the VAE decoder.
        """
        if self.models.get('latent_encoder') is None:
            return z_q
        return self.models['latent_encoder'].decode(z_q)

    def latent_to_codes(self, z):
        """
        Convert continuous latents to discrete VQ code indices (B, Ht, Wt) [long].
        Tries common VQ module APIs: encode_to_codes, encode, quantize.
        """
        vq = self.models.get('vq', None)
        if vq is None:
            raise RuntimeError("VQ module not provided in boat_config['vq']")

        # Preferred explicit API
        if hasattr(vq, 'encode_to_codes'):
            codes = vq.encode_to_codes(z)
            return codes.long()

        # Common taming/vqvae-style: encode -> (codes) or (z_q, codes)
        if hasattr(vq, 'encode'):
            out = vq.encode(z)
            if isinstance(out, (list, tuple)):
                # e.g., (z_q, codes) or (codes,)
                for item in out[::-1]:
                    if torch.is_tensor(item) and item.dtype == torch.long:
                        return item
                # last resort: if first item is indices-like
                if torch.is_tensor(out[0]) and out[0].dtype == torch.long:
                    return out[0]
            if torch.is_tensor(out) and out.dtype == torch.long:
                return out.long()

        # Another pattern: quantize -> (z_q, codes)
        if hasattr(vq, 'quantize'):
            out = vq.quantize(z)
            if isinstance(out, (list, tuple)) and len(out) >= 2:
                codes = out[1]
                return codes.long()

        raise AttributeError("VQ module must implement encode_to_codes(z)->codes, encode(z)->codes, or quantize(z)->(z_q,codes).")

    def codes_to_latent(self, codes):
        """
        Convert discrete code indices (B, Ht, Wt) back to quantized continuous latents.
        Tries common VQ APIs: decode_from_codes, decode, dequantize.
        """
        vq = self.models.get('vq', None)
        if vq is None:
            raise RuntimeError("VQ module not provided in boat_config['vq']")

        if hasattr(vq, 'decode_from_codes'):
            return vq.decode_from_codes(codes)

        if hasattr(vq, 'decode'):
            return vq.decode(codes)

        if hasattr(vq, 'dequantize'):
            return vq.dequantize(codes)

        raise AttributeError("VQ module must implement decode_from_codes(codes), decode(codes), or dequantize(codes).")

