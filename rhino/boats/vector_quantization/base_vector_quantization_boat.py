import torch

from rhino.boats.base.base_boat import BaseBoat
from trainer.utils.ddp_utils import move_to_device

class BaseVectorQuantizationBoat(BaseBoat):

    def __init__(self, config={}):
        super().__init__(config=config)

        # Optional freezing behavior
        if self.models["latent_encoder"] is not None and self.boat_config.get("freeze_latent_encoder", True):
            for p in self.models["latent_encoder"].parameters():
                p.requires_grad = False

        # Weight for the (optional) VQ/codebook loss
        self.vq_weight = float(self.boat_config.get("vq_weight", 1.0))

    def predict(self, inputs):
        z_hat = self.dequantize_codes(inputs)
        x_hat = self.decode_latents(z_hat)
        return torch.clamp(x_hat, -1, 1)

    def training_calc_losses(self, batch):

        gt = batch["gt"]

        with torch.no_grad():
            latents = self.encode_images(gt)

        z_hat, codes, *others = self.models['vector_quantizer'](latents)

        # Compute AR loss (e.g., CE/L2/etc. defined in config as self.losses['net'])
        train_output = {"preds": z_hat, "targets": latents, **batch}

        net_loss = self.losses["net"](train_output)

        total_loss = net_loss

        losses = {
            "total_loss": total_loss,
            "net": net_loss,
        }

        return losses

    def validation_step(self, batch, batch_idx):
        """
        Report metrics in data space: encode -> (optional) VQ (for logging) -> predict from zeros -> decode -> metrics.
        """
        batch = move_to_device(batch, self.device)
        gt = batch["gt"]

        with torch.no_grad():
            latents = self.encode_images(gt)

            z_hat, codes, *others = self.models['vector_quantizer'](latents)

            x_hat = self.decode_latents(z_hat)

            valid_output = {"preds": x_hat, "targets": gt}
            metrics = self._calc_metrics(valid_output)

            named_imgs = {"groundtruth": gt, "generated": x_hat}

        return metrics, named_imgs

    # ----------------------------
    # Latent helpers
    # ----------------------------
    def decode_latents(self, z):
        """Decode latents to the data space using the provided latent encoder/decoder."""
        return self.models["latent_encoder"].decode(z)

    def encode_images(self, x):
        """Encode data to latent space using the provided latent encoder/decoder."""
        return self.models["latent_encoder"].encode(x)

    def quantize_latents(self, z):

        c = self.models["vector_quantizer"].quantize(z)
        return c

    def dequantize_codes(self, c: torch.Tensor) -> torch.Tensor:
        """
        indices (LongTensor) -> continuous latents using the codebook.
        """
        if hasattr(self.models["vector_quantizer"], "embed_code"):      # VQ-VAE style
            return self.models["vector_quantizer"].embed_code(c)
        if hasattr(self.models["vector_quantizer"], "dequantize"):      # some libs call it dequantize()
            return self.models["vector_quantizer"].dequantize(c)
        if hasattr(self.models["vector_quantizer"], "lookup"):          # or lookup()
            return self.models["vector_quantizer"].lookup(c)
        raise RuntimeError(
            "Vector quantizer does not expose an embedding/lookup method "
            "(tried: embed_code/dequantize/lookup)."
        )
