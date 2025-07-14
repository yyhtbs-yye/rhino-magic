import torch
from typing import Optional

from diffusers.models import DiTTransformer2DModel


# ---------------------------------------------------------------------
# 1.  Concatenation-based conditioning for DiT
# ---------------------------------------------------------------------
class DiTConcatConditionModel(DiTTransformer2DModel):
    """
    DiT variant that conditions by **channel-wise concatenation**.

    Args
    ----
    sample_size         : Spatial resolution of the input (same as DiTTransformer2DModel).
    sample_channels     : #channels in the diffusion sample (e.g. 4 for latent-space DiT).
    condition_channels  : #channels in the conditioning tensor.  If ``None`` or ``0``,
                          this acts exactly like the base DiT.
    **kwargs            : Any other kwargs accepted by ``DiTTransformer2DModel.__init__``.
    """

    def __init__(
        self,
        sample_size: int = 32,
        sample_channels: Optional[int] = None,
        condition_channels: Optional[int] = None,
        **kwargs,
    ):
        # --- remember user-supplied channel counts --------------------
        self.sample_channels     = sample_channels or kwargs.get("in_channels", 4)
        self.condition_channels  = condition_channels or 0

        # --- build correct ``in_channels`` for the parent -------------
        in_channels = self.sample_channels + self.condition_channels
        # avoid sending a duplicate in_channels into **kwargs
        kwargs.pop("in_channels", None)

        super().__init__(in_channels=in_channels, sample_size=sample_size, **kwargs)

    # -----------------------------------------------------------------
    # forward
    # -----------------------------------------------------------------
    def forward(
        self,
        sample:  torch.FloatTensor,
        timestep: torch.IntTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        class_labels=None,
        **kwargs,
    ):
        """
        If ``encoder_hidden_states`` is supplied, we concatenate it with the sample
        on the **channel** dimension before running the DiT backbone.
        """
        if encoder_hidden_states is not None:
            if sample.shape[2:] != encoder_hidden_states.shape[2:]:
                raise ValueError(
                    "Spatial dimensions must match in concat mode: "
                    f"sample {tuple(sample.shape[2:])} vs "
                    f"condition {tuple(encoder_hidden_states.shape[2:])}"
                )
            sample = torch.cat([sample, encoder_hidden_states], dim=1)

        # provide zeros if user didn’t pass labels
        if class_labels is None:
            class_labels = torch.zeros(
                sample.shape[0],
                dtype=torch.long,
                device=sample.device,
            )

        return super().forward(
            sample,
            timestep,
            class_labels=class_labels,
            **kwargs,
        )


# ---------------------------------------------------------------------
# 2.  Factory / wrapper
# ---------------------------------------------------------------------
class DiT2DFlexibleWrapper:
    """
    Tiny convenience factory that mirrors your UNet helper.

    ``mode`` in the config determines what gets built:
        - "none"   : plain   ``DiTTransformer2DModel``
        - "concat" : custom  ``DiTConcatConditionModel``

    Cross-attention is **not** currently implemented for DiT in diffusers,
    so any other mode raises.
    """

    SUPPORTED_MODES = {"none", "concat"}

    @classmethod
    def from_config(cls, config: dict):
        # ---- pull bookkeeping options --------------------------------
        mode               = config.pop("mode", "none")
        sample_channels    = config.pop("sample_channels", None)
        condition_channels = config.pop("condition_channels", None)

        # ---- sanity --------------------------------------------------
        if mode not in cls.SUPPORTED_MODES:
            raise ValueError(
                f"Unsupported mode '{mode}'. "
                f"Choose from {sorted(cls.SUPPORTED_MODES)}."
            )

        # ---- 'concat'  ----------------------------------------------
        if mode == "concat":
            return DiTConcatConditionModel(
                sample_channels=sample_channels,
                condition_channels=condition_channels,
                **config,
            )

        # ---- 'none' (baseline DiT) ----------------------------------
        # Strip any conditioning-only keys the user might have provided.
        config.pop("condition_channels", None)
        config["in_channels"] = sample_channels  # override / ensure

        return DiTTransformer2DModel(**config)
