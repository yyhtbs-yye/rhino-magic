from rhino.nn.autoregression.pixelcnn.pixelcnn import PixelCNN

# -------------------------------
# (Optional) minimal factory for build_module-style configs
# -------------------------------
def build_pixelcnn_from_config(cfg: dict) -> PixelCNN:
    """
    Expects:
      cfg = {
        "target": "path.to.PixelCNN",   # ignored here, kept for compatibility
        "params": {
            "vocab_size": int,
            "in_channels": int,
            "hidden_channels": 128,
            "n_layers": 8,
            "kernel_size": 3,
            "embed_dim": 64,
            "dropout": 0.0,
        }
      }
    """
    params = cfg.get("params", cfg)
    return PixelCNN(**params)
