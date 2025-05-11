import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure

def setup_metrics(metrics_config, class_ref=None):
    """Set up metrics based on validation config for evaluation.

    Args:
        metrics_config (dict): Configuration dictionary containing metrics settings.
        class_ref: Reference to the class containing the get_class method for dynamic loading.

    Returns:
        torch.nn.ModuleDict: Dictionary of initialized metrics.
    """
    # Initialize metrics as a ModuleDict for PyTorch compatibility
    metrics = torch.nn.ModuleDict()
    
    # Setup FID if configured or by default
    if metrics_config.get('use_fid', True):
        fid_config = metrics_config.get('fid', {'feature': 2048, 'normalize': True})
        metrics['fid'] = FrechetInceptionDistance(**fid_config)
    
    # Setup PSNR if configured or by default
    if metrics_config.get('use_psnr', True):
        metrics['psnr'] = PeakSignalNoiseRatio(data_range=2.0)
    
    # Setup SSIM if configured or by default
    if metrics_config.get('use_ssim', True):
        metrics['ssim'] = StructuralSimilarityIndexMeasure(data_range=2.0)
    
    # Setup custom metrics if provided
    for metric_name, metric_config in metrics_config.get('custom_metrics', {}).items():
        try:
            metric_class = class_ref.get_class(
                metric_config['path'], metric_config['name']
            )
            metrics[metric_name] = metric_class(**metric_config['config'])
        except (ImportError, AttributeError, KeyError) as e:
            raise ValueError(f"Failed to initialize custom metric '{metric_name}': {str(e)}")
    
    return metrics