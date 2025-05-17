import os
from datetime import datetime
import torch
from torch.utils.tensorboard import SummaryWriter

class TensorBoardLogger:
    def __init__(self, log_dir="tensorboard_logs", name=None, version=None, default_hp_metric=False):
        """
        Initialize TensorBoard logger similar to PyTorch Lightning's implementation.
        
        Args:
            log_dir (str): Directory for storing logs
            name (str, optional): Name for this experiment
            version (str or int, optional): Version of the experiment
            default_hp_metric (bool): Whether to log a placeholder metric for hparams
        """
        self.log_dir = log_dir
        self.name = name or ""
        self.default_hp_metric = default_hp_metric
        
        # Handle versioning
        if version is None:
            self.version = datetime.now().strftime('%Y%m%d_%H%M%S')
        else:
            self.version = version
            
        # Create the log directory path
        if self.name:
            self.log_dir = os.path.join(log_dir, self.name, f"version_{self.version}")
        else:
            self.log_dir = os.path.join(log_dir, f"version_{self.version}")
            
        # Create the actual writer
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.hparams = {}
        
    def log_hyperparams(self, params):
        """
        Log hyperparameters.
        
        Args:
            params (dict or argparse.Namespace): The hyperparameters to log
        """
        # Convert to dict if it's not already
        if not isinstance(params, dict):
            params = vars(params)
            
        self.hparams.update(params)
        
        # Log with a default metric if requested
        if self.default_hp_metric:
            self.writer.add_hparams(self.hparams, {"hp_metric": 0})
        else:
            self.writer.add_hparams(self.hparams, {})
    
    def log_metrics(self, metrics, step=None, prefix=""):
        """
        Log metrics dictionary.
        
        Args:
            metrics (dict): Dictionary with metric names as keys and values as values
            step (int, optional): Global step value
            prefix (str, optional): Prefix to add to each metric name
        """
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
                
            tag = f"{prefix}/{k}" if prefix else k
            self.writer.add_scalar(tag, v, step)                
            
    def log_image(self, tag, img_tensor, step=None, dataformats='CHW'):
        """
        Log images to tensorboard.
        
        Args:
            tag (str): Name of the image
            img_tensor (torch.Tensor or numpy.ndarray): Image data
            step (int, optional): Global step value
        """
        if isinstance(img_tensor, torch.Tensor):
            img_tensor = img_tensor.detach().cpu()

        self.writer.add_image(tag, img_tensor, step, dataformats=dataformats)
    
    def log_histogram(self, tag, values, step=None, bins='tensorflow'):
        """
        Log histogram to tensorboard.
        
        Args:
            tag (str): Name of the histogram
            values (torch.Tensor or numpy.ndarray): Values to build histogram
            step (int, optional): Global step value
            bins (str): Binning strategy
        """
        self.writer.add_histogram(tag, values, step, bins=bins)
    
    def log_graph(self, model, input_array=None):
        """
        Log model graph.
        
        Args:
            model (torch.nn.Module): Model to log graph of
            input_array (torch.Tensor, optional): Example input to the model
        """
        if input_array is not None:
            self.writer.add_graph(model, input_array)
        else:
            # Try to create a dummy input based on model
            try:
                # This is a very simplistic approach, might not work for all models
                first_param = next(model.parameters())
                dummy_input = torch.zeros((1,) + first_param.size()[1:], 
                                        device=first_param.device)
                self.writer.add_graph(model, dummy_input)
            except Exception as e:
                print(f"Could not log model graph: {e}")
    
    def close(self):
        """Close the logger."""
        self.writer.close()
    
    def __del__(self):
        """Ensure writer is closed on deletion."""
        try:
            self.close()
        except:
            pass

    # In TensorBoardLogger
    def flush(self):
        self.writer.flush()
