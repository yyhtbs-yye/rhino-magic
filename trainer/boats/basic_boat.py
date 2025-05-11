import torch

from abc import ABC, abstractmethod
from trainer.utils.state_load_save import save_state, load_state

class BaseBoat(ABC):
    """
    Abstract base class for model containers to be used with the Trainer.
    
    A "Boat" represents a container for models, optimizers, and training logic,
    but is not itself a nn.Module.
    """

    def __init__(self):
        """
        Initialize the BaseBoat.
        
        """
        self.models = {}  # Dictionary to hold PyTorch models
        self.losses = {}  # Dictionary to hold PyTorch Loss functions
        self.optimizers = {}  # Dictionary to hold optimizers
        self.lr_schedulers = {}  # Dictionary to hold learning rate lr_schedulers
        self.device = None

    def to(self, device):
        """
        Move all models and metrics to the specified device.
        
        Args:
            device: The device to move the models to (e.g., 'cuda:3', 'cpu')
            
        Returns:
            self: The boat with models on the specified device
        """
        self.device = device
        for name, model in self.models.items():
            if hasattr(model, 'to'):
                self.models[name] = model.to(device)
        if hasattr(self, 'metrics'):
            for name, metric in self.metrics.items():
                if hasattr(metric, 'to'):
                    self.metrics[name] = metric.to(device)
        # Move optimizer states to the same device
        self.move_optimizer_to_device(device)
        return self

    def move_optimizer_to_device(self, device):
        """
        Explicitly move optimizer state tensors to the specified device.
        
        Args:
            device: The device to move the optimizer states to (e.g., 'cuda:3', 'cpu')
            
        Returns:
            None
        """
        for name, optim in self.optimizers.items():
            # Move all state tensors to the specified device
            for param in optim.state:
                state = optim.state[param]
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state[key] = value.to(device)
            # Update param_groups to ensure they reference parameters on the correct device
            for group in optim.param_groups:
                group['params'] = [p.to(device) for p in group['params']]

    def parameters(self):
        for model_name, model in self.models.items():
            if hasattr(model, 'parameters'):
                for param in model.parameters():
                    yield param

    @abstractmethod
    def training_step(self, batch, batch_idx):
        """
        Define the training step.
        
        This method should:
        1. Run the forward pass on necessary models
        2. Calculate the loss
        3. Run backward pass and optimizer steps
        
        Args:
            batch: The input batch
            batch_idx: Index of the current batch
            
        Returns:
            loss: The loss value for this batch
        """
        pass

    @abstractmethod
    def validation_step(self, batch, batch_idx):
        """
        Define the validation step.
        
        Args:
            batch: The input batch
            batch_idx: Index of the current batch
            
        Returns:
            Dictionary with validation metrics
        """
        pass

    @abstractmethod
    def configure_optimizers(self):
        """
        Configure optimizers and learning rate lr_schedulers.
        
        This method should set:
        - self.optimizers: Dictionary of optimizers
        - self.lr_schedulers: Dictionary of lr_schedulers (optional)
        
        Returns:
            None
        """
        pass

    def save_state(self, run_folder, prefix="boat_state", global_step=None, epoch=None):
        return save_state(run_folder, prefix, boat=self, global_step=global_step, epoch=epoch)

    def load_state(self, state_path, strict=True):
        return load_state(state_path, boat=self, strict=strict)
    
    def lr_scheduling_step(self):
        """
        Step all learning rate lr_schedulers.
        
        Called after each training step.
        
        Returns:
            None
        """
        for scheduler_name, scheduler in self.lr_schedulers.items():
            scheduler.step()
    
    def train(self):
        """
        Set all models to training mode.
        
        Returns:
            self
        """
        for model_name, model in self.models.items():
            if hasattr(model, 'train'):
                model.train()
        return self
    
    def eval(self):
        """
        Set all models to evaluation mode.
        
        Returns:
            self
        """
        for model_name, model in self.models.items():
            if hasattr(model, 'eval'):
                model.eval()
        return self
    
    def manual_backward(self, loss):
        """
        Perform manual backward pass, delegating to trainer if available.
        
        This method supports both standalone usage and integration with 
        distributed training frameworks by checking for a trainer object.
        
        Args:
            loss: The loss tensor to backpropagate
            
        Returns:
            None
        """
        if hasattr(self, 'trainer') and hasattr(self.trainer, 'backward'):
            # Delegate to trainer's backward implementation
            # This handles distributed training, mixed precision, etc.
            self.trainer.backward(loss)
        else:
            # Fallback to standard PyTorch backward
            loss.backward()