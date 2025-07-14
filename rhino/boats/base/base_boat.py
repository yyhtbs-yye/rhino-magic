import torch

from abc import ABC, abstractmethod
from trainer.utils.state_load_save import save_state, load_state

from trainer.utils.build_components import build_module, build_modules, build_optimizer, build_lr_scheduer

from trainer.visualizers.basic_visualizer import visualize_image_dict

from copy import deepcopy

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
    def training_backward(self, batch, batch_idx):
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

    def training_epoch_end(self, epoch):
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
    
    def configure_losses(self):
        for loss_name in self.boat_config.get('loss', {}):
            self.losses[loss_name] = build_module(self.boat_config['loss'][loss_name])         

    def configure_metrics(self):
        self.metrics = build_modules(self.validation_config.get('metrics', {}))

    def configure_optimizers(self):
        for model_name in self.optimization_config:
            if 'use_ema' in model_name:
                continue
            self.optimizers[model_name] = build_optimizer(
                self.models[model_name].parameters(), 
                self.optimization_config[model_name]
            )
        self.configure_lr_scheduers()

    def _calc_losses(self, train_output):
        losses = {'total_loss': torch.tensor(0.0, device=self.device)}

        for loss_name in self.losses:
            loss_fn = self.losses[loss_name]
            loss_val = loss_fn(train_output)
            losses['total_loss'] += loss_val
            losses[loss_name] = loss_val

        return losses

    def _calc_reference_quality_metrics(self, predictions, targets):
        results = {}
        for metric_name, metric in self.metrics.items():
            metric_val = metric(predictions, targets)
            results[metric_name] = metric_val
        return results

    def _calc_noreference_quality_metrics(self, predictions):
        results = {}
        for metric_name, metric in self.metrics.items():
            metric_val = metric(predictions)
            results[metric_name] = metric_val
        return results

    def _calc_metrics(self, valid_output):

        if self.use_reference:
            metrics = self._calc_reference_quality_metrics(valid_output['preds'], valid_output['targets'])
        else:
            metrics = self._calc_noreference_quality_metrics(valid_output['preds'])

        return metrics

    def _backward(self, loss):
        for model_name, optimizer in self.optimizers.items():
            optimizer.zero_grad()
            loss.backward()

    def _step(self):
        for model_name, optimizer in self.optimizers.items():
            optimizer.step()

    def _log_metrics(self, logger, results, prefix=''):

        logger.log_metrics(results, step=self.get_global_step(), prefix=prefix)

    def _log_metric(self, logger, result, metric_name, prefix=''):
        
        logger.log_metrics({metric_name: result}, step=self.get_global_step(), prefix=prefix)

    def log_train_losses(self, logger, losses):
        for loss_name, loss_value in losses.items():
            self._log_metric(logger, loss_value.detach(), metric_name=loss_name, prefix='train')

    def log_valid_metrics(self, logger, metrics):
        for metric_name, metric_value in metrics.items():
            self._log_metric(logger, metric_value.detach(), metric_name=metric_name, prefix='valid')

    def visualize_validation(self, logger, named_imgs, batch_idx, num_vis_samples=4, first_batch_only=True, texts=None):

        """Visualize validation results."""
        if first_batch_only and batch_idx == 0:
            # Limit the number of samples to visualize
            for key in named_imgs.keys():
                if named_imgs[key].shape[0] > num_vis_samples:
                    named_imgs[key] = named_imgs[key][:num_vis_samples]
            
            # Log visualizations to the experiment tracker
            visualize_image_dict(
                logger=logger,
                images_dict=named_imgs,
                keys=list(named_imgs.keys()),
                global_step=self.get_global_step(),
                wnb=(0.5, 0.5),
                prefix='val',
                texts=texts,
            )
        
    def get_global_step(self):
        return self.global_step()
    
    def attach_global_step(self, global_step):
        self.global_step = global_step

    def configure_lr_scheduers(self):

        if 'lr_scheduler' in self.optimization_config['net'] and len(self.optimization_config['net'].get('lr_scheduler', {})) > 0:
            self.lr_schedulers['net'] = build_lr_scheduer(self.optimizers['net'], 
                                                            self.optimization_config['net'].get('lr_scheduler', {}))

    def _setup_ema(self):
        """Set up Exponential Moving Average (EMA) model."""

        if isinstance(self.use_ema, bool):
            self.use_ema = {'ema_decay': 0.999, 'ema_start': 1000}

        if self.use_ema:
            self.models['net_ema'] = deepcopy(self.models['net'])
            for param in self.models['net_ema'].parameters():
                param.requires_grad = False
    
        self.ema_decay = self.use_ema.get('ema_decay', 0.999)
        self.ema_start = self.use_ema.get('ema_start', 1000)

    def _update_ema(self):
        """Update EMA model parameters."""
        if not self.use_ema: return

        for ema_param, param in zip(self.models['net_ema'].parameters(), self.models['net'].parameters()):
            ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)
        for ema_buffer, buffer in zip(self.models['net_ema'].buffers(), self.models['net'].buffers()):
            ema_buffer.data.copy_(buffer.data)
