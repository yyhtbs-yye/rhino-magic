from rhino.boats.base.base_boat import BaseBoat
from trainer.utils.build_components import build_module, build_modules, build_optimizer, build_lr_scheduer

from trainer.visualizers.basic_visualizer import visualize_image_dict
from copy import deepcopy

class ImageGenerationBoat(BaseBoat):
    def __init__(self, boat_config=None, optimization_config=None, validation_config=None):
        super().__init__()
        
        # Build the model
        self.models['net'] = build_module(boat_config['net'])
        
        # Store configurations
        self.boat_config = boat_config
        self.optimization_config = optimization_config
        self.validation_config = validation_config
        self.use_ema = self.optimization_config.get('use_ema', False)
        
        # Setup EMA if enabled
        if self.use_ema:
            self._setup_ema()

    def configure_optimizers(self):
        self.optimizers['net'] = build_optimizer(self.models['net'].parameters(), self.optimization_config['net'])
        if 'lr_scheduler' in self.optimization_config['net'] and len(self.optimization_config['net'].get('lr_scheduler', {})) > 0:
            self.lr_schedulers['net'] = build_lr_scheduer(self.optimizers['net'], 
                                                           self.optimization_config['net'].get('lr_scheduler', {}))

    def configure_losses(self):
        self.losses['net'] = build_module(self.boat_config.get('loss', {}).get('net', None))

    def configure_metrics(self):
        self.metrics = build_modules(self.validation_config.get('metrics', {}))


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