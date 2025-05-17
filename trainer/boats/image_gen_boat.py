from trainer.boats.basic_boat import BaseBoat
from trainer.utils.build_components import build_module, build_modules, build_optimizer, build_lr_scheduer

from trainer.visualizers.basic_visualizer import visualize_image_dict
from copy import deepcopy

class ImageGenerationBoat(BaseBoat):
    def __init__(self, boat_config=None, optimization_config=None, validation_config=None):
        super().__init__()
        
        # Build the model
        self.models['model'] = build_module(boat_config['model'])
        
        # Store configurations
        self.boat_config = boat_config
        self.optimization_config = optimization_config
        self.validation_config = validation_config
        self.use_ema = self.optimization_config.get('use_ema', False)
        
        # Setup EMA if enabled
        if self.use_ema:
            self._setup_ema()

    def attach_trainer(self, trainer):
        self.trainer = trainer

    def configure_optimizers(self):
        self.optimizers['model'] = build_optimizer(self.models['model'].parameters(), self.optimization_config['model'])
        if 'lr_scheduler' in self.optimization_config['model'] and len(self.optimization_config['model'].get('lr_scheduler', {})) > 0:
            self.lr_schedulers['model'] = build_lr_scheduer(self.optimizers['model'], 
                                                           self.optimization_config['model'].get('lr_scheduler', {}))

    def configure_losses(self):
        self.losses['model'] = build_module(self.boat_config.get('loss', {}).get('model', None))

    def configure_metrics(self):
        self.metrics = build_modules(self.validation_config.get('metrics', {}))


    def _setup_ema(self):
        """Set up Exponential Moving Average (EMA) model."""
        if isinstance(self.use_ema, bool):
            self.use_ema = {'ema_decay': 0.999, 'ema_start': 1000}

        if self.use_ema:
            self.models['model_ema'] = deepcopy(self.models['model'])
            for param in self.models['model_ema'].parameters():
                param.requires_grad = False
    
        self.ema_decay = self.use_ema.get('ema_decay', 0.999)
        self.ema_start = self.use_ema.get('ema_start', 1000)

    def _update_ema(self):
        """Update EMA model parameters."""
        if not self.use_ema: return

        for ema_param, param in zip(self.models['model_ema'].parameters(), self.models['model'].parameters()):
            ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)
        for ema_buffer, buffer in zip(self.models['model_ema'].buffers(), self.models['model'].buffers()):
            ema_buffer.data.copy_(buffer.data)

    def _log_metric(self, value, name, prefix=''):
        self.trainer.logger.log_metrics({name: value}, step=self.get_global_step(), prefix=prefix)

    def get_global_step(self):
        return self.trainer.global_step

    def _step(self, loss):
        # Manually optimize
        for optimizer_name, optimizer in self.optimizers.items():
            optimizer.zero_grad()
            self.manual_backward(loss)
            optimizer.step()

    def _visualize_validation(self, images_dict, batch_idx, num_vis=4):
        """Logs up to `num_vis` images under the 'val' prefix."""
        # trim samples if necessary
        for k, v in images_dict.items():
            if v.size(0) > num_vis:
                images_dict[k] = v[:num_vis]
        visualize_image_dict(
            logger=self.trainer.logger,
            images_dict=images_dict,
            keys=list(images_dict),
            global_step=self.get_global_step(),
            wnb=(0.5, 0.5),
            prefix='val'
        )
