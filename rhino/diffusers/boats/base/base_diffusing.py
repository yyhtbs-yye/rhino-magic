from trainer.boats.basic_boat import BaseBoat
from trainer.utils.build_components import build_module, build_modules, build_optimizer, build_lr_scheduer

from trainer.visualizers.basic_visualizer import visualize_image_dict

from copy import deepcopy

class BaseDiffusionBoat(BaseBoat):

    def __init__(self, boat_config=None, optimization_config=None, validation_config=None):
        super().__init__()
        
        self.models['model'] = build_module(boat_config['model'])
        self.models['scheduler'] = build_module(boat_config['scheduler'])
        self.models['solver'] = build_module(boat_config['solver'])

        self.boat_config = boat_config
        self.optimization_config = optimization_config
        self.validation_config = validation_config
        self.use_ema = self.optimization_config.get('use_ema', False)
        
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

    def _step(self, loss):
        # Manually optimize
        for optimizer_name, optimizer in self.optimizers.items():
            optimizer.zero_grad()
            self.manual_backward(loss)
            optimizer.step()

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

    def _log_metrics(self, results, prefix=''):

        self.trainer.logger.log_metrics(results, step=self.get_global_step(), prefix=prefix)

    def _log_metric(self, result, metric_name, prefix=''):
        
        self.trainer.logger.log_metrics({metric_name: result}, step=self.get_global_step(), prefix=prefix)

    def _visualize_validation(self, named_imgs, batch_idx, num_vis_samples=4, first_batch_only=True, texts=None):

        if not self.trainer.save_images:
            return

        """Visualize validation results."""
        if first_batch_only and batch_idx == 0:
            # Limit the number of samples to visualize
            for key in named_imgs.keys():
                if named_imgs[key].shape[0] > num_vis_samples:
                    named_imgs[key] = named_imgs[key][:num_vis_samples]
            
            # Log visualizations to the experiment tracker
            visualize_image_dict(
                logger=self.trainer.logger,
                images_dict=named_imgs,
                keys=list(named_imgs.keys()),
                global_step=self.get_global_step(),
                wnb=(0.5, 0.5),
                prefix='val',
                texts=texts,
            )
        
    def get_global_step(self):
        return self.trainer.global_step