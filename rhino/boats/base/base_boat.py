import torch

from copy import deepcopy

from rhino.boats.base.template_boat import TemplateBoat

from trainer.utils.state_load_save import save_state, load_state
from trainer.utils.build_components import build_module, build_modules, build_optimizer, build_lr_scheduler, build_logger
from trainer.visualizers.basic_visualizer import visualize_image_dict
from trainer.utils.ddp_utils import ddp_no_sync_all, move_to_device

class BaseBoat(TemplateBoat):

    def __init__(self, config={}):

        self.boat_config = config.get('boat') or {}
        self.optimization_config = config.get('optimization') or {}
        self.validation_config = config.get('validation') or {}
        self.logging_config = config.get('logging', {})

        self.total_micro_steps = self.optimization_config.pop('total_micro_steps', 1)
        self.target_loss_key = self.optimization_config.pop('target_loss_key', 'total_loss')

        self.build_models()
        self.build_losses()
        self.build_optimizers()
        self.build_metrics()
        self.build_others()
        self.build_loggers()

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

    def parameters(self):
        for model_name, model in self.models.items():
            if hasattr(model, 'parameters'):
                for param in model.parameters():
                    yield param    

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

    # ------------------------------------ Training Step ---------------------------------------------

    def training_backpropagation(self, loss, current_micro_step, scaler):

        use_no_sync = (self.total_micro_steps > 1) and (current_micro_step < self.total_micro_steps - 1)

        with ddp_no_sync_all(self, enabled=use_no_sync):
            if scaler is not None and getattr(scaler, "is_enabled", lambda: False)():
                scaler.scale(loss).backward()
            else:
                loss.backward()

    def training_gradient_descent(self, scaler, active_keys):

        if scaler is not None and scaler.is_enabled():
            for k in active_keys: scaler.step(self.optimizers[k])
            scaler.update()
        else:
            for k in active_keys: self.optimizers[k].step()

    def training_lr_scheduling_step(self, active_keys):

        for k in active_keys: 
            if k in self.lr_schedulers:
                self.lr_schedulers[k].step()
            
    def training_step(self, batch, batch_idx, epoch, *, scaler=None):
        
        active_keys = list(self.optimizers.keys())
        
        micro_batches = self._split_batch(batch, self.total_micro_steps)

        self._zero_grad(active_keys, set_to_none=True)

        micro_losses_list = []
        for current_micro_step, micro_batch in enumerate(micro_batches):
            micro_batch = move_to_device(micro_batch, self.device)
            micro_losses = self.training_calc_losses(micro_batch)
            micro_target_loss = micro_losses[self.target_loss_key] / self.total_micro_steps
            micro_losses_list.append(micro_losses)
            self.training_backpropagation(micro_target_loss, current_micro_step, scaler)

        self.training_gradient_descent(scaler, active_keys)
        
        self._update_ema()

        self.training_lr_scheduling_step(active_keys)

        return self._aggregate_loss_dicts(micro_losses_list)

    # ------------------------------------ Visualization ---------------------------------------------

    def visualize_validation(self, logger, named_imgs, batch_idx, trainer_config):

        visualization_config = trainer_config.get('visualization', {})

        # Backward compatibility
        if visualization_config.get('save_images', False) or trainer_config.get('save_images', False):

            """Visualize validation results."""
            if visualization_config.get('first_batch_only', True) and batch_idx == 0:
                # Limit the number of samples to visualize
                for key in named_imgs.keys():
                    if named_imgs[key].shape[0] > visualization_config.get('num_vis_samples', 4):
                        named_imgs[key] = named_imgs[key][:visualization_config.get('num_vis_samples')]
                
                wnb = visualization_config.get('wnb', (0.5, 0.5))
                # Log visualizations to the experiment tracker
                visualize_image_dict(
                    logger=logger,
                    images_dict=named_imgs,
                    keys=list(named_imgs.keys()),
                    global_step=self.get_global_step(),
                    wnb=wnb,
                    prefix='val',
                    texts='texts',
                )

    # ------------------------------------ Result Logging ---------------------------------------------

    def _log_metrics(self, logger, results, prefix=''):

        logger.log_metrics(results, step=self.get_global_step(), prefix=prefix)

    def _log_metric(self, logger, result, metric_name, prefix=''):
        
        logger.log_metrics({metric_name: result}, step=self.get_global_step(), prefix=prefix)

    def log_train_losses(self, logger, losses):
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                self._log_metric(logger, loss_value.detach(), metric_name=loss_name, prefix='train')

    def log_valid_metrics(self, logger, metrics):
        for metric_name, metric_value in metrics.items():
            self._log_metric(logger, metric_value.detach(), metric_name=metric_name, prefix='valid')

    # ------------------------------------ Global Step Mgmt ---------------------------------------------

    def get_global_step(self):
        return self.global_step()
    
    def attach_global_step(self, global_step):
        self.global_step = global_step

    # ------------------------------------ Build Component ---------------------------------------------
    def build_models(self):
        for model_name in self.boat_config.get('models', {}):
            new_module = build_module(self.boat_config['models'][model_name])
            if self.models.get(model_name) is None or type(new_module) != type(self.models[model_name]):
                self.models[model_name] = new_module

    def build_losses(self):
        for loss_name in self.boat_config.get('losses', {}):
            new_module = build_module(self.boat_config['losses'][loss_name])
            if self.losses.get(loss_name) is None or type(new_module) != type(self.losses[loss_name]):
                self.losses[loss_name] = new_module
                
    def build_metrics(self):
        self.metrics = build_modules(self.validation_config.get('metrics', {}))

    def build_optimizers(self):
        for model_name in self.optimization_config:
            if 'use_ema' in model_name or model_name == 'hyper_parameters':
                continue
            new_optimizer = build_optimizer(
                self.models[model_name].parameters(), 
                self.optimization_config[model_name]
            )
            if self.optimizers.get(model_name) is None or type(new_optimizer) != type(self.optimizers[model_name]):
                self.optimizers[model_name] = new_optimizer

        self.build_lr_scheduler_by_name(model_name)

    def build_lr_scheduler_by_name(self, model_name):

        if 'lr_scheduler' in self.optimization_config[model_name] and len(self.optimization_config[model_name].get('lr_scheduler', {})) > 0:
            new_lr_schelduler = build_lr_scheduler(self.optimizers[model_name], self.optimization_config[model_name].get('lr_scheduler', {}))
            if self.lr_schedulers.get(model_name) is None or type(new_lr_schelduler) != type(self.lr_schedulers[model_name]):
                self.lr_schedulers[model_name] = new_lr_schelduler
    
    def build_loggers(self):
        self.loggers = {}
        for logger_name in self.logging_config:
            self.loggers[logger_name] = build_logger(self.logging_config[logger_name])

    # ------------------------------------ EMA ---------------------------------------------
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

        if self.use_ema and self.get_global_step() >= self.ema_start:
            """Update EMA model parameters."""
            if not self.use_ema: return

            for ema_param, param in zip(self.models['net_ema'].parameters(), self.models['net'].parameters()):
                ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)
            for ema_buffer, buffer in zip(self.models['net_ema'].buffers(), self.models['net'].buffers()):
                ema_buffer.data.copy_(buffer.data)

    # ------------------------------------ State S/L ---------------------------------------------
    def save_state(self, run_folder, prefix="boat_state", global_step=None, epoch=None):
        return save_state(run_folder, prefix, boat=self, global_step=global_step, epoch=epoch)

    def load_state(self, state_path, strict=True):
        return load_state(state_path, boat=self, strict=strict)

    def move_optimizer_to_device(self, device):        # DO NOT edit optim.param_groups here
        for _, optim in self.optimizers.items():
            # move state tensors
            for state in optim.state.values():
                for k, v in list(state.items()):
                    if torch.is_tensor(v):
                        state[k] = v.to(device)

    # ------------------------------------ Metrics ---------------------------------------------
    def _reset_metrics(self):
        for _, metric in self.metrics.items(): # _: metric_name
            if hasattr(metric, 'reset'):
                metric.reset()

    def _calc_reference_quality_metrics(self, predictions, targets):
        results = {}
        for metric_name, metric in self.metrics.items():
            metric_val = metric(predictions, targets)
            if isinstance(metric_val, dict):
                results.update(metric_val)
            else:
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

    # ------------------------------------ Utilities ---------------------------------------------

    def _zero_grad(self, active_keys, set_to_none=True):
        for key in active_keys:
            self.optimizers[key].zero_grad(set_to_none=set_to_none)


    def _first_tensor_and_size(self, x):
        if torch.is_tensor(x):
            return x, x.shape[0]
        if isinstance(x, dict):
            for v in x.values():
                t, n = self._first_tensor_and_size(v)
                if t is not None:
                    return t, n
        if isinstance(x, (list, tuple)):
            for v in x:
                t, n = self._first_tensor_and_size(v)
                if t is not None:
                    return t, n
        return None, None

    def _split_batch(self, x, parts):
        # returns a list of length `parts` mirroring x's structure
        if torch.is_tensor(x):
            # torch.chunk handles non-divisible sizes
            return list(torch.chunk(x, parts, dim=0))
        if isinstance(x, dict):
            per = [dict() for _ in range(parts)]
            for k, v in x.items():
                chunks = self._split_batch(v, parts)
                for i in range(parts):
                    per[i][k] = chunks[i]
            return per
        if isinstance(x, (list, tuple)):
            elems = [self._split_batch(v, parts) for v in x]
            out = []
            for i in range(parts):
                out.append(type(x)(chunks[i] for chunks in elems))
            return out
        # non-tensor leaf: replicate reference (ok for e.g. scalars/strings)
        return [x for _ in range(parts)]
    
    def _aggregate_loss_dicts(self, loss_dicts):
        """
        Simple (unweighted) mean of each key across a list of micro-batch loss dicts.
        Missing keys are ignored for that key's average.
        Tensor values are detached and converted to floats.
        """
        if not loss_dicts:
            return {}

        keys = set().union(*(d.keys() for d in loss_dicts))
        out = {}

        for k in keys:
            vals = []
            for d in loss_dicts:
                if k not in d or d[k] is None:
                    continue
                v = d[k]
                if torch.is_tensor(v):
                    v = v.detach()
                    v = v.mean() if v.ndim > 0 else v
                else:
                    v = v
                vals.append(v)
            if vals:
                out[k] = sum(vals) / len(vals)

        return out

    def build_others(self):
        pass