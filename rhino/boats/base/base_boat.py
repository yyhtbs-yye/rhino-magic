import torch

from trainer.utils.state_load_save import save_state, load_state
<<<<<<< HEAD
from trainer.utils.build_components import build_module, build_modules, build_optimizer, build_lr_scheduler
=======
from trainer.utils.build_components import build_module, build_modules, build_optimizer, build_lr_scheduer
>>>>>>> 7e3c9336db6dd5c66c1e62faade22b251b0f42a3
from trainer.visualizers.basic_visualizer import visualize_image_dict
from rhino.helpers.hooks.basic_hooks import named_forward_hook
from copy import deepcopy

<<<<<<< HEAD
from rhino.boats.base.template_boat import TemplateBoat

from trainer.utils.ddp_utils import ddp_no_sync_all, move_to_device
=======
from functools import partial

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
        self.hook_memories = {}  # Memory for storing outputs from forward hooks
>>>>>>> 7e3c9336db6dd5c66c1e62faade22b251b0f42a3

class BaseBoat(TemplateBoat):

    def __init__(self, boat_config=None, optimization_config=None, validation_config=None):

        self.total_micro_steps = optimization_config.pop('total_micro_steps', 1)
        self.target_loss_key = optimization_config.pop('target_loss_key', 'total_loss')
        self.models = {}
        self.optimizers = {}
        self.losses = {}
        self.metrics = {}
        self.lr_schedulers = {}
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
        

        self.move_losses_to_device(device)

        # Move optimizer states to the same device
        self.move_optimizer_to_device(device)
        return self

<<<<<<< HEAD
=======
    def move_losses_to_device(self, device):
        """
        Move all loss functions to the specified device.
        
        Args:
            device: The device to move the losses to (e.g., 'cuda:3', 'cpu')
            
        Returns:
            None
        """
        for name, loss in self.losses.items():
            if hasattr(loss, 'to'):
                self.losses[name] = loss.to(device)

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

>>>>>>> 7e3c9336db6dd5c66c1e62faade22b251b0f42a3
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

<<<<<<< HEAD
    def training_backpropagation(self, loss, current_micro_step, scaler):
=======
    def configure_optimizers(self):
        for model_name in self.optimization_config:
            if 'use_ema' in model_name:
                continue
            self.optimizers[model_name] = build_optimizer(
                self.models[model_name].parameters(), 
                self.optimization_config[model_name]
            )

        for loss_name in self.losses:
            loss_fn = self.losses[loss_name]
            if hasattr(loss_fn, 'opt_id'):
                self.optimizers[loss_fn.opt_id].add_param_group({'params': loss_fn.parameters()})

        self.configure_lr_scheduers()
>>>>>>> 7e3c9336db6dd5c66c1e62faade22b251b0f42a3

        use_no_sync = (self.total_micro_steps > 1) and (current_micro_step < self.total_micro_steps - 1)

<<<<<<< HEAD
        with ddp_no_sync_all(self, enabled=use_no_sync):
            if scaler is not None and getattr(scaler, "is_enabled", lambda: False)():
                scaler.scale(loss).backward()
            else:
                loss.backward()
=======
        for loss_name in self.losses:
            loss_fn = self.losses[loss_name]
            loss_val = loss_fn(train_output)

            if isinstance(loss_val, tuple):
                loss_val, loss_info = loss_val
                
            losses['total_loss'] += loss_val
            losses[loss_name] = loss_val
>>>>>>> 7e3c9336db6dd5c66c1e62faade22b251b0f42a3

    def training_gradient_descent(self, scaler, active_keys):

        if scaler is not None and scaler.is_enabled():
            for k in active_keys: scaler.step(self.optimizers[k])
            scaler.update()
        else:
            for k in active_keys: self.optimizers[k].step()

    def training_lr_scheduling_step(self):
        for _, scheduler in self.lr_schedulers.items(): # _: scheduler_name
            scheduler.step()
            
    def training_step(self, batch, batch_idx, epoch, *, scaler=None):
        
        active_keys = list(self.optimizers.keys())
        
        micro_batches = self._split_batch(batch, self.total_micro_steps)

        self._zero_grad(active_keys, set_to_none=True)

        micro_losses_list = []
        for current_micro_step, micro_batch in enumerate(micro_batches):
            micro_batch = move_to_device(micro_batch, self.device)
            micro_losses = self.training_calc_losses(micro_batch, batch_idx)
            micro_target_loss = micro_losses[self.target_loss_key] / self.total_micro_steps
            micro_losses_list.append(micro_losses)
            self.training_backpropagation(micro_target_loss, current_micro_step, scaler)

        self.training_gradient_descent(scaler, active_keys)
        
        self._update_ema()

        self.training_lr_scheduling_step()

        return self._aggregate_loss_dicts(micro_losses_list)

    # ------------------------------------ Visualization ---------------------------------------------

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
    def build_losses(self):
        for loss_name in self.boat_config.get('loss', {}):
            self.losses[loss_name] = build_module(self.boat_config['loss'][loss_name])         

    def build_metrics(self):
        self.metrics = build_modules(self.validation_config.get('metrics', {}))

    def build_optimizers(self):
        for model_name in self.optimization_config:
            if 'use_ema' in model_name:
                continue
            self.optimizers[model_name] = build_optimizer(
                self.models[model_name].parameters(), 
                self.optimization_config[model_name]
            )
        self.build_lr_schedulers()

    def build_lr_schedulers(self):

        if 'lr_scheduler' in self.optimization_config['net'] and len(self.optimization_config['net'].get('lr_scheduler', {})) > 0:
            self.lr_schedulers['net'] = build_lr_scheduler(self.optimizers['net'], 
                                                            self.optimization_config['net'].get('lr_scheduler', {}))

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

<<<<<<< HEAD
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
=======
        for ema_param, param in zip(self.models['net_ema'].parameters(), self.models['net'].parameters()):
            ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)
        for ema_buffer, buffer in zip(self.models['net_ema'].buffers(), self.models['net'].buffers()):
            ema_buffer.data.copy_(buffer.data)

    def _install_forward_hooks(self, model_layer_names={}, hook_fn=named_forward_hook):
        for model_name in model_layer_names:

            if model_name not in self.hook_memories:
                self.hook_memories[model_name] = {}

            if model_name not in self.models:
                continue
            
            layer_names = model_layer_names[model_name]
            if isinstance(layer_names, str):
                layer_names = [layer_names]

            for layer_name in layer_names:
                hook_handle = self.models[model_name].get_submodule(layer_name).register_forward_hook(
                    partial(hook_fn, layer_name, self.hook_memories[model_name])
                )

    def _collect_from_forward_hooks(self, batch, batch_idx):
        if 'hook_fx' not in batch:
            batch['hook_fx'] = {}
        for model_name, memory in self.hook_memories.items():
            for layer_name, output in memory.items():
                if output is not None:
                    batch['hook_fx'][f"{model_name}_{layer_name}"] = output
        return batch

>>>>>>> 7e3c9336db6dd5c66c1e62faade22b251b0f42a3
