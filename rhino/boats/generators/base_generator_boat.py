import torch

from rhino.boats.base.base_boat import BaseBoat
from trainer.utils.build_components import build_module

class BaseGeneratorBoat(BaseBoat):
    def __init__(self, boat_config=None, optimization_config=None, validation_config=None):
        super().__init__()
        
        assert boat_config is not None, "boat_config must be provided"

        # Build the model
        self.models['net'] = build_module(boat_config['net'])
        
        # Store configurations
        self.boat_config = boat_config
        self.optimization_config = optimization_config or {}
        self.validation_config = validation_config or {}

        self.use_ema = self.optimization_config.get('use_ema', False)
        self.use_reference = validation_config.get('use_reference', True)
        
        # Setup EMA if enabled
        if self.use_ema:
            self._setup_ema()
            self.ema_start = self.optimization_config.get('ema_start', 0)

    def forward(self, x):
        
        network_in_use = self.models['net_ema'] if self.use_ema and 'net_ema' in self.models else self.models['net']
            
        x_sq = network_in_use(x)
        
        return x_sq

    def training_calc_losses(self, batch, batch_idx):

        x_gt = batch['gt']
        x_lq = batch['lq']

        batch_size = x_gt.size(0)

        preds = self.models['net'](x_lq)
        
        train_output = {
            'preds': preds,
            'targets': x_gt,
            'weights': torch.ones(batch_size, device=x_gt.device),
            **batch
        }
        
        losses = self._calc_losses(train_output)

        return losses

    def training_backward(self, losses):

        total_loss = losses['total_loss']
        
        self._backward(total_loss)

    def training_step(self):

        self._step()

        if self.use_ema and self.get_global_step() >= self.ema_start:
            self._update_ema()
        
        return

    def validation_step(self, batch, batch_idx):

        x_gt = batch['gt']
        x_lq = batch['lq']

        with torch.no_grad():

            preds = self.forward(x_lq)

            valid_output = {'preds': preds, 'targets': x_gt,}

            metrics = self._calc_metrics(valid_output)

            named_imgs = {'groundtruth': x_gt, 'generated': preds, 'low_quality': x_lq,}

        return metrics, named_imgs
    