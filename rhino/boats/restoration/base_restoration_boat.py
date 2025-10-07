import torch

from rhino.boats.base.base_boat import BaseBoat
from trainer.utils.build_components import build_module
from trainer.utils.ddp_utils import move_to_device

class BaseRestorationBoat(BaseBoat):
    def __init__(self, config={}):
        super().__init__(config=config)
        
        assert config is not None, "main config must be provided"

        # Store configurations
        self.boat_config = config.get('boat', {})
        self.optimization_config = config.get('optimization', {})
        self.validation_config = config.get('validation', {})

        self.use_ema = self.optimization_config.get('use_ema', False)
        self.use_reference = self.validation_config.get('use_reference', True)
        
        # Setup EMA if enabled
        if self.use_ema:
            self._setup_ema()
            self.ema_start = self.optimization_config.get('ema_start', 0)

    def predict(self, lq):
        
        network_in_use = self.models['net_ema'] if self.use_ema and 'net_ema' in self.models else self.models['net']
            
        restored = network_in_use(lq)
        
        return restored

    def training_calc_losses(self, batch):

        gt = batch['gt']
        lq = batch['lq']

        batch_size = gt.size(0)

        restored = self.models['net'](lq)
        
        train_output = {
            'preds': restored,
            'targets': gt,
            'weights': torch.ones(batch_size, device=self.device),
            **batch
        }
        
        losses = {'total_loss': torch.tensor(0.0, device=self.device)}

        losses['net'] = self.losses['net'](train_output)
        losses['total_loss'] += losses['net']

        return losses

    def validation_step(self, batch, batch_idx):

        batch = move_to_device(batch, self.device)

        gt = batch['gt']
        lq = batch['lq']

        with torch.no_grad():

            restored = self.predict(lq)

            valid_output = {'preds': restored, 'targets': gt,}

            metrics = self._calc_metrics(valid_output)

            named_imgs = {'groundtruth': gt, 'generated': restored, 'low_quality': lq,}

        return metrics, named_imgs
    