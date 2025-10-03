import torch
from rhino.boats.base.base_boat import BaseBoat
from trainer.utils.build_components import build_module
from trainer.utils.ddp_utils import move_to_device

class BaseGanBoat(BaseBoat):
    def __init__(self, config={}):
        super().__init__(config=config)
        
        assert config is not None, "main config must be provided"

        # Build the model
        self.models['net'] = build_module(boat_config['net'])
        self.models['critic'] = build_module(boat_config['critic'])

        self.boat_config = boat_config
        self.optimization_config = optimization_config or {}
        self.validation_config = validation_config or {}

        self.concurrent = bool(optimization_config.get('hyper_parameters', {}).get('concurrent', False))
        self.g_interval = int(optimization_config.get('hyper_parameters', {}).get('g_interval', 1))
        self.d_interval = int(optimization_config.get('hyper_parameters', {}).get('d_interval', 1))
        self.adversarial_weight = float(optimization_config.get('hyper_parameters', {}).get('adversarial_weight', 0.01))

        self.use_ema = self.optimization_config.get('use_ema', False)
        self.use_reference = validation_config.get('use_reference', False)

        # Setup EMA if enabled
        if self.use_ema:
            self._setup_ema()
            self.ema_start = self.optimization_config.get('ema_start', 0)
        else:
            self.ema_start = 0

    def predict(self, noise):
        
        network_in_use = self.models['net_ema'] if self.use_ema and 'net_ema' in self.models else self.models['net']

        return network_in_use(noise)

    def d_step_calc_losses(self, batch):

        gt = batch['gt']
        batch_size = gt.size(0)
        
        # Initialize random noise in latent space
        noise = self.noise_generator.next(batch_size, device=self.device)

        # Generate fake samples with current G (no grad to G)
        with torch.no_grad():
            x_fake = self.models['net'](noise)

        # Forward D on real & fake (under autocast if enabled)
        d_real, d_fake = self.models['critic'](gt), self.models['critic'](x_fake)
        # GPT suggest remove "* self.adversarial_weight"
        d_loss = self.losses['critic'](d_real, d_fake) 

        return d_loss

    def g_step_calc_losses(self, batch):

        gt = batch['gt']
        batch_size = gt.size(0)
        
        # Initialize random noise in latent space
        noise = self.noise_generator.next(batch_size, device=self.device)

        # Generate fake samples with current G (no grad to G)
        x_fake = self.models['net'](noise)
        
        # Forward G on fake
        d_fake_for_g = self.models['critic'](x_fake)
        g_loss = self.losses['critic'](d_fake_for_g, None) * self.adversarial_weight

        return g_loss

    def d_step(self, batch, scaler): # start_new_accum, scaler, loss_scale, should_step_now):

        micro_batches = self._split_batch(batch, self.total_micro_steps)

        # Enable G, freeze D so G doesn't update D
        self.models['net'].requires_grad_(False)
        self.models['critic'].requires_grad_(True)

        self._zero_grad(['critic'], set_to_none=True)

        micro_losses_list = []
        for current_micro_step, micro_batch in enumerate(micro_batches):
            micro_batch = move_to_device(micro_batch, self.device)
            micro_d_loss = self.d_step_calc_losses(micro_batch)
            micro_target_loss = micro_d_loss / self.total_micro_steps
            micro_losses_list.append({'d_loss': micro_d_loss})
            self.training_backpropagation(micro_target_loss, current_micro_step, scaler)

        self.training_gradient_descent(scaler, ['critic'])

        return self._aggregate_loss_dicts(micro_losses_list)
    
    def g_step(self, batch, scaler):

        micro_batches = self._split_batch(batch, self.total_micro_steps)

        # Enable G, freeze D so G doesn't update D
        self.models['net'].requires_grad_(True)
        self.models['critic'].requires_grad_(False)

        self._zero_grad(['net'], set_to_none=True)

        micro_losses_list = []
        for current_micro_step, micro_batch in enumerate(micro_batches):
            micro_batch = move_to_device(micro_batch, self.device)
            micro_g_loss = self.g_step_calc_losses(micro_batch)
            micro_target_loss = micro_g_loss / self.total_micro_steps
            micro_losses_list.append({'g_loss': micro_g_loss})
            self.training_backpropagation(micro_target_loss, current_micro_step, scaler)

        self.training_gradient_descent(scaler, ['net'])

        return self._aggregate_loss_dicts(micro_losses_list)

    def training_step(self, batch, batch_idx, epoch, *, scaler=None):

        self.epoch = epoch

        # Schedules
        do_g = (self.g_interval > 0) and (batch_idx % self.g_interval == 0)
        do_d = (self.d_interval > 0) and (batch_idx % self.d_interval == 0)

        losses = {}
        if do_d:
            d_loss_dict = self.d_step(batch, scaler)

        if do_g:
            g_loss_dict = self.g_step(batch, scaler)

        losses.update(d_loss_dict)
        losses.update(g_loss_dict)
        losses['total_loss'] = d_loss_dict.get('d_loss', 0.0) + g_loss_dict.get('g_loss', 0.0)

        self.models['net'].requires_grad_(True)
        self.models['critic'].requires_grad_(True)

        self._update_ema()

        self.training_lr_scheduling_step(active_keys=['net', 'critic'])

        return losses

    def validation_step(self, batch, batch_idx):

        batch = move_to_device(batch, self.device)

        gt = batch['gt']

        batch_size = gt.shape[0]

        with torch.no_grad():

            noise = self.noise_generator.next(batch_size, device=self.device)

            x_fake = self.predict(noise)

            valid_output = {'preds': x_fake, 'targets': gt}

            # Reset Metric in the begining iter in an epoch
            if batch_idx == 0:
                self._reset_metrics()

            metrics = self._calc_metrics(valid_output)

            named_imgs = {'groundtruth': gt, 'generated': x_fake,}

        return metrics, named_imgs

    def build_others(self):
        self.noise_generator = build_module(self.boat_config.get('noise_generator', None))