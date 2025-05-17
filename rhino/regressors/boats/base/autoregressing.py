from trainer.boats.image_gen_boat import ImageGenerationBoat
from trainer.utils.build_components import build_module
import torch

class AutoRegressionBoat(ImageGenerationBoat):
    def __init__(self, boat_config=None, optimization_config=None, validation_config=None):
        super().__init__(boat_config, optimization_config, validation_config)

        self.models['solver'] = build_module(boat_config['solver'])

        if 'metric_prep' in validation_config:
            self.metric_prep = build_module(validation_config['metric_prep'])

    def forward(self, zeros):

        network_in_use = self.models['model_ema'] if self.use_ema and 'model_ema' in self.models else self.models['model']

        generations = self.models['solver'].solve(network_in_use, zeros)

        return torch.clamp(generations, 0, 1)

    def training_step(self, batch, batch_idx):

        images = batch['gt']

        predictions = self.models['model'](images)

        if hasattr(predictions, 'sample'):
            predictions = predictions.sample # predictions.shape = (B, C, H, W, N)

        loss = self.losses['model'](predictions, images)

        self._step(loss)

        self._log_metric(loss, metric_name='image_loss', prefix='train')

        if self.use_ema and self.get_global_step() > self.ema_start:
            self._update_ema()

        return loss

    def validation_step(self, batch, batch_idx):

        with torch.no_grad():

            images = batch['gt']

            network_in_use = self.models['model_ema'] if self.use_ema and 'model_ema' in self.models else self.models['model']

            predictions = network_in_use(images)

            if hasattr(predictions, 'sample'):
                predictions = predictions.sample

            loss = self.losses['model'](predictions, images)

            self._log_metric(loss, metric_name='image_loss', prefix='valid')

            if hasattr(self, 'metric_prep'):
                predictions = self.metric_prep(predictions)

            results = self._calc_reference_quality_metrics(predictions, images)

            self._log_metrics(results, prefix='valid')

            # Maybe it is not good to have all zeros as input. 
            zeros = torch.zeros_like(images)

            generations = self.forward(zeros)

            named_imgs = {'groundtruth': images, 'generated': generations,}

            self._visualize_validation(named_imgs, batch_idx)

        return loss
    