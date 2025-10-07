from abc import ABC, abstractmethod

class TemplateBoat(ABC):
    """
    Abstract base class for model containers to be used with the Trainer.
    
    A "Boat" represents a container for models, optimizers, and training logic,
    but is not itself a nn.Module.
    """

    @abstractmethod
    def __init__(self, config):
        pass

    @abstractmethod
    def predict(self, x):
        pass

    @abstractmethod
    def to(self, device):
        pass

    @abstractmethod
    def move_optimizer_to_device(self, device):
        pass
    
    @abstractmethod
    def parameters(self):
        pass
    
    @abstractmethod
    def training_backpropagation(self, loss, current_micro_step, scaler):
        pass

    @abstractmethod
    def training_gradient_descent(self, scaler, active_keys):
        pass

    @abstractmethod
    def validation_step(self, batch, batch_idx):
        pass

    @abstractmethod
    def save_state(self, run_folder, prefix, global_step, epoch):
        pass

    @abstractmethod
    def load_state(self, state_path, strict):
        pass
    
    @abstractmethod
    def training_lr_scheduling_step(self):
        pass
    
    @abstractmethod
    def train(self):
        pass      
    
    @abstractmethod
    def eval(self):
        pass      
    
    @abstractmethod
    def build_losses(self):
        pass      

    @abstractmethod
    def build_metrics(self):
        pass      

    @abstractmethod
    def build_optimizers(self):
        pass      

    @abstractmethod
    def _calc_reference_quality_metrics(self, predictions, targets):
        pass

    @abstractmethod
    def _calc_noreference_quality_metrics(self, predictions):
        pass

    @abstractmethod
    def _calc_metrics(self, valid_output):
        pass

    @abstractmethod
    def _log_values(self, logger, results, prefix):
        pass

    @abstractmethod
    def _log_value(self, logger, result, metric_name, prefix):
        pass

    @abstractmethod
    def log_train_losses(self, logger, losses):
        pass

    @abstractmethod
    def log_valid_metrics(self, logger, metrics):
        pass

    # @abstractmethod
    # def training_calc_losses(self, batch):
    #     pass

    @abstractmethod
    def _zero_grad(self, active_keys):
        pass

    @abstractmethod
    def training_step(self, batch, batch_idx, epoch, *, scaler):
        pass

    @abstractmethod
    def visualize_validation(self, logger, named_imgs, batch_idx, trainer_config):
        pass
        
    @abstractmethod
    def get_global_step(self):
        pass
    
    @abstractmethod
    def attach_global_step(self, global_step):
        pass

    @abstractmethod
    def build_lr_scheduler_by_name(self, model_name):
        pass

    @abstractmethod
    def _setup_ema(self):
        pass

    @abstractmethod
    def _update_ema(self):
        pass
