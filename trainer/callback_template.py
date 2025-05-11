
class Callback:
    """
    Base class for Trainer hooks (callbacks). Override any methods to inject
    behavior at various stages of training/validation.
    """
    def on_train_start(self, trainer, model):
        pass

    def on_train_end(self, trainer, model):
        pass

    def on_epoch_start(self, trainer, model, epoch):
        pass

    def on_epoch_end(self, trainer, model, epoch):
        pass

    def on_batch_start(self, trainer, model, batch, batch_idx):
        pass

    def on_batch_end(self, trainer, model, batch, batch_idx, loss):
        pass

    def on_validation_start(self, trainer, model):
        pass

    def on_validation_batch_end(self, trainer, model, batch, batch_idx, outputs=None):
        pass

    def on_validation_end(self, trainer, model):
        pass

    def on_checkpoint(self, trainer, model, ckpt_path):
        pass
