import os
import torch
from pathlib import Path
from tqdm import tqdm

class Trainer:
    def __init__(
        self, boat, trainer_config, 
        callbacks=None, logger=None,
        run_folder=None, resume_from=None,
    ):
        self.max_epochs = trainer_config.get('max_epochs', 10)
        self.device = torch.device(trainer_config.get('device', 'cpu'))
        self.callbacks = callbacks or []
        self.val_check_steps = trainer_config.get('val_check_steps', None)
        self.val_check_epochs = trainer_config.get('val_check_epochs', None)
        self.state_save_steps = trainer_config.get('state_save_steps', None)
        self.state_save_epochs = trainer_config.get('state_save_epochs', None)
        self.save_images = trainer_config.get('save_images', False)
        self.logger = logger
        
        boat.configure_optimizers()
        boat.configure_losses()
        boat.configure_metrics()

        if resume_from:
            self.resume_from = Path(resume_from) if isinstance(resume_from, str) else resume_from
            self.run_folder = self.resume_from.parent
            self.boat, metadata = boat.load_state(self.resume_from)
            self.global_step = metadata.get('global_step', 0)
            self.start_epoch = metadata.get('epoch', 0)
        else:
            self.resume_from = None
            self.run_folder = Path(run_folder) if isinstance(run_folder, str) else run_folder
            self.global_step = 0
            self.start_epoch = 0
            self.boat = boat

        # Attach the trainer to the boat
        self.boat.attach_trainer(self)

        self.valid_step_records = {}
        self.valid_epoch_records = {}
        
    def fit(self, data_module):
        
        self.boat.to(self.device)

        for cb in self.callbacks:
            cb.on_train_start(self, self.boat)

        for epoch in range(self.start_epoch, self.max_epochs):
            for cb in self.callbacks:
                cb.on_epoch_start(self, self.boat, epoch)

            self.boat.train()
            
            total_batches = len(data_module.train) if hasattr(data_module.train, '__len__') else None

            for batch_idx, batch in tqdm(enumerate(data_module.train), total=total_batches, desc=f"Epoch {epoch}", unit="batch"):
                if batch_idx == 0 and epoch == self.start_epoch and self.resume_from:
                    self.global_step -= 1  # Adjust for resuming
                self.global_step += 1

                batch = self._move_batch_to_device(batch)

                for cb in self.callbacks:
                    cb.on_batch_start(self, self.boat, batch, batch_idx)
                
                loss = self.boat.training_step(batch, batch_idx)

                self.boat.lr_scheduling_step() 

                for cb in self.callbacks:
                    cb.on_batch_end(self, self.boat, batch, batch_idx, loss)

                if data_module.valid is not None: 
                    if self.val_check_steps is not None and self.global_step % self.val_check_steps == 0:
                        avg_loss = self._run_validation(data_module.valid, self.global_step, None)
                        self.valid_step_records[self.global_step] = {'avg_loss': avg_loss}

                if self.state_save_steps is not None and self.global_step % self.state_save_steps == 0:
                    state_path = self.boat.save_state(self.run_folder, 'boat_state', global_step=self.global_step+1, epoch=epoch)
                    
                    if self.global_step not in self.valid_step_records:
                        self.valid_step_records[self.global_step] = {}
                    self.valid_step_records[self.global_step]['state_path'] = state_path

            if data_module.valid is not None:
                if self.val_check_epochs is not None and epoch % self.val_check_epochs == 0:
                    avg_loss = self._run_validation(data_module.valid, None, epoch)
                    self.valid_epoch_records[epoch] = {'avg_loss': avg_loss}

            if self.state_save_epochs is not None and epoch % self.state_save_epochs == 0:
                state_path = self.boat.save_state(self.run_folder, 'boat_state', global_step=self.global_step, epoch=epoch+1)
                
                if epoch not in self.valid_epoch_records:
                    self.valid_epoch_records[epoch] = {}
                self.valid_epoch_records[epoch]['state_path'] = state_path

            for cb in self.callbacks:
                cb.on_epoch_end(self, self.boat, epoch)

        # on train end
        for cb in self.callbacks:
            cb.on_train_end(self, self.boat)

    def _run_validation(self, val_dataloader, global_step=None, epoch=None):
        for cb in self.callbacks:
            cb.on_validation_start(self, self.boat)
        
        step_losses = []

        self.boat.eval()

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader):
                batch = self._move_batch_to_device(batch)
                loss = self.boat.validation_step(batch, batch_idx)
                for cb in self.callbacks:
                    cb.on_validation_batch_end(self, self.boat, batch, batch_idx, outputs=loss)
                step_losses.append(loss)
        
        if len(step_losses) == 0:
            raise ValueError("Validation step returned no losses. Check your validation step implementation.")
        avg_loss = sum(step_losses) / len(step_losses)

        for cb in self.callbacks:
            cb.on_validation_end(self, self.boat)

        return avg_loss

    def _move_batch_to_device(self, batch):
        if isinstance(batch, (list, tuple)):
            return [self._move_batch_to_device(x) for x in batch]
        elif isinstance(batch, dict):
            return {k: self._move_batch_to_device(v) for k, v in batch.items()}
        elif hasattr(batch, 'to'):
            return batch.to(self.device)
        else:
            return batch
