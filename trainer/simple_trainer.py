import torch
import psutil
from pathlib import Path
from datetime import datetime
from tqdm import tqdm


class GlobalStep:
    def __init__(self, initial_value):
        self.value = initial_value
    
    def __call__(self):
        return self.value
    
    def __iadd__(self, other):
        if isinstance(other, int):
            self.value += other
        else:
            raise ValueError("GlobalStep can only be incremented by an integer.")
        return self
    
    def __isub__(self, other):
        if isinstance(other, int):
            self.value -= other
        else:
            raise ValueError("GlobalStep can only be decremented by an integer.")
        return self

def get_ram_info():
    # Get RAM statistics
    ram = psutil.virtual_memory()
    # Return used memory in GB and percentage
    return f"RAM: {ram.used/1073741824:.1f}GB/{ram.total/1073741824:.1f}GB ({ram.percent}%)"

class Trainer:
    def __init__(
        self, boat, trainer_config, 
        callbacks=None, logger=None,
        run_folder=None, resume_from=None, 
        devices=None
    ):
        self.max_epochs = trainer_config.get('max_epochs', 10)
        self.device = torch.device(trainer_config.get('device', 'cpu'))
        self.callbacks = callbacks or []
        self.val_check_steps = trainer_config.get('val_check_steps', None)
        self.val_check_epochs = trainer_config.get('val_check_epochs', None)
        self.state_save_steps = trainer_config.get('state_save_steps', None)
        self.state_save_epochs = trainer_config.get('state_save_epochs', None)
        self.target_metric_name = boat.validation_config.get('target_metric_name', 'psnr')
        self.save_images = trainer_config.get('save_images', False)
        self.logger = logger
        
        boat.configure_optimizers()
        boat.configure_losses()
        boat.configure_metrics()

        if resume_from:
            self.resume_from = Path(resume_from) if isinstance(resume_from, str) else resume_from
            self.run_folder = Path(run_folder) if isinstance(run_folder, str) else run_folder
            self.boat, metadata = boat.load_state(self.resume_from)
            self.global_step = GlobalStep(metadata.get('global_step', 0))
            self.start_epoch = metadata.get('epoch', 0)
        else:
            self.resume_from = None
            self.run_folder = Path(run_folder) if isinstance(run_folder, str) else run_folder
            self.global_step = GlobalStep(0)
            self.start_epoch = 0
            self.boat = boat        

        # Attach the trainer to the boat
        self.boat.attach_global_step(self.global_step)

        self.valid_step_records = {}
        self.valid_epoch_records = {}


    def fit(self, data_module):

        self.boat.to(self.device)

        for cb in self.callbacks:
            cb.on_train_start(self, self.boat)

        for epoch in range(self.start_epoch, self.max_epochs):

            self.epoch = epoch

            for cb in self.callbacks:
                cb.on_epoch_start(self, self.boat, epoch)

            self.boat.train()

            total_batches = len(data_module.train) if hasattr(data_module.train, '__len__') else None

            progress_bar = tqdm(
                enumerate(data_module.train),
                total=total_batches,
                desc=f"Epoch {epoch} | Training Total Loss N/A | {datetime.now():%Y-%m-%d %H:%M:%S}",
                unit="batch",
            )

            for batch_idx, batch in progress_bar:
                
                if batch_idx == 0 and epoch == self.start_epoch and self.resume_from:
                    self.global_step -= 1  # Adjust for resuming
                self.global_step += 1

                batch = self._move_batch_to_device(batch)

                for cb in self.callbacks:
                    cb.on_batch_start(self, self.boat, batch, batch_idx)

                losses = self.boat.training_calc_losses(batch, batch_idx)

                self.boat.training_backward(losses)
                self.boat.training_step()

                self.boat.lr_scheduling_step() 

                total_loss = losses['total_loss']

                self.boat.log_train_losses(self.logger, losses)

                for cb in self.callbacks:
                    cb.on_batch_end(self, self.boat, batch, batch_idx, total_loss)

                if self.val_check_steps is not None and self.global_step() % self.val_check_steps == 0:
                    avg_loss = self._run_validation(data_module.valid)
                    self.valid_step_records[self.global_step()] = {'avg_loss': avg_loss}

                if self.state_save_steps is not None and self.global_step() % self.state_save_steps == 0:
                    state_path = self.boat.save_state(self.run_folder, 'boat_state', global_step=self.global_step()+1, epoch=epoch)
                    
                    if self.global_step() not in self.valid_step_records:
                        self.valid_step_records[self.global_step()] = {}
                    self.valid_step_records[self.global_step()]['state_path'] = state_path

                if self.logger:
                    self.logger.flush()
                tqdm_desc = f"Epoch {epoch} | Training Total Loss {total_loss:.4f} | {datetime.now():%Y-%m-%d %H:%M:%S}"
                progress_bar.set_description(tqdm_desc)

            self.boat.training_epoch_end(epoch)
            
            print(f"epoch {epoch} has been ended, postfix = {get_ram_info()}")

            if self.val_check_epochs is not None and epoch % self.val_check_epochs == 0:
                avg_loss = self._run_validation(data_module.valid)
                self.valid_epoch_records[epoch] = {'avg_loss': avg_loss.detach().cpu()}

            if self.state_save_epochs is not None and epoch % self.state_save_epochs == 0:
                state_path = self.boat.save_state(self.run_folder, 'boat_state', global_step=self.global_step(), epoch=epoch+1)
                
                if epoch not in self.valid_epoch_records:
                    self.valid_epoch_records[epoch] = {}
                self.valid_epoch_records[epoch]['state_path'] = state_path

            for cb in self.callbacks:
                cb.on_epoch_end(self, self.boat, epoch)

        # on train end
        for cb in self.callbacks:
            cb.on_train_end(self, self.boat)

    def _run_validation(self, val_dataloader):
        for cb in self.callbacks:
            cb.on_validation_start(self, self.boat)

        self.boat.eval()
        aggr_metrics = {}
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader):
                batch = self._move_batch_to_device(batch)
                metrics, named_imgs = self.boat.validation_step(batch, batch_idx)
                for cb in self.callbacks:
                    cb.on_validation_batch_end(self, self.boat, batch, batch_idx, outputs=metrics)
                # average each metric in metrics
                for key, value in metrics.items():
                    if key not in aggr_metrics:
                        aggr_metrics[key] = metrics[key]
                    else:
                        aggr_metrics[key] += metrics[key]
        
                self.boat.visualize_validation(self.logger, named_imgs, batch_idx)

            for key in aggr_metrics:
                aggr_metrics[key] /= batch_idx

        if not aggr_metrics:
            raise ValueError("Validation loop produced no losses.")

        self.boat.log_valid_metrics(self.logger, aggr_metrics)
        
        for cb in self.callbacks:
            cb.on_validation_end(self, self.boat)

        return aggr_metrics[self.target_metric_name]
    
    def _move_batch_to_device(self, batch):
        if isinstance(batch, (list, tuple)):
            return [self._move_batch_to_device(x) for x in batch]
        elif isinstance(batch, dict):
            return {k: self._move_batch_to_device(v) for k, v in batch.items()}
        elif hasattr(batch, 'to'):
            return batch.to(self.device)
        else:
            return batch
