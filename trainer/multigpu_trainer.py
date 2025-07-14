import copy
import threading
from typing import List
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F

from tqdm import tqdm
import psutil

from trainer.simple_trainer import GlobalStep

from trainer.utils.split_data import split_batch
from trainer.dist_tools.grad_aggr import average_gradients_into_main
def get_ram_info():
    ram = psutil.virtual_memory()
    return f"RAM: {ram.used/1073741824:.1f}GB/{ram.total/1073741824:.1f}GB ({ram.percent}%)"

class MultiGPUTrainer:
    """
    Single-process, multi-thread, multi-GPU trainer (no DDP).
    The public API mirrors your previous `Trainer`.
    """
    def __init__(
        self,
        boat,
        trainer_config,
        callbacks=None,
        logger=None,
        run_folder=None,
        resume_from=None,
        devices: List[int] | None = None,          # ← NEW
    ):
        # ───────────── basic stuff ─────────────
        self.max_epochs       = trainer_config.get("max_epochs", 10)
        self.devices          = devices
        self.callbacks        = callbacks or []
        self.val_check_steps  = trainer_config.get("val_check_steps", None)
        self.val_check_epochs = trainer_config.get("val_check_epochs", None)
        self.state_save_steps = trainer_config.get("state_save_steps", None)
        self.state_save_epochs= trainer_config.get("state_save_epochs", None)
        self.target_metric_name = boat.validation_config.get('target_metric_name', 'psnr')
        self.save_images      = trainer_config.get("save_images", False)
        self.logger           = logger

        # ───── model(s) and checkpoint restore ─────
        if resume_from:
            ckpt_path   = Path(resume_from)
            self.run_folder = Path(run_folder)
            boat, meta = boat.load_state(ckpt_path)
            self.global_step = GlobalStep(meta.get("global_step", 0))
            self.start_epoch = meta.get("epoch", 0)
        else:
            self.run_folder  = Path(run_folder) if run_folder else None
            self.global_step = GlobalStep(0)
            self.start_epoch = 0

        # --------------- NEW: replicas ---------------
        self.boat_replicas = []
        for rank, dev in enumerate(self.devices):
            replica = copy.deepcopy(boat) if rank else boat      # keep original as replica-0
            replica.configure_optimizers()
            replica.configure_losses()
            replica.configure_metrics()
            replica.to(torch.device(f"cuda:{dev}") if torch.cuda.is_available() else "cpu")
            replica.attach_global_step(self.global_step)         # keep your callback hooks working
            self.boat_replicas.append(replica)

        self.boat_main = self.boat_replicas[0]    # convenience alias

        self.valid_step_records  = {}
        self.valid_epoch_records = {}

    def fit(self, data_module):

        for cb in self.callbacks:
            cb.on_train_start(self, self.boat_main)

        for epoch in range(self.start_epoch, self.max_epochs):

            self.epoch = epoch
            for cb in self.callbacks:
                cb.on_epoch_start(self, self.boat_main, epoch)

            for m in self.boat_replicas:
                m.train()

            total_batches = len(data_module.train) if hasattr(data_module.train, "__len__") else None

            avg_loss = 0.0

            progress_bar = tqdm(
                enumerate(data_module.train),
                total=total_batches,
                desc=f"Epoch {epoch} | Training Total Loss N/A | {datetime.now():%Y-%m-%d %H:%M:%S}",
                unit="batch",
            )
            for batch_idx, batch in progress_bar:
                if batch_idx == 0 and epoch == self.start_epoch and self.global_step() > 0:
                    self.global_step -= 1
                self.global_step += 1

                # ---------------- multi-GPU core ---------------
                total_loss = self._multi_gpu_train_step(batch, batch_idx)  # ← NEW

                # ---------------- callbacks ---------------
                for cb in self.callbacks:
                    cb.on_batch_end(self, self.boat_main, batch, batch_idx, total_loss)

                # ---------------- optional validation / save ----------------
                if self.val_check_steps and self.global_step() % self.val_check_steps == 0:
                    avg_loss = self._run_validation(data_module.valid)
                    self.valid_step_records[self.global_step()] = {"avg_loss": avg_loss}

                if self.state_save_steps and self.global_step() % self.state_save_steps == 0:
                    state_path = self.boat_main.save_state(
                        self.run_folder, "boat_state",
                        global_step=self.global_step() + 1, epoch=epoch
                    )
                    self.valid_step_records.setdefault(self.global_step(), {})["state_path"] = state_path

                self.logger.flush()
                tqdm_desc = f"Epoch {epoch} | Training Total Loss {total_loss:.4f} | {datetime.now():%Y-%m-%d %H:%M:%S}"
                progress_bar.set_description(tqdm_desc)

            self.boat_main.training_epoch_end(epoch)

            print(f"epoch {epoch} ended - {get_ram_info()}")

            if self.val_check_epochs and epoch % self.val_check_epochs == 0:
                avg_loss = self._run_validation(data_module.valid)
                self.valid_epoch_records[epoch] = {"avg_loss": avg_loss.detach().cpu()}

            if self.state_save_epochs and epoch % self.state_save_epochs == 0:
                state_path = self.boat_main.save_state(
                    self.run_folder, "boat_state",
                    global_step=self.global_step(), epoch=epoch + 1
                )
                self.valid_epoch_records.setdefault(epoch, {})["state_path"] = state_path

            for cb in self.callbacks:
                cb.on_epoch_end(self, self.boat_main, epoch)

        for cb in self.callbacks:
            cb.on_train_end(self, self.boat_main)

    def _multi_gpu_train_step(self, batch, batch_idx):

        n_gpu = len(self.devices)
        split_batches = split_batch(batch, n_gpu)

        losses_gpu = [None] * n_gpu
        threads = []

        def _worker(rank: int):
            torch.cuda.set_device(self.devices[rank])
            replica = self.boat_replicas[rank]
            sub_batch = self._move_batch_to_device(split_batches[rank], torch.device(f"cuda:{self.devices[rank]}"))
            for cb in self.callbacks:
                cb.on_batch_start(self, replica, sub_batch, batch_idx)
            losses = replica.training_calc_losses(sub_batch, batch_idx)
            replica.training_backward(losses)   # computes grads
            losses_gpu[rank] = losses

        # Launch one thread per GPU
        for r in range(n_gpu):
            t = threading.Thread(target=_worker, args=(r,))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

        # ------------- reduce / average gradients -------------
        average_gradients_into_main(self.boat_replicas, self.devices)

        self.boat_main.log_train_losses(self.logger, losses_gpu[0])
        # ------------- optimiser & schedulers on main ----------
        self.boat_main.training_step()
        self.boat_main.lr_scheduling_step()

        # ------------- broadcast weights to replicas -----------
        # Get all model keys from the main boat
        model_keys = self.boat_main.models.keys()

        # Sync state for each model
        for model_key in model_keys:
            main_model = self.boat_main.models[model_key]
            if hasattr(main_model, 'parameters'):
                main_state = main_model.state_dict()
                for replica in self.boat_replicas[1:]:
                    replica_model = replica.models[model_key]
                    if hasattr(replica_model, 'parameters'):
                        replica_model.load_state_dict(main_state, strict=True)

        # 1) put every loss on CPU (or on GPU-0) …
        total_losses_cpu = [l['total_loss'].detach().to("cpu") for l in losses_gpu]

        # 2) then aggregate
        avg_loss = torch.stack(total_losses_cpu).mean()

        return avg_loss          # a CPU tensor, safe for logging


    def _move_batch_to_device(self, batch, device):
        if isinstance(batch, (list, tuple)):
            return [self._move_batch_to_device(x, device) for x in batch]
        elif isinstance(batch, dict):
            return {k: self._move_batch_to_device(v, device) for k, v in batch.items()}
        elif hasattr(batch, "to"):
            return batch.to(device)
        else:
            return batch

    # Validation is unchanged except we run only on replica 0
    def _run_validation(self, val_dataloader):
        for cb in self.callbacks:
            cb.on_validation_start(self, self.boat_main)

        self.boat_main.eval()
        aggr_metrics = {}
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader):
                batch = self._move_batch_to_device(batch, self.boat_main.device)
                metrics, named_imgs = self.boat_main.validation_step(batch, batch_idx)
                for cb in self.callbacks:
                    cb.on_validation_batch_end(self, self.boat_main, batch, batch_idx, outputs=metrics)
                # average each metric in metrics
                for key, value in metrics.items():
                    if key not in aggr_metrics:
                        aggr_metrics[key] = metrics[key]
                    else:
                        aggr_metrics[key] += metrics[key]
        
                self.boat_main.visualize_validation(self.logger, named_imgs, batch_idx)

            for key in aggr_metrics:
                aggr_metrics[key] /= batch_idx

        if not aggr_metrics:
            raise ValueError("Validation loop produced no losses.")

        self.boat_main.log_valid_metrics(self.logger, aggr_metrics)
        
        for cb in self.callbacks:
            cb.on_validation_end(self, self.boat_main)

        return aggr_metrics[self.target_metric_name]