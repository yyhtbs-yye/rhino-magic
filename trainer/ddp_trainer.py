# ddp_trainer_demo.py
from __future__ import annotations

import os, copy

import torch
import torch.multiprocessing as mp
from pathlib import Path

from trainer.simple_trainer import GlobalStep

from trainer.utils.ddp_utils import (
    find_free_port,
)

from trainer.workers.ddp_train_worker import ddp_train_worker

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class DDPTrainer:
    """
    One-process-per-GPU trainer. No torchrun needed.
    - Trainer owns DDP/AMP/no_sync.
    - Boat owns training_step (loss -> backward -> maybe step), DDP-agnostic.
    """
    def __init__(
        self,
        boat,
        trainer_config,
        callback_configs=None,
        experiment_name = None,
        run_folder=None,
        resume_from=None,
        devices = None,
    ):
        
        self.train_states = {}
        self.callback_configs = callback_configs

        self.trainer_config = trainer_config
        self.devices          = devices
        self.train_states['val_check_epochs']   = trainer_config.get("val_check_epochs", None)
        self.train_states['state_save_epochs']  = trainer_config.get("state_save_epochs", None)
        self.train_states['target_metric_name'] = boat.validation_config.get('target_metric_name', 'psnr')
        self.train_states['save_images']        = trainer_config.get("save_images", False)
        self.train_states['experiment_name']    = experiment_name

        # ───── model(s) and checkpoint restore ─────
        if resume_from:
            ckpt_path   = Path(resume_from)
            boat, meta = boat.load_state(ckpt_path)
            self.train_states['run_folder'] = Path(run_folder)
            self.train_states['global_step'] = GlobalStep(meta.get("global_step", 0))
            self.train_states['start_epoch'] = meta.get("epoch", 0)
        else:
            self.train_states['run_folder']  = Path(run_folder) if run_folder else None
            self.train_states['global_step'] = GlobalStep(0)
            self.train_states['start_epoch'] = 0

        # keep a CPU template; each worker deep-copies it
        self._boat_template = copy.deepcopy(boat)

    def fit(self, data_module):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Please ensure a CUDA-capable device is present and PyTorch is installed with CUDA support.")
        
        total = len(self.devices)
        world_size = total if self.devices is None else max(1, min(total, total))

        if world_size == 1:
            addr, port = "127.0.0.1", find_free_port()
            ddp_train_worker(
                rank=0,
                world_size=1,
                addr=addr,
                port=port,
                devices=self.devices,
                boat_template=self._boat_template,
                trainer_config=self.trainer_config,
                data_module=data_module,
                train_states=self.train_states,
                callback_configs=self.callback_configs,
            )
            return

        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(find_free_port()))
        addr, port = os.environ["MASTER_ADDR"], int(os.environ["MASTER_PORT"])

        # spawn – pass only picklable state
        mp.start_processes(
            ddp_train_worker,
            nprocs=world_size,
            args=(world_size, addr, port, self.devices, self._boat_template, self.trainer_config, 
                  data_module, self.train_states, self.callback_configs),
            start_method="spawn",
            join=True,
        )

