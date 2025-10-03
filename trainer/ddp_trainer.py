# ddp_trainer_demo.py
from __future__ import annotations

import os

import torch
import torch.multiprocessing as mp


from trainer.utils.ddp_utils import find_free_port

from trainer.workers.ddp_train_worker import ddp_train_worker

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

class DDPTrainer:
    """
    One-process-per-GPU trainer. No torchrun needed.
    - Trainer owns DDP/AMP/no_sync.
    - Boat owns training_step (loss -> backward -> maybe step), DDP-agnostic.
    """
    def __init__(self, config):
        assert config is not None, "config must be provided"

        self.devices = config['trainer'].get('devices', [0])
        total = len(self.devices)
        world_size = total if self.devices is None else max(1, min(total, total))
        self.config['world_size'] = world_size

        self.config = config

    def fit(self, data_module):

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Please ensure a CUDA-capable device is present and PyTorch is installed with CUDA support.")

        if self.config['world_size'] == 1:

            addr, port = "127.0.0.1", find_free_port()

            ddp_train_worker(rank=0, world_size=1, addr=addr, port=port,
                             devices=self.devices, config=self.config, 
                             data_module=data_module)

            return

        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(find_free_port()))
        addr, port = os.environ["MASTER_ADDR"], int(os.environ["MASTER_PORT"])

        # spawn – pass only picklable state
        mp.start_processes(
            ddp_train_worker,
            nprocs=self.config['world_size'],
            args=(self.config['world_size'], addr, port, self.devices, self.config, data_module),
            start_method="spawn",
            join=True,
        )

