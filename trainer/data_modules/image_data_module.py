from trainer.utils.build_components import build_dataset
from torch.utils.data import DataLoader, DistributedSampler
import torch.multiprocessing as mp
from functools import partial

# top-level, importable by name
def seed_worker(worker_id, base_seed: int = 42, rank: int = 0):
    import random, numpy as np, torch
    seed = int(base_seed) + int(rank) * 1000 + int(worker_id)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Optional: avoid OpenCV thread contention in workers
    try:
        import cv2
        cv2.setNumThreads(0)
    except Exception:
        pass

class DistTrainSingleValidDataModule:
    def __init__(self, data_config=None):
        super().__init__()
        self.data_config = data_config
        self.train_dataloader_config = self.data_config['train_dataloader']
        self.valid_dataloader_config = self.data_config['valid_dataloader']
        self.train_dataset = build_dataset(self.train_dataloader_config['dataset'])
        self.valid_dataset = build_dataset(self.valid_dataloader_config['dataset'])

    def make_train_loader(self, world_size, rank):
        cfg = self.train_dataloader_config
        num_workers = int(cfg.get('num_workers', 4))
        pin_memory  = bool(cfg.get('pin_memory', True))
        persistent  = bool(cfg.get('persistent_workers', False))
        prefetch    = cfg.get('prefetch_factor', 2)
        base_seed   = int(cfg.get('seed', 42))

        sampler = DistributedSampler(
            self.train_dataset,
            num_replicas=world_size, rank=rank,
            shuffle=True, drop_last=True
        ) if world_size > 1 else None

        kwargs = dict(
            dataset=self.train_dataset,
            batch_size=cfg.get('batch_size', 16),
            sampler=sampler if world_size > 1 else None,
            shuffle=(sampler is None),
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
            multiprocessing_context=mp.get_context("spawn"),
        )

        # Only set these when workers > 0 (PyTorch requires it)
        if num_workers > 0:
            kwargs["worker_init_fn"] = partial(seed_worker, base_seed=base_seed, rank=rank)
            kwargs["persistent_workers"] = persistent
            if prefetch is not None:
                kwargs["prefetch_factor"] = int(prefetch)

        return DataLoader(**kwargs)

    def make_valid_loader(self):
        num_workers   = int(self.valid_dataloader_config.get('num_workers', 4))
        pin_memory    = bool(self.valid_dataloader_config.get('pin_memory', True))
        persistent    = bool(self.valid_dataloader_config.get('persistent_workers', False))
        prefetch      = self.valid_dataloader_config.get('prefetch_factor', 2)

        kwargs = dict(
            dataset=self.valid_dataset,
            batch_size=self.valid_dataloader_config.get('batch_size', 16),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
            multiprocessing_context=mp.get_context("spawn"),  # ← safe too
            persistent_workers=persistent if num_workers > 0 else False,
            timeout=self.valid_dataloader_config.get('timeout', 0),
        )
        if num_workers > 0 and prefetch is not None:
            kwargs['prefetch_factor'] = int(prefetch)

        return DataLoader(**kwargs)

class DistTrainDistValidDataModule:
    def __init__(self, data_config=None):
        super().__init__()
        self.data_config = data_config
        self.train_dataloader_config = self.data_config['train_dataloader']
        self.valid_dataloader_config = self.data_config['valid_dataloader']
        self.train_dataset = build_dataset(self.train_dataloader_config['dataset'])
        self.valid_dataset = build_dataset(self.valid_dataloader_config['dataset'])

    def make_train_loader(self, world_size, rank):
        cfg = self.train_dataloader_config
        num_workers = int(cfg.get('num_workers', 4))
        pin_memory  = bool(cfg.get('pin_memory', True))
        persistent  = bool(cfg.get('persistent_workers', False))
        prefetch    = cfg.get('prefetch_factor', 2)
        base_seed   = int(cfg.get('seed', 42))

        sampler = DistributedSampler(
            self.train_dataset,
            num_replicas=world_size, rank=rank,
            shuffle=True, drop_last=True
        ) if world_size > 1 else None

        kwargs = dict(
            dataset=self.train_dataset,
            batch_size=cfg.get('batch_size', 16),
            sampler=sampler if world_size > 1 else None,
            shuffle=(sampler is None),
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
            multiprocessing_context=mp.get_context("spawn"),
        )

        # Only set these when workers > 0 (PyTorch requires it)
        if num_workers > 0:
            kwargs["worker_init_fn"] = partial(seed_worker, base_seed=base_seed, rank=rank)
            kwargs["persistent_workers"] = persistent
            if prefetch is not None:
                kwargs["prefetch_factor"] = int(prefetch)

        return DataLoader(**kwargs)

    def make_valid_loader(self, world_size, rank):
        cfg = self.valid_dataloader_config
        num_workers = int(cfg.get('num_workers', 4))
        pin_memory  = bool(cfg.get('pin_memory', True))
        persistent  = bool(cfg.get('persistent_workers', False))
        prefetch    = cfg.get('prefetch_factor', 2)
        base_seed   = int(cfg.get('seed', 42))

        sampler = DistributedSampler(
            self.valid_dataset,
            num_replicas=world_size, rank=rank,
            shuffle=True, drop_last=True
        ) if world_size > 1 else None

        kwargs = dict(
            dataset=self.valid_dataset,
            batch_size=cfg.get('batch_size', 16),
            sampler=sampler if world_size > 1 else None,
            shuffle=(sampler is None),
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
            multiprocessing_context=mp.get_context("spawn"),
        )

        # Only set these when workers > 0 (PyTorch requires it)
        if num_workers > 0:
            kwargs["worker_init_fn"] = partial(seed_worker, base_seed=base_seed, rank=rank)
            kwargs["persistent_workers"] = persistent
            if prefetch is not None:
                kwargs["prefetch_factor"] = int(prefetch)

        return DataLoader(**kwargs)
