import copy

import torch
import torch.distributed as dist
from torch.utils.data import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from trainer.loggers.tensorboard import TensorBoardLogger
from trainer.utils.build_components import build_module

from contextlib import nullcontext
from datetime import datetime
from tqdm import tqdm

from trainer.utils.ddp_utils import (
    ddp_setup_env,
    seed_everything,
    wrap_models_with_ddp,
    broadcast_module_state,
    is_rank0,
    is_dist_ready,
    get_primary_trainable_module,
)

def ddp_train_worker(
    rank,
    world_size,
    addr,
    port,
    devices,
    boat_template,
    trainer_config,
    data_module,
    train_states, 
    callback_configs,
):
    ddp_setup_env(rank, world_size, addr, port)

    use_cuda = True
    device = torch.device(f"cuda:{devices[rank]}")
    if use_cuda:
        torch.cuda.set_device(device)

    backend = "nccl" if use_cuda else "gloo"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    seed_everything(trainer_config.get('seed', 42), rank)

    # Build per-rank boat & wrap with DDP
    boat = copy.deepcopy(boat_template)
    boat.attach_global_step(train_states['global_step'])
    boat.build_optimizers()
    boat.build_losses()
    boat.build_metrics()
    boat.to(device)

    wrap_models_with_ddp(boat, device, fup_by_key=trainer_config.get('fup_by_key'), broadcast_buffers=True)

    # Identify gradient-free modules as EMA-like (or mark them in your code with a flag)
    ema_keys = [
        k for k, m in boat.models.items()
        if isinstance(m, torch.nn.Module) and not any(p.requires_grad for p in m.parameters()) and 'ema' in k
    ]

    # Build utils for GPU-0 Monitoring
    if is_rank0():
        callbacks   = [build_module(cb) for cb in (callback_configs or [])]
        logger     = TensorBoardLogger(log_dir=train_states['run_folder'],
                                       name=train_states['experiment_name'])
        train_states['valid_epoch_records'] = {}
        train_states['valid_step_records'] = {}

    primary = get_primary_trainable_module(boat)

    # Optionally initialize EMA from primary, then broadcast once so every rank matches
    if ema_keys and primary is not None and is_dist_ready():
        if is_rank0():
            src = primary.module if isinstance(primary, DDP) else primary
            for k in ema_keys:
                try:
                    # params only; optionally also copy buffers to start identical
                    boat.models[k].load_state_dict(src.state_dict(), strict=False)
                except Exception:
                    pass
        for k in ema_keys:
            broadcast_module_state(boat.models[k], src_rank=0)

    precision = trainer_config.get('precision', None)
    # AMP context + scaler
    if precision is None:
        amp_dtype = None
    elif precision == "bf16-mixed":
        amp_dtype = torch.bfloat16
    elif precision == "16-mixed":
        amp_dtype = torch.float16
    else:
        amp_dtype = None

    if amp_dtype is not None and device.type == "cuda":
        autocast_ctx = lambda: torch.autocast(device_type="cuda", dtype=amp_dtype)
    else:
        autocast_ctx = lambda: nullcontext()

    if precision is not None:
        scaler = torch.cuda.amp.GradScaler(enabled=(precision == "16-mixed"))
    else:
        scaler = None
    
    # Dataloaders
    train_loader = data_module.make_train_loader(world_size, rank)
    valid_loader = data_module.make_valid_loader() if is_rank0() else None

    # Train
    for epoch in range(trainer_config.get("max_epochs", 10)):
        if isinstance(getattr(train_loader, "sampler", None), DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        if is_rank0():
            for cb in callbacks:
                cb.on_epoch_start(train_states, boat, epoch)

        boat.train()

        print(f"rank_{rank} before train_loader")
        iterator = enumerate(train_loader)
        print(f"rank_{rank} after train_loader")
        
        total_batches = len(train_loader) if hasattr(train_loader, "__len__") else None
        if is_rank0() and tqdm is not None:
            iterator = tqdm(iterator, total=total_batches, unit="batch",
                            desc=f"Epoch {epoch} | loss N/A | {datetime.now():%Y-%m-%d %H:%M:%S}")

        for step, (batch_idx_batch) in enumerate(iterator, start=1):

            # dataloader yields dict already; support both tuple or single
            if isinstance(batch_idx_batch, tuple):
                batch_idx, batch = batch_idx_batch
            else:
                batch_idx, batch = step - 1, batch_idx_batch

            # bump global step
            if boat.get_global_step() > 0 and epoch == 0 and batch_idx == 0:
                boat.global_step -= 1
            boat.global_step += 1

            if is_rank0():
                for cb in callbacks:
                    cb.on_batch_start(train_states, boat, batch, batch_idx)

            # no_sync across all DDP modules until boundary; AMP is trainer-owned
            with autocast_ctx():
                losses = boat.training_step(
                    batch, batch_idx, epoch, 
                    scaler=scaler,
                )

            # --- start of EMA ---
            if ema_keys and primary is not None:
                for k in ema_keys:
                    broadcast_module_state(boat.models[k], src_rank=0)
            # ---  end of EMA  ---

            total_loss = losses.get("total_loss", None)
            loss_scalar = torch.tensor(0.0, device=device)

            if total_loss is not None:
                loss_scalar = total_loss.detach()
                
                if dist.is_initialized():
                    dist.all_reduce(loss_scalar, op=dist.ReduceOp.SUM)
                    loss_scalar /= world_size
                
                if is_rank0():
                    boat.log_train_losses(logger, losses)

                    if tqdm is not None:
                        iterator.set_description(
                            f"Epoch {epoch} | loss {loss_scalar.item():.4f} | {datetime.now():%Y-%m-%d %H:%M:%S}"
                        )

            if is_rank0():
                for cb in callbacks:
                    cb.on_batch_end(train_states, boat, batch, batch_idx, total_loss)

        # validation
        if is_rank0():

            if train_states['val_check_epochs'] is not None and epoch % train_states['val_check_epochs'] == 0:

                for cb in callbacks:
                    cb.on_validation_start(train_states, boat)

                avg_loss = run_validation(boat, valid_loader, train_states, device, logger)
                train_states['valid_epoch_records'][epoch] = {'avg_loss': avg_loss.detach().cpu()}

                for cb in callbacks:
                    cb.on_validation_end(train_states, boat)

            if train_states['state_save_epochs'] is not None and epoch % train_states['state_save_epochs'] == 0:
                state_path = boat.save_state(train_states['run_folder'], 'boat_state', global_step=train_states['global_step'](), epoch=epoch+1)
                
                if epoch not in train_states['valid_epoch_records']:
                    train_states['valid_epoch_records'][epoch] = {}
                train_states['valid_epoch_records'][epoch]['state_path'] = state_path

            for cb in callbacks:
                cb.on_epoch_end(train_states, boat, epoch)
    
    if is_rank0():
        for cb in callbacks:
            cb.on_train_end(train_states, boat)

    dist.destroy_process_group()

def run_validation(boat, val_dataloader, train_states, device, logger):

    boat.eval()
    aggr_metrics = {}
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            metrics, named_imgs = boat.validation_step(batch, batch_idx)
            # average each metric in metrics
            for key, value in metrics.items():
                if key not in aggr_metrics:
                    aggr_metrics[key] = metrics[key]
                else:
                    aggr_metrics[key] += metrics[key]
    
            boat.visualize_validation(logger, named_imgs, batch_idx)

        for key in aggr_metrics:
            aggr_metrics[key] /= batch_idx

    if not aggr_metrics: raise ValueError("Validation loop produced no losses.")

    boat.log_valid_metrics(logger, aggr_metrics)
    
    if train_states['target_metric_name'] not in aggr_metrics:
        raise KeyError(f"'{train_states['target_metric_name']}' not found in validation metrics: {list(aggr_metrics)}")

    return aggr_metrics[train_states['target_metric_name']]

