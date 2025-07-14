# trainer/rhino_train.py

import os
import torch
import yaml
from pathlib import Path
from trainer.loggers.tensorboard import TensorBoardLogger
from trainer.utils.path_helpers import determine_run_folder
from trainer.utils.build_components import get_class, build_module
from trainer.utils.yaml_reader import load_yaml_config

torch.set_float32_matmul_precision('high')

def main(args):
    config = load_yaml_config(args.config)
    use_multigpu = config.get('use_multigpu', True)

    # build boat, datamodule, callbacks, logger, etc…
    boat_conf = config['boat']
    Boat = get_class(boat_conf['path'], boat_conf['name'])
    boat = Boat(boat_config=boat_conf,
                optimization_config=config['optimization'],
                validation_config=config['validation'])
    callbacks   = [build_module(cb) for cb in config.get('callbacks', [])]
    data_module = build_module(config['data'])

    root_dir = Path(config['logging']['root_dir']) / config['logging']['experiment_name']
    root_dir.mkdir(parents=True, exist_ok=True)
    run_folder = determine_run_folder(root_dir)
    logger     = TensorBoardLogger(log_dir=run_folder,
                                   name=config['logging']['experiment_name'])

    trainer_cfg = config['trainer'].copy()
    
    # Check if we're in a distributed environment (torchrun sets these)
    if use_multigpu:
        from trainer.multigpu_trainer import MultiGPUTrainer as Trainer
        # devices = trainer_cfg.get('devices', list(range(min(9, torch.cuda.device_count()))))
        devices = trainer_cfg.get('devices', [0, 9, 10, 11])
    else:
        # Single process training
        from trainer.simple_trainer import Trainer
        devices = None

    trainer = Trainer(
        boat=boat,
        trainer_config=trainer_cfg,
        callbacks=callbacks,
        logger=logger,
        run_folder=run_folder,
        resume_from=args.resume_from,
        devices=devices,
    )
    trainer.fit(data_module=data_module)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--resume_from', help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    main(args)