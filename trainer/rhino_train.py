import os
import argparse
import torch
import yaml
from trainer.simple_trainer import Trainer
from trainer.loggers.tensorboard import TensorBoardLogger
from trainer.utils.path_helpers import determine_run_folder
from pathlib import Path
from trainer.utils.build_components import get_class, build_module
from trainer.utils.yaml_reader import load_yaml_config


torch.set_float32_matmul_precision('high')

def main(args):
    """
    Main training function for setting up and running the training pipeline.

    Args:
        args: Object with 'config' (path to YAML file) and optional 'resume_from' (path to checkpoint) attributes.
    """
    try:
        # Load YAML configuration
        config = load_yaml_config(args.config)
    except FileNotFoundError:
        print(f"Error: Config file '{args.config}' not found.")
        return
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML in config file: {e}")
        return
    except AttributeError:
        print("Error: 'config' attribute missing in args.")
        return

    boat_config = config['boat']
    class_ = get_class(boat_config['path'], boat_config['name'])

    # Extract configurations
    optimization_config = config['optimization']
    validation_config = config['validation']
    trainer_config = config['trainer']
    data_config = config['data']
    logging_config = config['logging']

    boat = class_(boat_config=boat_config,
                  optimization_config=optimization_config,
                  validation_config=validation_config)

    callbacks = [build_module(callback) for callback in config['callbacks']]
    data_module = build_module(data_config)

    root_dir = Path(config['logging']['root_dir']) / config['logging']['experiment_name']
    
    # Create the root directory if it doesn't exist
    if not root_dir.exists():
        os.makedirs(root_dir)
    
    run_folder = determine_run_folder(root_dir)
    
    # Set up logger
    logger = TensorBoardLogger(
        log_dir=run_folder,
        name=logging_config['experiment_name']
    )

    # Initialize trainer
    trainer = Trainer(
        boat=boat,
        trainer_config=trainer_config,
        callbacks=callbacks,
        logger=logger,
        run_folder=run_folder,
        resume_from=args.resume_from
    )

    trainer.fit(data_module=data_module)

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Training script with configuration')
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file')
    parser.add_argument('--resume_from', type=str, default=None, help='Path to checkpoint to resume training from')
    args = parser.parse_args()
    main(args)