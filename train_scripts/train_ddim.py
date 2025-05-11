import os
import torch
import yaml
from trainer.simple_trainer import Trainer

from trainer.data_modules.image_data_module import SimpleTrainValidDataModule
from rhino.diffusers.boats.unconditioned.pixel_diffusing import UnconditionedPixelDiffusionBoat 

from trainer.loggers.tensorboard import TensorBoardLogger
from trainer.utils.path_helpers import determine_run_folder
from trainer.callbacks.state_cleaner import KeepTopKStateCallback
from pathlib import Path

torch.set_float32_matmul_precision('high')
model_type = 'ddim'
resolution = 256
top_k = 5

# Path to configuration file
config_path = f'configs/train_{model_type}_{resolution}.yaml'  # Updated config path

resume_from = None
# Load YAML configuration
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Extract configurations
boat_config = config['boat']
optimization_config = config['optimization']
validation_config = config['validation']
trainer_config = config['trainer']
data_config = config['data']
logging_config = config['logging']

boat = UnconditionedPixelDiffusionBoat(
    boat_config=boat_config,
    optimization_config=optimization_config,
    validation_config=validation_config,
)

callbacks = [KeepTopKStateCallback(top_k)]

# Create data module
data_module = SimpleTrainValidDataModule(data_config)

if resume_from is None:
    root_dir = Path(f"work_dirs/{model_type}_{resolution}")

    # Create the root directory if it doesn't exist
    if not root_dir.exists():
        os.makedirs(root_dir)

    run_folder = determine_run_folder(root_dir)
else:
    run_folder = Path(resume_from).parent
    
# Set up logger
logger = TensorBoardLogger(
    log_dir=run_folder,
    name=logging_config['experiment_name']
)

# Initialize trainer
trainer = Trainer(boat=boat, trainer_config=trainer_config, device='cuda:3', callbacks=callbacks,
                  logger=logger, run_folder=run_folder, resume_from=resume_from,
)

trainer.fit(data_module=data_module)