import os
import torch
import yaml

from trainer.data_modules.image_data_module import SimpleTrainValidDataModule
from rhino.diffusers.boats.superresolution.latent_diffusing import ConditionedLatentDiffusionBoat

torch.set_float32_matmul_precision('high')

# Path to configuration file
config_path = "configs/train_cldm_64.yaml"

# Load YAML configuration
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Extract configurations
model_config = config['model']
validation_config = config['validation']
train_config = config['train']
data_config = config['data']
logging_config = config['logging']
checkpoint_config = config['checkpoint']

# Create a new model or load from checkpoint
checkpoint_path = "last-v7.ckpt"
if checkpoint_path and os.path.exists(checkpoint_path):
    print(f"Loading model from checkpoint: {checkpoint_path}")
    model = ConditionedLatentDiffusionBoat.load_from_checkpoint(
        checkpoint_path, 
        model_config=model_config, 
        train_config=train_config,
        validation_config=validation_config,
    )
else:
    print("Creating new model")
    model = ConditionedLatentDiffusionBoat(
        model_config=model_config,
        train_config=train_config,
        validation_config=validation_config
    )

# Create data module
data_module = SimpleTrainValidDataModule(data_config)

# Set up logger
logger = TensorBoardLogger(
    save_dir=logging_config['log_dir'],
    name=logging_config['experiment_name']
)

# Set up callbacks
callbacks = []

# Checkpoint callback
checkpoint_callback = ModelCheckpoint(
    dirpath=os.path.join(logging_config['log_dir'], 
                         logging_config['experiment_name'], 
                        'checkpoints'),
    filename='{epoch:02d}-{' + checkpoint_config['save_best_metric'] + ':.4f}',
    monitor=checkpoint_config['save_best_metric'],
    mode=checkpoint_config['save_best_mode'],
    save_top_k=checkpoint_config['save_top_k'],
    save_last=checkpoint_config['save_last']
)

callbacks.append(checkpoint_callback)

# Learning rate monitor
lr_monitor = LearningRateMonitor(logging_interval='step')
callbacks.append(lr_monitor)

# Initialize trainer
trainer = Trainer(
    max_epochs=train_config['max_epochs'],
    accelerator="auto",
    devices=4 if torch.cuda.is_available() else None,
    strategy=DDPStrategy(
        find_unused_parameters=False,  # Optional: Disable if not needed
        timeout=timedelta(seconds=3600)  # Set timeout to 1 hour
    ),
    logger=logger,
    callbacks=callbacks,
    check_val_every_n_epoch=train_config['val_check_interval'],
    log_every_n_steps=logging_config['log_every_n_steps']
)

# Train model
trainer.fit(model=model, datamodule=data_module)