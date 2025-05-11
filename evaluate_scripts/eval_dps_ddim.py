import os
import torch
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from trainer.data_modules.image_data_module import SimpleTrainValidDataModule
from rhino.diffusers.boats.superresolution.dps_pixel_diffusing import DPSPixelDiffusionModel

# Path to configuration file
config_path = "configs/eval_dps_ddim_ffhq_10m_256.yaml"

# Load YAML configuration
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Extract configurations
model_config = config['model']
validation_config = config['validation']
logging_config = config['logging']
data_config = config['data']

print("Creating new model")
model = DPSPixelDiffusionModel(
    model_config=model_config,
    train_config={},
    validation_config=validation_config
)

# Create data module
data_module = SimpleTrainValidDataModule(data_config)
data_module.setup("validate")  # Only setup the validation part

# Set up logger
logger = TensorBoardLogger(
    save_dir=logging_config['log_dir'],
    name=logging_config['experiment_name']
)

# Initialize trainer for evaluation only
trainer = Trainer(
    inference_mode=False,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,  # Single device for consistent evaluation
    logger=logger,  # No logging needed for evaluation
    enable_checkpointing=False,  # Disable checkpointing
    enable_progress_bar=True  # Show progress during evaluation
)

# Run validation
print("Starting validation...")
validation_results = trainer.validate(model=model, datamodule=data_module)
print(f"Validation results: {validation_results}")

# Optionally, generate some samples
samples_dir = "samples"
os.makedirs(samples_dir, exist_ok=True)

# Generate a fixed number of samples
num_samples = 10
print(f"Generating {num_samples} samples...")

# Set model to eval mode
model.eval()
with torch.no_grad():
    for i in range(num_samples):
        # Generate single sample
        sample = model.sample(batch_size=1)
        
        # Save the sample
        from torchvision.utils import save_image
        save_image(sample[0], os.path.join(samples_dir, f"sample_{i}.png"), normalize=True)

print(f"Samples saved to {samples_dir}")