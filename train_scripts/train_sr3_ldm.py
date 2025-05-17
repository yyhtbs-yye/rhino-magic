import types
from trainer.rhino_train import main

# Define named dictionary with arguments
args_dict = {
    'config': 'configs/restoration/train_sr3_unet_ldm_64_256.yaml',
    'resume_from': None
}

# Convert dictionary to object with dot notation
args = types.SimpleNamespace(**args_dict)

# Call main function
main(args)