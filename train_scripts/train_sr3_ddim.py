import types
from trainer.rhino_train import main

# Define named dictionary with arguments
args_dict = {
    'config': 'configs/restoration/train_sr3_unet_ddim_16_128.yaml',
    'resume_from': None #'work_dirs/sr3_ddim_32_256/run_1/boat_state_step=253748_epoch=29.pt'
}

# Convert dictionary to object with dot notation
args = types.SimpleNamespace(**args_dict)

# Call main function
main(args)