import types
from trainer.rhino_train import main

args_dict = {
    'config': 'configs/restoration/train_hat_ffhq_32_256_tiny.yaml',
    'resume_from': 'work_dirs/hat_ffhq_32_256/run_1/boat_state_step=1094_epoch=1.pt',
}

args = types.SimpleNamespace(**args_dict)

main(args)