import types
from trainer.rhino_train import main

args_dict = {
    'config': 'configs/restoration/train_hat_ffhq_32_256_tiny.yaml',
    'resume_from': None
}

args = types.SimpleNamespace(**args_dict)

main(args)