import types
from trainer.rhino_train import main

args_dict = {
    'config': 'configs/restoration/train_sr3_dit_dinov2_base_ffhq_32_256.yaml',
    'resume_from': None,
}

args = types.SimpleNamespace(**args_dict)

main(args)