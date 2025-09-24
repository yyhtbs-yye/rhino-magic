import types
from trainer.rhino_ddp_train import main

args_dict = {
    'config': 'configs/restoration/train_sr3_unet_ldm_ffhq_32_256.yaml',
    'resume_from': None,
}

args = types.SimpleNamespace(**args_dict)

if __name__ == "__main__":

    main(args)