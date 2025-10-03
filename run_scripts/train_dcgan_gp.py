import types
from trainer.rhino_ddp_train import main

args_dict = {
    'config': 'configs/generation/gan/train_dcgan_gp_ffhq_256.yaml',
    'resume_from': 'work_dirs/dcgan_gp_ffhq_256/run_1/last.pt',
}

args = types.SimpleNamespace(**args_dict)

if __name__ == "__main__":

    main(args)