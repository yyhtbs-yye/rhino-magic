import types
from trainer.rhino_ddp_train import main

args_dict = {
    'config': 'configs/generation/autoregression/train_vqvae_ffhq_256.yaml',
    'resume_from': 'work_dirs/vqvae_ffhq_256/run_12/last.pt',
}

args = types.SimpleNamespace(**args_dict)

if __name__ == "__main__":

    main(args)
