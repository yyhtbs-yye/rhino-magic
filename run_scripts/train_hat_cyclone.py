import types
from trainer.rhino_ddp_train import main

args_dict = {
    'config': 'configs/restoration/discriminative_one_pass/train_hat_cyclone_ffhq_32_256_tiny.yaml',
    'resume_from': 'work_dirs/hat_cyclone_ffhq_32_256/run_2/last.pt',
}

args = types.SimpleNamespace(**args_dict)

if __name__ == "__main__":

    main(args)