import types
from trainer.rhino_train import main

args_dict = {
    'config': 'configs/restoration/discriminative_one_pass/train_hat_cyclone_wt_ffhq_32_256_tiny.yaml',
    'resume_from': 'work_dirs/hat_cyclone_ffhq_32_256/run_2/last.pt'
}

args = types.SimpleNamespace(**args_dict)

main(args)