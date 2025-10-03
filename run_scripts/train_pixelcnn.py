import types
from trainer.rhino_ddp_train import main

args_dict = {
    'config': 'configs/generation/ar/train_pixelcnn_ffhq_32.yaml',
    'resume_from': None,
}

args = types.SimpleNamespace(**args_dict)

if __name__ == "__main__":

    main(args)


