import torch

def load_state(state_path, target_name):
    """
    """

    if not state_path.exists():
        raise FileNotFoundError(f"No state file found at {state_path}")
    
    # Load the state dictionary
    state = torch.load(state_path, map_location=torch.device('cpu'))
    
    # Load model weights
    model_states = state['model_states']
    for name, state_dict in model_states.items():
        if name == target_name:
            return state_dict

def save_pretrained(state_dict, save_path):
    torch.save(state_dict, save_path)
    print(f"Pretrained model state saved to {save_path}")

if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Convert training state to pretrained model state")
    parser.add_argument('--state_path', type=str, default='work_dirs/vq_ffhq_256/run_4/last.pt', help='Path to the full training state file (e.g., last.pt)')
    parser.add_argument('--target_name', type=str, default='vector_quantizer', help='Name of the model to extract (e.g., "vector_quantizer")')
    parser.add_argument('--save_path', type=str, default='pretrained/pixel_vq_model_d4k_ffhq256.pt', help='Path to save the pretrained model state (e.g., vq_model_pretrained.pt)')
    
    args = parser.parse_args()
    
    state_path = Path(args.state_path)
    save_path = Path(args.save_path)
    
    state_dict = load_state(state_path, args.target_name)
    save_pretrained(state_dict, save_path)

    # Example usage:
    # python utils/cvt_state_to_pretrained.py --state_path path/to/last.pt