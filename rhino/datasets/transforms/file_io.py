from PIL import Image
import numpy as np
import torch

class LoadImageFromFile:
    """Load an image from file and convert to tensor."""
    
    def __init__(self, keys, to_tensor=True):
        # Support both single key or list of keys
        if isinstance(keys, str):
            self.keys = [keys]
        else:
            self.keys = keys
        self.to_tensor = to_tensor
        
    def __call__(self, results):
        """Load image from file and convert to tensor."""
        for key in self.keys:
            
            filepath = results[f"{key}_path"]
            
            try:
                img = Image.open(filepath).convert('RGB')
                img = np.array(img)
                
                # Convert numpy array to tensor if to_tensor=True
                if self.to_tensor:
                    # Convert HWC to CHW format (height, width, channels) -> (channels, height, width)
                    img = img.transpose(2, 0, 1)
                    # Convert to PyTorch tensor
                    img = torch.from_numpy(img).float() / 255.0
                
            except Exception as e:
                raise Exception(f'Failed to load image: {filepath}, {e}')
                
            # Store the loaded image and related information in results
            results[key] = img
            results[f'{key}_ori_shape'] = img.shape
            
        return results

class LoadArrayFromNpz:
    """Load array(s) from a .npz file and convert to tensor."""

    def __init__(self, keys, to_tensor=True):
        # Accept either a single key or a list of keys
        if isinstance(keys, str):
            self.keys = [keys]
        else:
            self.keys = keys
        self.to_tensor = to_tensor

    def __call__(self, results):
        for key in self.keys:

            """Load array from .npz file and convert to tensor."""
            filepath = results[f"{key}_path"]
            if not filepath:
                raise ValueError(f"{key}_path must be provided in the results dictionary.")

            try:
                npz_file = np.load(filepath)
            except Exception as e:
                raise Exception(f"Failed to load .npz file: {filepath}, {e}")

            array = npz_file.f.patch_tokens

            # Convert to tensor if needed
            if self.to_tensor:
                array = torch.from_numpy(array).float()

            # Store in results
            results[key] = array
            results[f"{key}_ori_shape"] = array.shape

            npz_file.close()
        return results
