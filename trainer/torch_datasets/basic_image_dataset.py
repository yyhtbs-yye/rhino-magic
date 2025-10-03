import os
from pathlib import Path
from torch.utils.data import Dataset
from trainer.torch_datasets import transforms

class BasicImageDataset(Dataset):
    """Simplified paired image dataset for training/inference with image_paths as a dictionary."""
    
    def __init__(self, **dataset_config):
        super().__init__()
        
        # Extract configuration from dataset_config dictionary
        self.folder_paths = dataset_config.get('folder_paths', {})
        self.data_prefix = dataset_config.get('data_prefix', {})
        
        self.max_dataset_size = dataset_config.get('max_dataset_size', None)

        # Now image_paths will be a dictionary instead of a list
        self.image_paths = self._scan_images()

        pipeline_cfg = dataset_config.get('pipeline', [])
        self.transform_pipeline = self._build_pipeline(pipeline_cfg)
        # Convert dictionary keys to a list to enable indexing
        self.image_keys = list(self.image_paths.keys())


    def _scan_images(self):
        """Scan images in all folders and return their intersection as a dictionary."""
        extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        paths_by_key = {}
        base_names_by_key = {}
        
        # First, collect all image folder_paths for each key
        for key in self.folder_paths:
            folder_path = self.folder_paths[key]
            path_prefix = os.path.join(folder_path, self.data_prefix.get(key, ''))
            image_paths = []
            
            for ext in extensions:
                folder_paths = list(Path(path_prefix).glob(f'**/*{ext}'))
                image_paths.extend([str(path) for path in folder_paths])
                            
            if len(image_paths) == 0:
                raise ValueError(f'No images found in {folder_path}')
                
            # Store folder_paths and extract base filenames for intersection check
            paths_by_key[key] = image_paths
            base_names_by_key[key] = {os.path.basename(p): p for p in image_paths}
            
            print(f'Found {len(image_paths)} images in {folder_path}')
        
        # Find common base filenames across all folders
        all_keys = list(self.folder_paths.keys())
        if not all_keys:
            raise ValueError("No folder_paths specified")
            
        # Start with all base filenames from the first key
        common_base_names = set(base_names_by_key[all_keys[0]].keys())
        
        # Find intersection with all other keys
        for key in all_keys[1:]:
            common_base_names &= set(base_names_by_key[key].keys())
        
        if not common_base_names:
            raise ValueError(f"No common images found across all folders")
        
        # Sort the common base names to maintain consistent order
        common_base_names = sorted(common_base_names)
        
        # Create a dictionary where keys are the base names and values are dictionaries of folder_paths
        result = {}
        for i, base_name in enumerate(common_base_names):
            paths_dict = {}
            for key in self.folder_paths:
                paths_dict[f"{key}_path"] = base_names_by_key[key][base_name]

            result[base_name] = paths_dict
            if self.max_dataset_size is not None and i + 1 >= self.max_dataset_size:
                break
        print(f'Found {len(result)} common images across all folders')
        return result

    def _build_pipeline(self, pipeline_cfg):
        """Build the data processing pipeline using getattr for dynamic class loading."""
        
        transforms_list = []
        
        for transform_cfg in pipeline_cfg:
            transform_cfg = transform_cfg.copy()  # Create a copy to avoid modifying original
            transform_type = transform_cfg.pop('type')
           
            # Get the class from the module using getattr
            transform_class = getattr(transforms, transform_type)
            
            # Create an instance of the transform class
            transform = transform_class(**transform_cfg)
            transforms_list.append(transform)
            
        return transforms_list
        
    def __len__(self):
        if self.max_dataset_size is not None:
            return min(len(self.image_paths), self.max_dataset_size)
        
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        # Get the key at the specified index
        key = self.image_keys[idx]
        # Get the data for this key
        data = self.image_paths[key]
        
        # Apply transforms
        for transform in self.transform_pipeline:
            data = transform(data)
            
        return data
