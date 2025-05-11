import json
import os
from diffusers.torch_datasets.basic_image_dataset import BasicImageDataset
import torch


class ConditionedImageDataset(BasicImageDataset):
    
    def __init__(self, dataset_config):
        # Initialize the parent class
        json_path = dataset_config.pop('json_path', None)
        condition_keys = dataset_config.pop('condition_keys', None)
        
        super().__init__(dataset_config)
        
        # Store the condition filter
        self.condition_keys = condition_keys

        # Load the condition information from JSON
        self.conditions = self._load_conditions(json_path)

        
    def _load_conditions(self, json_path):
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Condition JSON file not found: {json_path}")
            
        with open(json_path, 'r') as f:
            conditions_data = json.load(f)
            
        conditions = {}
        
        if conditions_data:
            all_feature_keys = list(conditions_data[0]['features'].keys())
        else:
            all_feature_keys = []
        
        for item in conditions_data:
            image_id = item['image_id']
            features = item['features']
            
            # Create a list to hold all values in a consistent order
            all_values = []
            for key in self.condition_keys:
                value = features.get(key, None)
                if value is not None:
                    if isinstance(value, list):
                        all_values.extend(value)  # Extend with all elements from the list
                    else:
                        all_values.append(value)  # Append the single value
            
            # Convert the list to a tensor
            conditions[image_id] = torch.tensor(all_values, dtype=torch.float32)

        backup_condition = next(iter(conditions.values()))
        
        for image_key in self.image_keys:
            image_id = os.path.splitext(image_key)[0].split('_')[0]
            if image_id not in conditions:
                conditions[image_id] = torch.zeros_like(backup_condition)

        print(f"Loaded conditions for {len(conditions)} images from {json_path}")
        return conditions
        
    def _get_conditions(self, key):
        """
        Get the conditions for a specific image, optionally filtered.
        
        Args:
            key: Image key
            
        Returns:
            Tensor containing condition information
        """
        # Extract image_id from the filename
        image_id = os.path.splitext(key)[0].split('_')[0]
        
        # Get the full condition tensor
        condition_tensor = self.conditions.get(image_id, None)
        
        
        return condition_tensor
        
    def __getitem__(self, idx):
        """
        Get an item from the dataset including both image data and conditions.
        
        Args:
            idx: Index of the item to get
            
        Returns:
            Dictionary containing image data and conditions
        """
        # Get the key at the specified index
        key = self.image_keys[idx]
        
        # Get the image data from parent class
        data = super().__getitem__(idx)
        
        # Get condition information
        conditions = self._get_conditions(key)

        data.update({'cond': conditions.unsqueeze(0)}) # Add condition tensor to the data dictionary, unsqueeze(0) is for seq length [T, D]
        
        return data
    
if __name__=="__main__":
    import yaml
    from torch.utils.data import DataLoader

    # Load YAML config
    with open('configs/cldm_train_64.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Extract dataset config
    dataset_config = config['data']['train_dataloader']['dataset']['config']

    json_path = 'ffhq_disentangled_features.json'

    # Initialize dataset
    dataset = ConditionedImageDataset(dataset_config)

    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=2,  # Small batch for testing
        shuffle=True,
        num_workers=0  # Set to 0 for simplicity in testing
    )

    # Test one batch
    for batch in dataloader:
        print("Batch keys:", batch.keys())
        print("GT shape:", batch['gt'].shape)
        break  # Only process one batch for testing