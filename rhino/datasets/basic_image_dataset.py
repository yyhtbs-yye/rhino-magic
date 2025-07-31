from torch.utils.data import Dataset
from rhino.datasets import transforms
from rhino.datasets.utils.folder_utils import ScanFiles, ReshapeByFilename

class BasicImageDataset(Dataset):
    
    def __init__(self, **dataset_config):
        super().__init__()

        # Extract configuration
        self.folder_paths = dataset_config.get('folder_paths', {})
        self.data_prefix = dataset_config.get('data_prefix', {})
        self.extensions = dataset_config.get('extensions', {})
        self.depth = dataset_config.get('depth', 1)

        """Use ScanFiles and ReshapeByFilename to get grouped image paths."""
        scanner = ScanFiles(self.depth)
        reshape = ReshapeByFilename()

        raw_paths = scanner(self.folder_paths, self.extensions, self.data_prefix)
        
        self.image_paths = reshape(raw_paths)

        if not self.image_paths:
            raise RuntimeError("No valid image groups found after reshaping. Check extensions and folder paths.")

        # Load pipeline
        pipeline_cfg = dataset_config.get('pipeline', [])
        self.transform_pipeline = self._build_pipeline(pipeline_cfg)

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
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        data = self.image_paths[idx]
        
        # Apply transforms
        for transform in self.transform_pipeline:
            data = transform(data)
            
        return data
