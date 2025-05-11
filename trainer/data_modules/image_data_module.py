from torch.utils.data import DataLoader
from .samplers import DefaultSampler
from trainer.utils.build_components import build_dataset


class SimpleTrainValidDataModule():
    
    def __init__(self, data_config=None):
        super().__init__()
        
        self.data_config = data_config

        train_dataloader_config = self.data_config['train_dataloader']
        self.train_dataset = build_dataset(train_dataloader_config['dataset'])     
        train_sampler_config = train_dataloader_config.get('sampler', {'type': 'DefaultSampler', 'shuffle': False})
        if train_sampler_config['type'] == 'DefaultSampler':
            train_sampler = DefaultSampler(
                dataset_size=len(self.train_dataset),
                shuffle=train_sampler_config.get('shuffle', False)
            )
        self.train = DataLoader(
            dataset=self.train_dataset,
            batch_size=train_dataloader_config.get('batch_size', 16),
            num_workers=train_dataloader_config.get('num_workers', 4),
            persistent_workers=train_dataloader_config.get('persistent_workers'),
            pin_memory=train_dataloader_config.get('pin_memory'),
            prefetch_factor=train_dataloader_config.get('prefetch_factor', None),
            sampler=train_sampler
        )
        
        valid_dataloader_config = self.data_config['valid_dataloader']
        self.valid_dataset = build_dataset(valid_dataloader_config['dataset'])
        valid_sampler_config = valid_dataloader_config.get('sampler', {'type': 'DefaultSampler', 'shuffle': False})
        if valid_sampler_config['type'] == 'DefaultSampler':
            valid_sampler = DefaultSampler(
                dataset_size=len(self.valid_dataset),
                shuffle=valid_sampler_config.get('shuffle', False)
            )

        self.valid = DataLoader(
            dataset=self.valid_dataset,
            batch_size=valid_dataloader_config.get('batch_size', 16),
            num_workers=valid_dataloader_config.get('num_workers', 4),
            persistent_workers=valid_dataloader_config.get('persistent_workers'),
            pin_memory=valid_dataloader_config.get('pin_memory'),
            prefetch_factor=valid_dataloader_config.get('prefetch_factor', None),
            sampler=valid_sampler
        )
