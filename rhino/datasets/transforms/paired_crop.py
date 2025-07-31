import torch

class PairedCrop:
    def __init__(self, crop_size, is_pad_zeros=True, random_crop=False, scale_factor=4):
        self.crop_size = crop_size if isinstance(crop_size, list) else [crop_size, crop_size]
        self.is_pad_zeros = is_pad_zeros
        self.random_crop = random_crop
        self.scale_factor = scale_factor
        
    def __call__(self, results):
        # Validate that both 'gt' and 'lr' keys exist
        if 'gt' not in results or 'lr' not in results:
            raise KeyError("Both 'gt' and 'lr' keys must be present in results")
        
        lr_img = results['lr']
        gt_img = results['gt']

        # Get dimensions
        c_lr, h_lr, w_lr = lr_img.shape
        c_gt, h_gt, w_gt = gt_img.shape

        self.scale_factor = h_gt // h_lr

        assert h_gt % h_lr == 0 and w_gt % w_lr == 0 and h_gt // h_lr == w_gt // w_lr

        # LR crop size (as specified in init)
        lr_crop_h, lr_crop_w = self.crop_size
        
        # GT crop size (scaled up from LR crop size)
        gt_crop_h = lr_crop_h * self.scale_factor
        gt_crop_w = lr_crop_w * self.scale_factor
        
        if self.random_crop:
            # Random crop - calculate offsets for LR, then scale for GT
            lr_x_offset = torch.randint(0, max(0, w_lr - lr_crop_w) + 1, (1,)).item()
            lr_y_offset = torch.randint(0, max(0, h_lr - lr_crop_h) + 1, (1,)).item()
            
            gt_x_offset = lr_x_offset * self.scale_factor
            gt_y_offset = lr_y_offset * self.scale_factor
        else:
            # Center crop
            lr_x_offset = max(0, (w_lr - lr_crop_w) // 2)
            lr_y_offset = max(0, (h_lr - lr_crop_h) // 2)
            
            gt_x_offset = max(0, (w_gt - gt_crop_w) // 2)
            gt_y_offset = max(0, (h_gt - gt_crop_h) // 2)
        
        # Calculate crop boundaries for LR
        lr_crop_y1, lr_crop_y2 = lr_y_offset, min(lr_y_offset + lr_crop_h, h_lr)
        lr_crop_x1, lr_crop_x2 = lr_x_offset, min(lr_x_offset + lr_crop_w, w_lr)
        
        # Calculate crop boundaries for GT
        gt_crop_y1, gt_crop_y2 = gt_y_offset, min(gt_y_offset + gt_crop_h, h_gt)
        gt_crop_x1, gt_crop_x2 = gt_x_offset, min(gt_x_offset + gt_crop_w, w_gt)
        
        # Crop both images
        lr_cropped = lr_img[:, lr_crop_y1:lr_crop_y2, lr_crop_x1:lr_crop_x2]
        gt_cropped = gt_img[:, gt_crop_y1:gt_crop_y2, gt_crop_x1:gt_crop_x2]
        
        # Pad with zeros if needed (for LR)
        if self.is_pad_zeros and (lr_cropped.shape[1] < lr_crop_h or lr_cropped.shape[2] < lr_crop_w):
            lr_pad_h = max(0, lr_crop_h - lr_cropped.shape[1])
            lr_pad_w = max(0, lr_crop_w - lr_cropped.shape[2])
            lr_cropped = torch.nn.functional.pad(
                lr_cropped,
                (0, lr_pad_w, 0, lr_pad_h),
                mode='constant',
                value=0
            )
        
        # Pad with zeros if needed (for GT)
        if self.is_pad_zeros and (gt_cropped.shape[1] < gt_crop_h or gt_cropped.shape[2] < gt_crop_w):
            gt_pad_h = max(0, gt_crop_h - gt_cropped.shape[1])
            gt_pad_w = max(0, gt_crop_w - gt_cropped.shape[2])
            gt_cropped = torch.nn.functional.pad(
                gt_cropped,
                (0, gt_pad_w, 0, gt_pad_h),
                mode='constant',
                value=0
            )
        
        results['lr'] = lr_cropped
        results['gt'] = gt_cropped
        
        return results