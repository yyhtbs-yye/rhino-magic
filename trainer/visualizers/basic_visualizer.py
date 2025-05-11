import torch

def visualize_image_dict(
    logger,
    images_dict,
    keys,
    global_step,
    wnb=(0, 1),
    prefix="val",
    max_images=4,
    texts=None,
):
    """Create and log comparison visualizations between different image types.
    
    Args:
        logger: The lightning logger object with an add_image method
        images_dict: Dictionary with image tensors of shape [B, C, H, W]
        keys: List of lists, where each inner list contains keys to compare side by side
        global_step: Current global step for logging
        wnb: Tuple for normalization: (multiplier, offset). Default: (0, 1)
        prefix: Prefix for the logged image names
        max_images: Maximum number of images to include from each batch
        texts: Optional list of text descriptions to include with images
    """
    for comp_idx, key in enumerate(keys):
        # Skip if any key is missing
        if key not in images_dict:
            print(f"Warning: Keys {key} not found in images_dict. Skipping comparison.")
            continue
        
        # Create a name for this comparison
        comparison_name = f"{prefix}/visualization-{key}"
        
        # Normalize images if needed
        images_to_compare = []
        img = images_dict[key]
        img = (img.clamp(-1, 1) * wnb[0]) + wnb[1]
        images_to_compare.append(img)
        
        # Ensure all images have the same batch size and dimensions
        batch_size = min(max_images, min(img.shape[0] for img in images_to_compare))
        images_to_compare = [img[:batch_size] for img in images_to_compare]
        
        # Create comparison grid - first horizontally concatenate pairs in each batch item
        comparison_rows = []
        descriptions = []
        
        for b in range(batch_size):
            # Concatenate images horizontally for this batch item
            row_images = [img[b] for img in images_to_compare]
            
            # Store text as description instead of drawing on image
            if texts is not None and b < len(texts):
                descriptions.append(texts[b])
            
            row = torch.cat(row_images, dim=2)  # Concatenate along width (dim=2)
            comparison_rows.append(row)
        
        # Then vertically concatenate all batch items
        comparison = torch.cat(comparison_rows, dim=1)  # Concatenate along height (dim=1)
        
        # Prepare description text if available
        description = None
        if texts is not None:
            description = " | ".join([f"Image {i+1}: {desc}" for i, desc in enumerate(descriptions) if i < batch_size])
        
        # Log to the logger with description
        logger.log_image(
            comparison_name,
            comparison,
            global_step,
            dataformats='CHW'
        )