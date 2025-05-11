import torch
import torchvision.transforms as T
from PIL import Image
from rhino.diffusers.distortions.linear_ops import MotionBlurOperator

def test_motion_blur(image_path, output_path, kernel_size=9, angle=45.0, intensity=0.5, device='cpu'):
    """Test MotionBlurOperator by applying motion blur to an image.
    
    Args:
        image_path (str): Path to input image.
        output_path (str): Path to save blurred image.
        kernel_size (int): Size of the motion blur kernel (odd number).
        angle (float): Angle of motion in degrees (0=horizontal, 90=vertical).
        intensity (float): Strength of the blur effect.
        device (str): Device to run computation ('cpu' or 'cuda').
    """
    try:
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Convert image to tensor (C, H, W)
        transform = T.ToTensor()
        image_tensor = transform(image).unsqueeze(0)  # Shape: (1, C, H, W)
        image_tensor = image_tensor.to(device)
        
        # Initialize MotionBlurOperator
        blur_op = MotionBlurOperator(
            kernel_size=kernel_size,
            angle=angle,
            intensity=intensity,
            device=device
        )
        
        # Apply motion blur
        blurred_tensor = blur_op.forward(image_tensor)
        
        # Convert back to PIL image
        blurred_tensor = blurred_tensor.squeeze(0).clamp(0, 1)  # Remove batch dim, clamp to [0,1]
        to_pil = T.ToPILImage()
        blurred_image = to_pil(blurred_tensor)
        
        # Save blurred image
        blurred_image.save(output_path)
        print(f"Blurred image saved to {output_path}")
        
    except FileNotFoundError:
        print(f"Error: Image file {image_path} not found.")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    # Example usage
    input_image = "data/ffhq/ffhq_imgs/ffhq_64/00000.png"  # Replace with your image path
    output_image = "ffhq_64_00000_blur.png"
    test_motion_blur(
        image_path=input_image,
        output_path=output_image,
        kernel_size=3,
        angle=90.0,
        intensity=1,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )