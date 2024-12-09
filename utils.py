import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def view_tensor_images(tensor, nrow=8, title=None, normalize=True):
    """
    Display images from a BxCxHxW tensor.
    
    Args:
        tensor (torch.Tensor): Image tensor in BxCxHxW format
        nrow (int): Number of images per row in grid
        title (str): Optional title for the plot
        normalize (bool): Whether to normalize values to [0,1]
    """
    # Convert to numpy and move channels to end
    if isinstance(tensor, torch.Tensor):
        images = tensor.detach().cpu().numpy()
    else:
        images = np.array(tensor)
    
    # Handle single image case
    if len(images.shape) == 3:
        images = images[None, ...]
    
    # Transpose from BCHW to BHWC
    images = images.transpose(0, 2, 3, 1)
    
    if normalize:
        images = (images - images.min()) / (images.max() - images.min())
    
    # Create grid
    batch_size = images.shape[0]
    ncols = min(nrow, batch_size)
    nrows = (batch_size + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(2*ncols, 2*nrows))
    axes = axes.flatten() if batch_size > 1 else [axes]
    
    for ax, img in zip(axes, images):
        if img.shape[-1] == 1:  # Grayscale
            ax.imshow(img.squeeze(), cmap='gray')
        else:  # RGB
            ax.imshow(img)
        ax.axis('off')
    
    if title:
        plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def visualize_noise_levels(image, sigmas, seed=42):
    """
    Visualize an image with different noise levels.
    
    Args:
        image (torch.Tensor): Input image tensor in CxHxW format
        sigmas (list): List of noise standard deviations
        seed (int): Random seed for reproducibility
    """
    torch.manual_seed(seed)
    
    # Ensure image is tensor
    if not isinstance(image, torch.Tensor):
        image = torch.tensor(image)
    
    # Add batch dimension if needed
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    
    # Generate noisy versions
    noisy_images = [image.clone()]
    for sigma in sigmas:
        noise = torch.randn_like(image) * sigma
        noisy_images.append(image + noise)
    
    # Create grid
    total_images = len(sigmas) + 1
    ncols = min(4, total_images)
    nrows = (total_images + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(3*ncols, 3*nrows))
    axes = axes.flatten() if total_images > 1 else [axes]
    
    # Plot images
    titles = ['Original'] + [f'σ = {sigma:.2f}' for sigma in sigmas]
    for ax, img, title in zip(axes, noisy_images, titles):
        img_np = img.squeeze().permute(1, 2, 0).clamp(0, 1).numpy()
        ax.imshow(img_np)
        ax.set_title(title)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def center_crop_arr(pil_image, image_size):
    """
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])