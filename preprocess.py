import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from diffusers import AutoencoderKL
from tqdm import tqdm

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


def center_crop_resize(img, size):
    return transforms.functional.center_crop(
        transforms.functional.resize(img, size, transforms.InterpolationMode.BICUBIC),
        size
    )

def main():
    batch_size = 128  
    image_size = 128  # Original image size
    data_path = 'afhq/train'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_dir = 'preprocessed_data'
    os.makedirs(output_dir, exist_ok=True)

    # Set up transforms using standard torchvision transforms
    transform = transforms.Compose([
        transforms.Lambda(lambda x: center_crop_arr(x, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = ImageFolder(data_path, transform=transform)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  
        pin_memory=True
    )

    vae = AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-ema').to(device)

    total_samples = len(dataset)
    latent_size = image_size // 8  # VAE reduces spatial dimensions by factor of 8
    features_shape = (total_samples, 4, latent_size, latent_size)
    
    print(f"Allocating array for {total_samples} images...")
    all_features = np.zeros(features_shape, dtype=np.float32)
    
    current_idx = 0
    
    try:
        with torch.no_grad():
            for batch_idx, (images, _) in enumerate(tqdm(dataloader, desc="Processing batches")):
                # Move images to device and encode
                images = images.to(device)
                latents = vae.encode(images).latent_dist.sample().mul_(0.18215)
                
                # Move to CPU and store
                batch_features = latents.detach().cpu().numpy()
                batch_size = batch_features.shape[0]
                all_features[current_idx:current_idx + batch_size] = batch_features
                current_idx += batch_size

        # Save features and metadata
        print("\nSaving processed features...")
        output_file = os.path.join(output_dir, 'all_features.npy')
        np.save(output_file, all_features)
        
        metadata = {
            'shape': features_shape,
            'dtype': str(all_features.dtype),
            'total_samples': total_samples,
            'image_size': image_size,
            'latent_size': latent_size
        }
        np.save(os.path.join(output_dir, 'features_metadata.npy'), metadata)
        
        print(f"Successfully saved features to {output_file}")
        print(f"Feature shape: {features_shape}")
        print(f"Memory usage: {all_features.nbytes / 1e9:.2f} GB")

    except Exception as e:
        print(f"Error during processing: {str(e)}")
        raise e

if __name__ == '__main__':
    main()