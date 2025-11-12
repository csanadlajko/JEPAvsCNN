import torch
import matplotlib.pyplot as plt
import numpy as np
from src.IJEPA.mask.masking import Mask, apply_mask
from src.IJEPA.transform.datatransform import train_loader
import json

file = open("parameters.json")
parameters = json.load(file)["ijepa"]

def visualize_masks(images, context_masks, target_masks, num_samples=4):
    batch_size = min(images.shape[0], num_samples)
    patch_size = parameters["PATCH_SIZE"]
    img_size = parameters["IMAGE_SIZE"]
    num_patches_per_side = img_size // patch_size
    
    fig, axes = plt.subplots(batch_size, 3, figsize=(15, 5 * batch_size))
    if batch_size == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(batch_size):
        img = images[i].cpu()
        img = img.permute(1, 2, 0).numpy()
        img = np.clip((img + 1) / 2, 0, 1) 
        
        context_mask_2d = torch.zeros(num_patches_per_side, num_patches_per_side)
        target_mask_2d = torch.zeros(num_patches_per_side, num_patches_per_side)
        
        if isinstance(context_masks[i], torch.Tensor) and context_masks[i].numel() > 0:
            ctx_indices = context_masks[i].cpu()
            for idx in ctx_indices:
                row = idx // num_patches_per_side
                col = idx % num_patches_per_side
                if 0 <= row < num_patches_per_side and 0 <= col < num_patches_per_side:
                    context_mask_2d[row, col] = 1
        
        if isinstance(target_masks[i], list):
            for block_idx in target_masks[i]:
                if isinstance(block_idx, torch.Tensor) and block_idx.numel() > 0:
                    for idx in block_idx.cpu():
                        row = idx // num_patches_per_side
                        col = idx % num_patches_per_side
                        if 0 <= row < num_patches_per_side and 0 <= col < num_patches_per_side:
                            target_mask_2d[row, col] = 1
        
        context_mask_2d_np = context_mask_2d.numpy()
        target_mask_2d_np = target_mask_2d.numpy()
        
        context_mask_img_2d = np.kron(context_mask_2d_np, np.ones((patch_size, patch_size)))
        target_mask_img_2d = np.kron(target_mask_2d_np, np.ones((patch_size, patch_size)))
        
        if context_mask_img_2d.shape[0] != img_size or context_mask_img_2d.shape[1] != img_size:
            h_ratio = img_size / context_mask_img_2d.shape[0]
            w_ratio = img_size / context_mask_img_2d.shape[1]
            context_mask_img_2d = np.repeat(np.repeat(context_mask_img_2d, int(h_ratio), axis=0), int(w_ratio), axis=1)
            target_mask_img_2d = np.repeat(np.repeat(target_mask_img_2d, int(h_ratio), axis=0), int(w_ratio), axis=1)
            context_mask_img_2d = context_mask_img_2d[:img_size, :img_size]
            target_mask_img_2d = target_mask_img_2d[:img_size, :img_size]
        
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f'Original Image {i+1}')
        axes[i, 0].axis('off')
        
        context_mask_bool = context_mask_img_2d == 1
        context_overlay = np.zeros_like(img)
        context_overlay[:, :, 1] = context_mask_bool.astype(float)
        
        axes[i, 1].imshow(img)
        axes[i, 1].imshow(context_overlay, alpha=0.5)
        axes[i, 1].set_title(f'Context Mask (green) - Image {i+1}')
        axes[i, 1].axis('off')
        
        target_mask_bool = target_mask_img_2d == 1
        target_overlay = np.zeros_like(img)
        target_overlay[:, :, 0] = target_mask_bool.astype(float)
        
        axes[i, 2].imshow(img)
        axes[i, 2].imshow(target_overlay, alpha=0.5)
        axes[i, 2].set_title(f'Target Masks (red) - Image {i+1}')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    mask = Mask()
    
    for images, labels in train_loader:
        images = images.cuda() if torch.cuda.is_available() else images
        
        context_masks, target_masks = mask(images)
        
        print(f"Batch shape: {images.shape}")
        print(f"Number of context masks: {len(context_masks)}")
        print(f"Number of target mask sets: {len(target_masks)}")
        print(f"Target blocks per image: {[len(tm) if isinstance(tm, list) else 1 for tm in target_masks]}")
        
        for i, tm in enumerate(target_masks):
            if isinstance(tm, list):
                total_tokens = sum(block.numel() if isinstance(block, torch.Tensor) else 0 for block in tm)
                print(f"Image {i}: {len(tm)} target blocks, {total_tokens} total target tokens")
        
        fig = visualize_masks(images.cpu(), context_masks, target_masks)
        plt.savefig('mask_visualization.png', dpi=150, bbox_inches='tight')
        print("Saved visualization to mask_visualization.png")
        plt.show()
        
        break

