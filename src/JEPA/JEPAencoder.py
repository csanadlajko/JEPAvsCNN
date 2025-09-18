import torch

class Patchify:
    
    """
    Patchify images from a given tensor.\n
    Tensor format must be: (B, C, H, W), where:\n
        - B is the number (batch) of images\n
        - C is the color spectrum of the image\n
        - H is height of the image\n
        - W is the width of the image
    """
    
    def __init__(self, image_data: torch.Tensor, patch_size: int = 8):
        self.image_data = image_data
        self.patch_size = patch_size
        
    def patchify_image(self) -> torch.Tensor:
        """
        Takes a tensor (B, C, H, W) as an input, and returns a vector containing flattened patches.\n
        Result vector: (B, N, patch_dimension), where:
            - ``N`` is the number of patches
            - ``patch_dimension`` is the dimension of a single patch
        """
        B, C, _, _ = self.image_data.shape
        
        patches: torch.Tensor = self.image_data.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        
        # patches: (B, C, patch_per_row, patch_per_column, patch_size_per_row, patch_size_per_column)
        
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        
        # patches: (B, patch_per_row, patch_per_column, C, patch_size_per_row, patch_size_per_column) -> format for flattening
        
        patches = patches.view(B, -1, C *self.patch_size * self.patch_size)
        
        # patches: (B, number_of_patches, patch_dimension)
        
        return patches