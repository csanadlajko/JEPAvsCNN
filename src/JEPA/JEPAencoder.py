import torch

class PatchEmbed:
    
    """
    Patchify and embed representations from a given tensor.\n
    Tensor format must be: (B, C, H, W), where:\n
        - B is the number (batch) of images\n
        - C is the color spectrum of the image\n
        - H is height of the image\n
        - W is the width of the image
    """
    
    def __init__(self, image_data: torch.Tensor, patch_size: int = 8, emb_dim: int = 2):
        self.image_data = image_data
        self.patch_size = patch_size
        self.emb_dim = emb_dim
        
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
        
        return self._patch_embed(patches)
    
    def _patch_embed(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Embed patches into ``emb_dim`` dimension declared in the ``PatchEmbed`` instance.
        """
        linear = torch.nn.Linear(self.patch_size, self.emb_dim)
        return linear(patches)
    
class EncoderBlock():
    
    """
    Encodes a latent patch tensor into a token matrix.
    """
    
    def __init__(self, patch: torch.Tensor, d_model: int = 768, d_k:int = 32):
        self._patch = patch
        self.d_model = d_model
        self.d_k = d_k
        self.patch_vectors = self._patch.unsqueeze(0)
        self._W_Q = torch.nn.Linear(d_model, d_k, bias=False)
        self._W_K = torch.nn.Linear(d_model, d_k, bias=False)
        self._W_V = torch.nn.Linear(d_model, d_k, bias=False)
        self.Q = self._W_Q(self.patch_vectors)
        self.K = self._W_K(self.patch_vectors)
        self.V = self._W_V(self.patch_vectors)
        
    def _get_attention_matrix(self) -> torch.Tensor:
        scores = (self.Q @ self.K.T) / (self.d_k ** 0.5)
        return torch.softmax(scores, dim=-1)
    
    def encode(self) -> torch.Tensor:
        A: torch.Tensor = self._get_attention_matrix()
        Z: torch.Tensor = A @ self.v 
        return Z