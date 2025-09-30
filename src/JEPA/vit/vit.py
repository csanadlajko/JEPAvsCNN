import torch.nn as nn
import torch

class MLP(nn.Module):
    
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class PatchEmbed(nn.Module):
    
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.in_chans = in_chans
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        B, C, H, W = x.shape # -> should be B N D
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
    
class ViTInput(nn.Module):
    
    def __init__(self, patch_embed):
        super().__init__()
        self.patch_embed = patch_embed
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.patch_embed.embed_dim))    
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.patch_embed.embed_dim))
        
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        x = x + self.pos_embed
        return x
    
class InputTest:
    
    def __init__(self):
        patch_embed = PatchEmbed(
            img_size=4,
            patch_size=2,
            in_chans=1,
            embed_dim=3
        )
        self.vit_input = ViTInput(patch_embed)
        self.img = torch.arange(1, 17).float().reshape(1, 1, 4, 4) # numbers 1-16 in correct shape
        
    def __test__(self):
        print("-----STARTING VIT TRANSFORM-----")
        out = self.vit_input(self.img)
        print("-----VIT TRANSFORM FINISHED-----")
        
        print(out)
        
t2 = InputTest()
t2.__test__()