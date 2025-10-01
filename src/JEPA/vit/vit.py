import torch.nn as nn
import torch
import copy

## hyperparameters

DEPTH = 6
DROP_RATE = 0.1
BATCH_SIZE = 32
CHANNELS = 3
EMBED_DIM = 256
IMG_SIZE = 128
PATCH_SIZE = 16
MLP_DIM = 512
NUM_HEADS = 8
EPOCHS = 10

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
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        
    def forward(self, x):
        B, C, H, W = x.shape # -> should be B N D
        x = self.proj(x).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        return x
    

class TransformerEncoder(nn.Module):
    
    def __init__(self, num_heads, embed_dim, mlp_dim, drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.att = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=drop, batch_first=True)
        self.mlp = MLP(embed_dim, embed_dim*4, mlp_dim, nn.GELU, drop)


    def forward(self, x):
        x = x + self.att(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
    
class VisionTransformer(nn.Module):
    
    def __init__(self, img_size, patch_size, in_chans, embed_dim, num_heads, depth, mlp_dim, drop_rate):
        super().__init__()
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )
        self.encoder = nn.Sequential(*[
            TransformerEncoder(
                num_heads=num_heads,
                embed_dim=embed_dim,
                mlp_dim=mlp_dim,
                drop=drop_rate
            )
            for _ in range(DEPTH)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        
        # optional -> head to predict classes (nn.Linear(embed_dim, num_classes))
        
    def forward(self, x):
        x = self.patch_embed(x)
        x = self.encoder(x)
        x = self.norm(x)
        cls_token = x[:, 0]
        return cls_token # return self.head(cls_token) when classification

teacher_model = VisionTransformer(
    img_size=IMG_SIZE,
    patch_size=PATCH_SIZE,
    in_chans=CHANNELS,
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    depth=DEPTH,
    mlp_dim=MLP_DIM,
    drop_rate=DROP_RATE
)

student_model = copy.deepcopy(teacher_model)