import torch.nn as nn
import torch
import copy
import json
from typing import Any
from src.IJEPA.mask.masking import apply_mask
from transformers import AutoTokenizer, AutoModelForCausalLM

file = open("././parameters.json")

param_file = json.load(file)

parameters: dict[str, Any] = param_file["ijepa"]
parameters_llm: dict[str, Any] = param_file["llmjepa"]

text_encoder = AutoModelForCausalLM.from_pretrained(parameters_llm["MODEL_NAME"])
tokenizer = AutoTokenizer.from_pretrained(parameters_llm["MODEL_NAME"])

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

## hyperparameters

DEPTH = parameters["DEPTH"]
DROP_RATE = parameters["DROP_RATE"]
BATCH_SIZE = parameters["BATCH_SIZE"]
CHANNELS = parameters["CHANNELS"]
EMBED_DIM = parameters["EMBED_DIM"]
IMG_SIZE = parameters["IMAGE_SIZE"]
PATCH_SIZE = parameters["PATCH_SIZE"]
MLP_DIM = parameters["MLP_DIM"]
NUM_HEADS = parameters["NUM_HEADS"]
EPOCHS = parameters["EPOCHS"]
NUM_CLASSES = parameters["NUM_CLASSES"]

class ViTPredictor(nn.Module):

    def __init__(self, num_patches, embed_dim=EMBED_DIM, pred_dim=None, depth=DEPTH, num_heads=NUM_HEADS, drop_rate=0.1, init_std=0.02):
        super().__init__()
        if pred_dim is None:
            pred_dim = embed_dim
        self.predictor_embed = nn.Linear(embed_dim, pred_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, pred_dim)) # learnable parameters to predict masked region

        self.pred_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, pred_dim), requires_grad=False)

        self.pred_blocks = nn.Sequential(*[
            TransformerEncoder(
                num_heads=num_heads,
                embed_dim=pred_dim,
                mlp_dim=MLP_DIM,
                drop=drop_rate
            )
            for _ in range(depth)
        ])

        self.label_to_embed = nn.Sequential(
            nn.Linear(768, pred_dim), ## 768 -> embed size of gpt2 model
            nn.LayerNorm(pred_dim),
            nn.GELU(),
            nn.Linear(pred_dim, pred_dim)
        )

        self.post_pred_mhsa = nn.MultiheadAttention(pred_dim, num_heads, drop_rate, batch_first=True)

        self.init_std = init_std
        self.predictor_norm = nn.LayerNorm(pred_dim)
        self.predictor_proj = nn.Linear(pred_dim, embed_dim) # back to encoder dimension

    def forward(self, x, context_mask, target_mask, labels: torch.Tensor, multimodal: bool):

        B = x.size(0)
        
        x = self.predictor_embed(x)
        
        target_positions = apply_mask(self.pred_pos_embed.repeat(B, 1, 1), target_mask)

        num_target_tokens = target_positions.size(1)
        mask_tokens = self.mask_token.repeat(target_positions.size(0), num_target_tokens, 1)
        mask_tokens = mask_tokens + target_positions
        
        context_tokens_repeated = x.repeat(target_positions.size(0) // x.size(0), 1, 1)
        
        x = torch.cat([context_tokens_repeated, mask_tokens], dim=1)
        
        for block in self.pred_blocks:
            x = block(x)
        
        x = self.predictor_norm(x)
        
        context_length = context_tokens_repeated.size(1)
        predicted_tokens = x[:, context_length:]
        
        predicted_tokens = self.predictor_proj(predicted_tokens)

        # only enter if model is ran in multimodal mode
        if multimodal: 
            
            label_list = [f"a photo of class: {label}" for label in labels]

            label_tokens = tokenizer(label_list, return_tensors='pt', padding=True)
            
            enc_labels = text_encoder(**label_tokens, output_hidden_states=True)

            enc_labels = enc_labels.hidden_states[-1]

            num_masks = predicted_tokens.size(0) // B

            enc_labels = enc_labels.unsqueeze(1).expand(-1, num_masks, -1, -1)
            enc_labels = enc_labels.reshape(-1, enc_labels.size(2), enc_labels.size(3))

            enc_labels = self.label_to_embed(enc_labels)

            pred_attended, _ = self.post_pred_mhsa(predicted_tokens, enc_labels, enc_labels)

            predicted_tokens = predicted_tokens + pred_attended
        
        return predicted_tokens

class MLP(nn.Module):
    
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.1):
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
    
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=EMBED_DIM):
        super().__init__()
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.in_chans = in_chans
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        
    def forward(self, x):
        B, C, H, W = x.shape # -> should be B N D
        x = self.proj(x).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        return x
    
class TransformerEncoder(nn.Module):
    
    def __init__(self, num_heads, embed_dim, mlp_dim, drop=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.att = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=drop, batch_first=True)
        self.mlp = MLP(in_features=embed_dim, hidden_features=mlp_dim, out_features=embed_dim, act_layer=nn.GELU, drop=drop)
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        attn_output, _ = self.att(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + self.dropout(attn_output)
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x
    
class VisionTransformer(nn.Module):
    
    def __init__(self, img_size, patch_size, in_chans, embed_dim, num_heads, depth, mlp_dim, drop_rate, num_classes=None):
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
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self.cls_fc1 = nn.Linear(embed_dim, embed_dim // 2)
        self.cls_fc2 = nn.Linear(embed_dim //2, NUM_CLASSES)

        nn.init.xavier_uniform_(self.cls_fc1.weight)
        nn.init.zeros_(self.cls_fc1.bias)
        nn.init.xavier_uniform_(self.cls_fc2.weight)
        nn.init.zeros_(self.cls_fc2.bias)
        
        self.cls_head = nn.Sequential(
            self.cls_fc1,
            nn.GELU(),
            nn.Dropout(drop_rate),
            self.cls_fc2
        )
        
    def forward(self, x, masks=None, return_cls_only=False, return_logits=False):
        x = self.patch_embed(x) # patch embed and pos encoding
        
        if masks is not None and not return_cls_only:
            x = apply_mask(x, masks) # only needed when entering with student model

        for block in self.encoder:
            x = block(x)

        x = self.norm(x)
        
        if return_cls_only:
            cls_token = x[:, 0, :]
            if return_logits:
                return self.cls_head(cls_token)
            return cls_token
        
        return x
    

teacher_model = VisionTransformer(
    img_size=IMG_SIZE,
    patch_size=PATCH_SIZE,
    in_chans=CHANNELS,
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    depth=DEPTH,
    mlp_dim=MLP_DIM,
    drop_rate=DROP_RATE,
    num_classes=10
)

student_model = copy.deepcopy(teacher_model)