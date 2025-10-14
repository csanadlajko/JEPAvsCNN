from src.IJEPA.mask.masking import Mask
from src.IJEPA.transform.datatransform import train_loader
from src.IJEPA.vit.vit import PredictionHead
from src.IJEPA.vit.vit import teacher_model, student_model
import torch.nn as nn
import torch
import json

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")

file = open("parameters.json")
parameters: dict[str, int] = json.load(file)["ijepa"]

loss = nn.MSELoss()
optim = torch.optim.Adam(params=student_model.parameters(), lr=parameters["LEARNING_RATE"])
mask = Mask()
pred_head = PredictionHead(student_dim=parameters["EMBED_DIM"], teacher_dim=parameters["EMBED_DIM"])

teacher_model.to(device)
student_model.to(device)
pred_head.to(device)

@torch.no_grad()
def _ema_update(teacher_mod, student_mod, momentum=parameters["MOMENTUM"]):
    for t_param, s_param in zip(teacher_mod.parameters(), student_mod.parameters()):
        t_param.data.mul_(momentum).add_(s_param.data, alpha=1.0 - momentum)

def train(teacher_mod, student_mod, loader, optimizer):
    teacher_mod.eval()
    student_mod.train()
    
    print("---STARTING TRAINING---")
    total_loss = 0.0
    
    for x, _ in loader:
        x = x.to(device)
        
        with torch.no_grad():
            teacher_tokens = teacher_mod.patch_embed(x)
        batch_tokens, ctx_tokens, target_tokens = mask(teacher_tokens)

        optimizer.zero_grad()

        with torch.no_grad():
            teacher_target = teacher_mod(target_tokens)
        student_ctx = student_mod(ctx_tokens)
        
        predicted = pred_head(student_ctx)

        loss_curr = loss(teacher_target, predicted)
        loss_curr.backward()
        optimizer.step()
        _ema_update(teacher_mod, student_mod)
        total_loss = total_loss + loss_curr.item() * x.size(0)
        
        print(f"Iteration ended with loss: {loss_curr:.4f}")
        
    print("---TRAINING ENDED---")
    return total_loss / len(loader.dataset)

if __name__ == "__main__":
    train(teacher_model, student_model, train_loader, optim)
    