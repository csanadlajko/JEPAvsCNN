from src.JEPA.mask.masking import Mask
from src.JEPA.transform.datatransform import train_loader, test_loader
from src.JEPA.vit.vit import VisionTransformer
from src.JEPA.vit.vit import teacher_model, student_model
import torch.nn as nn
import torch


LEARNING_RATE = 0.003

loss = nn.MSELoss()
optim = torch.optim.Adam(params=teacher_model.parameters(), lr=LEARNING_RATE)
total_loss = 0
mask = Mask()

def train(teacher_mod, student_mod, loader, optimizer):
    teacher_mod.train()
    
    print("---STARTING TRAINING---")
    
    for x, _ in loader:
        ## MASKING WORKING FAULTY, NOT RETURNING CORRECT SHAPE
        batch, ctx_mask, target_mask = mask(x)
        
        optimizer.zero_grad()
        
                
        ctx_out = teacher_mod(ctx_mask[0])
        target_out = student_mod(target_mask[0])
        
        print(ctx_out)
        loss_curr = loss(ctx_out, target_out)
        loss_curr.backward()
        
        total_loss += loss_curr.item() * x.size(0)
        
    print("---TRAINING ENDED---")
    return total_loss / len(loader.dataset)

if __name__ == "__main__":
    
    train(teacher_model, student_model, train_loader, optim)
    