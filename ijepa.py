from src.IJEPA.mask.masking import Mask
from src.IJEPA.transform.datatransform import train_loader, test_loader
from src.IJEPA.vit.vit import ViTPredictor
from src.IJEPA.vit.vit import teacher_model, student_model
from src.IJEPA.mask.masking import apply_mask
import torch.nn as nn
import torch
import torch.nn.functional as F
import json

cls_loss = nn.CrossEntropyLoss()

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")

file = open("parameters.json")
parameters: dict[str, int] = json.load(file)["ijepa"]

loss = nn.MSELoss()
mask = Mask()
predictor = ViTPredictor(
    teacher_model.patch_embed.num_patches
)

optim_cls = torch.optim.AdamW(
    params=student_model.parameters(),
    lr=0.001,
    weight_decay=1e-4
)

lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim_cls, 
                                                    T_max=1, 
                                                    eta_min=0)

optim_student = torch.optim.AdamW(
    params=student_model.parameters(), 
    lr=parameters["LEARNING_RATE"],
    weight_decay=1e-4
)

student_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim_student, 10, 0)

optim_predictor= torch.optim.AdamW(
    params=predictor.parameters(), 
    lr=parameters["LEARNING_RATE"],
    weight_decay=1e-4
)


# xavier initialization for model weights
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        torch.nn.init.ones_(m.weight)
        torch.nn.init.zeros_(m.bias)

teacher_model.apply(init_weights)
student_model.apply(init_weights)
predictor.apply(init_weights)

teacher_model.to(device)
student_model.to(device)
predictor.to(device)

@torch.no_grad()
def _ema_update(teacher_mod, student_mod, momentum=parameters["MOMENTUM"]):
    for t_param, s_param in zip(teacher_mod.parameters(), student_mod.parameters()):
        t_param.data.mul_(momentum).add_(s_param.data, alpha=1.0 - momentum)

def train(teacher_mod, student_mod, loader, optimizer):
    teacher_mod.eval()
    student_mod.train()
    predictor.train()

    print("---STARTING TRAINING---")
    total_loss = 0.0
    num_batches = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)
        
        context_masks, target_masks = mask(images) # only indices
        
        with torch.no_grad():
            teacher_tokens = teacher_mod(images)
            teacher_tokens = F.layer_norm(teacher_tokens, (teacher_tokens.size(-1),))
            teacher_target_tokens = apply_mask(teacher_tokens, target_masks)
        
        student_tokens = student_mod(images, context_masks)

        predicted_target_tokens = predictor(student_tokens, context_masks, target_masks)
        
        optimizer.zero_grad()
        optim_predictor.zero_grad()
        
        loss_curr = loss(predicted_target_tokens, teacher_target_tokens)
            
        loss_curr.backward()
        
        optimizer.step()
        optim_predictor.step()
        
        _ema_update(teacher_mod, student_mod)
        
        total_loss += loss_curr.item() 
        num_batches += 1
        
        if batch_idx % 100 == 0:
            current_lr = optimizer.param_groups[0]['lr'] ## lr should update when entering another epoch (scheduler)
            print(f"loss at batch {batch_idx}: {loss_curr.item():.4f}, lr: {current_lr:.6f}")

        
    print("---TRAINING ENDED---")
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    print(f"Average training loss: {avg_loss:.4f}")

def train_cls(student_model, train_dataset):
    student_model.train()
    
    print("---STARTING CLS TRAINING---")
    total_loss = 0.0
    num_batches = 0
    correct_predictions = 0
    total_predictions = 0

    for name, param in student_model.named_parameters():
        if "cls_fc" not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
    
    for batch_idx, (images, labels) in enumerate(train_dataset):
        images = images.to(device)
        labels = labels.to(device)
        
        pred_classes = student_model(images, masks=None, return_cls_only=True, return_logits=True)

        optim_cls.zero_grad()

        loss = cls_loss(pred_classes, labels)
        loss.backward()
        optim_cls.step()

        _, predicted = torch.max(pred_classes, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_predictions += labels.size(0)

        total_loss += loss.item()
        num_batches += 1

        if batch_idx % 100 == 0:
            current_acc = correct_predictions / total_predictions
            print(f"CLS Loss at batch {batch_idx}: {loss.item():.4f}, Accuracy: {current_acc:.4f}")

    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    print(f"Average CLS training loss: {avg_loss:.4f}")
    print("---CLS TRAINING ENDED---")

def eval_cls(model, test_dataset):
    """
    Evaluate the model using CLS token classification
    """
    model.eval()
    total_correct = 0
    total_samples = 0
    
    print("---STARTING CLS EVALUATION---")
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_dataset):
            images = images.to(device)
            labels = labels.to(device)
            
            pred_classes = model(images, masks=None, return_cls_only=True, return_logits=True)
            _, predicted = torch.max(pred_classes, 1)
            
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            
            if batch_idx % 50 == 0:
                current_acc = total_correct / total_samples
                print(f"Batch {batch_idx}, Current accuracy: {current_acc:.4f}")
            
    
    final_accuracy = total_correct / total_samples
    print(f"---CLS EVALUATION ENDED---")
    print(f"Final CLS accuracy: {final_accuracy:.4f}")
    return final_accuracy

if __name__ == "__main__":
    print(f"Training for {parameters['EPOCHS']} epochs...")
    
    for epoch in range(parameters['EPOCHS']):
        print(f"\n=== EPOCH {epoch + 1}/{parameters['EPOCHS']} ===")
        train(teacher_model, student_model, train_loader, optim_student)
        student_scheduler.step()
    
    for epoch in range(parameters['EPOCHS']):
        train_cls(student_model, train_loader)
    
    print("\n=== FINAL EVALUATION ===")
    
    cls_acc = eval_cls(student_model, test_loader)
    print(f"-- CLS token classification accuracy: {cls_acc:.4f}")