from src.IJEPA.mask.masking import Mask
from src.IJEPA.transform.datatransform import train_loader, test_loader
from src.IJEPA.vit.vit import ViTPredictor
from src.IJEPA.vit.vit import teacher_model, student_model
from src.IJEPA.mask.masking import apply_mask
import torch.nn as nn
import torch
import torch.nn.functional as F
import json
import faiss

import transformers.models.ijepa

cls_loss = nn.CrossEntropyLoss()

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")

file = open("parameters.json")
parameters: dict[str, int] = json.load(file)["ijepa"]

loss = nn.MSELoss()
optim = torch.optim.Adam(params=student_model.parameters(), lr=parameters["LEARNING_RATE"], weight_decay=1e-4)
mask = Mask()
predictor = ViTPredictor(
    teacher_model.patch_embed.num_patches
)

optim_cls = torch.optim.Adam(
    params=student_model.parameters(),
    lr=parameters["LEARNING_RATE"] * 0.1,
    weight_decay=1e-4
)

optim_student = torch.optim.Adam(
    params=student_model.parameters(), 
    lr=parameters["LEARNING_RATE"], 
    weight_decay=1e-4
)

optim_predictor= torch.optim.Adam(
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

TRAIN_EMBEDDINGS = []
TRAIN_LABELS = []

K_CLOSEST = 4

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
        
        predicted_embeddings = predicted_target_tokens.mean(dim=1)
        
        TRAIN_EMBEDDINGS.append(predicted_embeddings.cpu())
        TRAIN_LABELS.append(labels.cpu())
        
        optimizer.step()
        optim_predictor.step()
        _ema_update(teacher_mod, student_mod)
        total_loss += loss_curr.item() 
        num_batches += 1
        
        print(f"loss at batch {batch_idx}: {loss_curr.item():.4f}")

        if batch_idx == 500:
            break
        
    print("---TRAINING ENDED---")
    if TRAIN_EMBEDDINGS:
        train_embs = torch.cat(TRAIN_EMBEDDINGS)
        train_labels = torch.cat(TRAIN_LABELS)
    else:
        train_embs = torch.empty(0)
        train_labels = torch.empty(0)
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    print(f"Average training loss: {avg_loss:.4f}")
    return train_embs, train_labels

def train_cls(student_model, train_dataset):
    student_model.train()
    
    print("---STARTING CLS TRAINING---")
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (images, labels) in enumerate(train_dataset):
        images = images.to(device)
        labels = labels.to(device)
        
        pred_classes = student_model(images, return_cls_only=True, return_logits=True)

        optim_cls.zero_grad()
        optim_predictor.zero_grad()

        loss = cls_loss(pred_classes, labels)

        loss.backward()
        
        optim_cls.step()
        optim_predictor.step()

        total_loss += loss.item()
        num_batches += 1

        if batch_idx % 50 == 0:
            print(f"CLS Loss at batch {batch_idx}: {loss.item():.4f}")

        if batch_idx == 500:
            break
    
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
            
            pred_classes = model(images, return_cls_only=True, return_logits=True)
            _, predicted = torch.max(pred_classes, 1)
            
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            
            if batch_idx % 50 == 0:
                current_acc = total_correct / total_samples
                print(f"Batch {batch_idx}, Current accuracy: {current_acc:.4f}")
            
            if batch_idx == 300:
                break
    
    final_accuracy = total_correct / total_samples
    print(f"---CLS EVALUATION ENDED---")
    print(f"Final CLS accuracy: {final_accuracy:.4f}")
    return final_accuracy



def eval_ijepa(model, test_dataset, train_embs, train_labels):
    """
    Core logic:
        1. Encode the given test image with the frozen student model.
        2. Given the result embedding, search the k most relevant results.
        3. Calculate the accuracy by comparing the result embedding's label to the current label.
        4. Get the final results by stacking the correct / total guesses.
    """

    model.eval()
    total_acc = 0
    n_test = 0

    print("---STARTING MODEL EVALUATION---")

    with torch.no_grad():
        train_np = train_embs.detach().cpu().numpy().astype('float32')
        faiss.normalize_L2(train_np)
        index = faiss.IndexFlatIP(train_np.shape[1])
        index.add(train_np)

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_dataset):
            images = images.to(device)
            labels = labels.to(device)
            
            for i in range(images.shape[0]):
                image = images[i:i+1] 
                label = labels[i:i+1]
                                
                ctx_masks, target_masks = mask(image)
                
                ctx_embedding = model(image, ctx_masks)
                
                predicted_target_embedding = predictor(ctx_embedding, ctx_masks, target_masks)
                predicted_target_embedding = predicted_target_embedding.mean(dim=1)
                
                query_emb = predicted_target_embedding.detach().cpu().numpy().astype('float32')
                faiss.normalize_L2(query_emb)
                
                distances, indices = index.search(query_emb, K_CLOSEST)
                neighbor_labels = train_labels[indices[0]] 
                
                matches = (neighbor_labels == label.cpu()).sum().item()
                acc = matches / K_CLOSEST
                
                total_acc += acc
                n_test += 1
                
                print(f"Current accuracy: {acc:.4f}")

            if batch_idx == 300:
                break
    print("---EVALUATING ENDED---")
    final_acc = total_acc / n_test
    print(f"Final accuracy: {final_acc:.4f}")
    return final_acc
        

if __name__ == "__main__":
    print(f"Training for {parameters['EPOCHS']} epochs...")
    
    for epoch in range(parameters['EPOCHS']):
        print(f"\n=== EPOCH {epoch + 1}/{parameters['EPOCHS']} ===")
        train_embs, train_labels = train(teacher_model, student_model, train_loader, optim_student)
        
        TRAIN_EMBEDDINGS.clear()
        TRAIN_LABELS.clear()
    
    train_cls(student_model, train_loader)
    
    print("\n=== FINAL EVALUATION ===")
    
    cls_acc = eval_cls(student_model, test_loader)
    print(f"-- CLS token classification accuracy: {cls_acc:.4f}")
    
    # Evaluate I-JEPA retrieval
    # ijepa_acc = eval_ijepa(student_model, test_loader, train_embs, train_labels)
    # print(f"-- I-JEPA retrieval accuracy: {ijepa_acc:.4f}")