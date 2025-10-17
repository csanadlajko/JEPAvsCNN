from src.IJEPA.mask.masking import Mask
from src.IJEPA.transform.datatransform import train_loader, test_loader
from src.IJEPA.vit.vit import PredictionHead
from src.IJEPA.vit.vit import teacher_model, student_model
import torch.nn as nn
import torch
import json
import numpy as np
import faiss

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")

file = open("parameters.json")
parameters: dict[str, int] = json.load(file)["ijepa"]

loss = nn.MSELoss()
optim = torch.optim.Adam(params=student_model.parameters(), lr=parameters["LEARNING_RATE"], weight_decay=1e-4)
mask = Mask()
pred_head = PredictionHead(student_dim=parameters["EMBED_DIM"], teacher_dim=parameters["EMBED_DIM"])

teacher_model.to(device)
student_model.to(device)
pred_head.to(device)

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

    print("---STARTING TRAINING---")
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (x, label) in enumerate(loader):
        x = x.to(device)
        label = label.to(device)

        with torch.no_grad():
            teacher_tokens = teacher_mod.patch_embed(x)
        batch_tokens, ctx_tokens, target_tokens = mask(teacher_tokens)

        optimizer.zero_grad()

        with torch.no_grad():
            teacher_target = teacher_mod(target_tokens)
        
        student_ctx = student_mod(ctx_tokens)
        
        predicted = pred_head(student_ctx)
        loss_curr = loss(predicted, teacher_target)
        loss_curr.backward()
        predicted = predicted.mean(dim=1)
        teacher_target = teacher_target.mean(dim=1)
        TRAIN_EMBEDDINGS.append(predicted.cpu())
        TRAIN_LABELS.append(label.cpu())
        
        optimizer.step()
        _ema_update(teacher_mod, student_mod)
        total_loss = total_loss + loss_curr.item() * x.size(0)
        num_batches += 1
        
        print(f"Loss: {loss_curr:.4f}")
        
    print("---TRAINING ENDED---")
    train_embs = torch.cat(TRAIN_EMBEDDINGS)
    train_labels = torch.cat(TRAIN_LABELS)
    avg_loss = total_loss / num_batches
    print(f"Average training loss: {avg_loss:.4f}")
    return train_embs, train_labels

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
                
                patch_embed = model.patch_embed(image)
                
                _, ctx_masks, _ = mask(patch_embed)
                ctx_embedding = model(ctx_masks)
                predicted_target_embedding = pred_head(ctx_embedding)
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

    print("---EVALUATING ENDED---")
    final_acc = total_acc / n_test
    print(f"Final accuracy: {final_acc:.4f}")
    return final_acc
        

if __name__ == "__main__":
    print(f"Training for {parameters['EPOCHS']} epochs...")
    
    for epoch in range(parameters['EPOCHS']):
        print(f"\n=== EPOCH {epoch + 1}/{parameters['EPOCHS']} ===")
        train_embs, train_labels = train(teacher_model, student_model, train_loader, optim)
                
        TRAIN_EMBEDDINGS.clear()
        TRAIN_LABELS.clear()
    
    print("\n=== FINAL EVALUATION ===")
    final_acc = eval_ijepa(student_model, test_loader, train_embs, train_labels)
    print(f"-- Final accuracy for I-JEPA model: {final_acc:.4f}")