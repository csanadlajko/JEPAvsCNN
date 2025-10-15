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
optim = torch.optim.Adam(params=student_model.parameters(), lr=parameters["LEARNING_RATE"])
mask = Mask()
pred_head = PredictionHead(student_dim=parameters["EMBED_DIM"], teacher_dim=parameters["EMBED_DIM"])

teacher_model.to(device)
student_model.to(device)
pred_head.to(device)

TRAIN_EMBEDDINGS = []
TRAIN_LABELS = []

K_CLOSEST = 10

@torch.no_grad()
def _ema_update(teacher_mod, student_mod, momentum=parameters["MOMENTUM"]):
    for t_param, s_param in zip(teacher_mod.parameters(), student_mod.parameters()):
        t_param.data.mul_(momentum).add_(s_param.data, alpha=1.0 - momentum)

def train(teacher_mod, student_mod, loader, optimizer):
    teacher_mod.eval()
    student_mod.train()

    print("---STARTING TRAINING---")
    total_loss = 0.0
    
    for x, label in list(loader)[-200:]:
        x = x.to(device)

        with torch.no_grad():
            teacher_tokens = teacher_mod.patch_embed(x)
        batch_tokens, ctx_tokens, target_tokens = mask(teacher_tokens)

        optimizer.zero_grad()

        with torch.no_grad():
            teacher_target = teacher_mod(target_tokens)
        student_ctx = student_mod(ctx_tokens)
        
        predicted = pred_head(student_ctx)
        predicted = predicted.mean(dim=1)
        teacher_target = teacher_target.mean(dim=1)
        TRAIN_EMBEDDINGS.append(predicted.cpu())
        TRAIN_LABELS.append(label)

        loss_curr = loss(teacher_target, predicted)
        loss_curr.backward()
        optimizer.step()
        _ema_update(teacher_mod, student_mod)
        total_loss = total_loss + loss_curr.item() * x.size(0)
        
        print(f"Iteration ended with loss: {loss_curr:.4f}")
        
    print("---TRAINING ENDED---")
    train_embs = torch.cat(TRAIN_EMBEDDINGS)
    train_labels = torch.cat(TRAIN_LABELS)
    return train_embs, train_labels

def eval_ijepa(model, test_dataset, train_embs, train_labels):
    """
    Core logic:\n
        1, Encode the given test image with the frozen student model.\n
        2, Given the result embedding, search the k most relevant results.\n
        3, Calculate the accuracy by comparing the result embedding's label to the current label.\n
        4, Get the final results by stacking the correct / total guesses.\n
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

    with torch.no_grad(): ## freeze student model
        for image, label in test_dataset:
            patch_embed = model.patch_embed(image)

            _, ctx_masks, _ = mask(patch_embed)
            ctx_embedding = model(ctx_masks)
            predicted_target_embedding = pred_head(ctx_embedding)
            predicted_target_embedding = predicted_target_embedding.mean(dim=1)

            faiss.normalize_L2(predicted_target_embedding.numpy())

            distances, indices = index.search(predicted_target_embedding.numpy(), K_CLOSEST)
            neighbor_labels = train_labels[indices]

            matches = (neighbor_labels == label.unsqueeze(1)).sum(dim=1)
            acc = (matches / K_CLOSEST).float().mean().item()

            total_acc += acc
            n_test += 1
            print(f"current accuracy for {K_CLOSEST} neighbours: {acc}")

    print("---EVALUATING ENDED---")
    return total_acc / n_test
        

if __name__ == "__main__":
    train_embs, train_labels = train(teacher_model, student_model, train_loader, optim)
    final_acc = eval_ijepa(student_model, test_loader, train_embs, train_labels)

    print(f"-- final accuracy for ijepa model is: {final_acc}")