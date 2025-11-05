from src.IJEPA.vit.vit import ViTPredictor, VisionTransformer
import json
import copy
import torch

file = open("parameters.json")
params_all= json.load(file)

parameters = params_all["ijepa"]

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

finetuned_student = VisionTransformer(
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

finetuned_teacher = copy.deepcopy(finetuned_student)

finetuned_predictor = ViTPredictor(
    finetuned_teacher.patch_embed.num_patches
)

finetuned_student.load_state_dict(torch.load("trained_student_jepa_20251104195320.pth", map_location="cpu"))
finetuned_predictor.load_state_dict(torch.load("trained_predictor_jepa_20251104195320.pth", map_location="cpu"))
finetuned_teacher.load_state_dict(torch.load("teacher_model_jepa_20251104195320.pth", map_location="cpu"))

trained_teacher = finetuned_teacher
trained_predictor = finetuned_predictor
trained_student = finetuned_student