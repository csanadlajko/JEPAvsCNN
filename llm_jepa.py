import json
from src.LLMJEPA.encoder.encoder import TextEncoder
import torch
from src.LLMJEPA.dataprep.dataprep import get_dummy_data
from transformers import TrainingArguments

param_file = open("parameters.json")
parameters = json.load(param_file)["llmjepa"]

dummy_json = get_dummy_data()

EPOCHS = parameters["EPOCHS"]
LR = parameters["LEARNING_RATE"]
BATCH_SIZE = parameters["BATCH_SIZE"]

# optimizer = torch.optim.Adam(teach_model.parameters(), LR) --> own training process might be implemented (instead of transformers.Trainer)

training_args = TrainingArguments(
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LR,
    num_train_epochs=EPOCHS,
    fp16=False,
    bf16=True,
    gradient_checkpointing=True,
    dataloader_drop_last=True,
    dataloader_num_workers=0,
    eval_strategy="no",
    gradient_accumulation_steps=1
)

trainer = TextEncoder(
    model=parameters["MODEL_NAME"],
    args=training_args,
    pred_tokens=parameters["PRED_TOKENS"],
    gamma=parameters["LM_GAMMA"],
    jp_lambda=["JEPA_LAMBDA"],
)

def train():
    print(f"---TRAINING STARTED WITH EPOCHS: {EPOCHS}")
    trainer.train()
    print("---TRAINING ENDED SUCCESSFULLY---")


if __name__ == "__main__":
    train()