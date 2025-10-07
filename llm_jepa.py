import json
from src.LLMJEPA.encoder.encoder import TextEncoder
import torch
from src.LLMJEPA.dataprep.dataprep import get_dummy_data
from src.LLMJEPA.encoder.encoder import init_tokenizer
from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset

param_file = open("parameters.json")
parameters = json.load(param_file)["llmjepa"]

dummy_json = get_dummy_data()

EPOCHS = parameters["EPOCHS"]
LR = parameters["LEARNING_RATE"]
BATCH_SIZE = parameters["BATCH_SIZE"]
tokenizer = AutoTokenizer.from_pretrained(parameters["MODEL_NAME"])
new_tokens = [f"<|predictor_{i+1}|>" for i in range(parameters["PRED_TOKENS"])]
new_tokens += ["<|perception|>", "<|eot_id|>"]
tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})

model = AutoModelForCausalLM.from_pretrained(parameters["MODEL_NAME"])

model.resize_token_embeddings(len(tokenizer))

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

device = "cuda" if torch.cuda.is_available() else "cpu"

# optimizer = torch.optim.Adam(teach_model.parameters(), LR) --> own training process might be implemented (instead of transformers.Trainer)

class DummyDataset(Dataset):
    """
    Dummy class for Dataset obj parsing (for training).
    """
    def __init__(self, inputs):
        self.inputs = inputs
        
    def __len__(self):
        return self.inputs["input_ids"].size(0)
    
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.inputs.items()}

def train_model(dataset):
    
    inputs = init_tokenizer(tokenizer, dataset)
    
    input_data = DummyDataset(inputs)

    training_args = TrainingArguments(
        output_dir="./outputs",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LR,
        num_train_epochs=EPOCHS,
        fp16=False,
        bf16=True if device == "cuda" else False,
        gradient_checkpointing=True,
        dataloader_drop_last=True,
        dataloader_num_workers=0,
        eval_strategy="no",
        gradient_accumulation_steps=1,
        push_to_hub=False,
        remove_unused_columns=False, ## VERY IMPORTANT !! in order to use asssistant and user data in forward method! otherwise columns will be dropped
    )

    trainer = TextEncoder(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        gamma=parameters["LM_GAMMA"],
        jp_lambda=parameters["JEPA_LAMBDA"],
        train_dataset=input_data,
        eval_dataset=input_data,
    )
    
    print(f"--- STARTING TRAINING WITH EPOCHS: {EPOCHS} ---")
    trainer.train()
    print("--- TRAINING ENDED SUCCESSFULLY ---")
    
if __name__ == "__main__":
    data_json = get_dummy_data()
    train_model(data_json)