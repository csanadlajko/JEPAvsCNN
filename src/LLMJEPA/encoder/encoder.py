from transformers import Trainer
import torch
import json
from typing import Any
from src.LLMJEPA.dataprep.dataprep import get_dummy_data

file = open("././parameters.json")
parameters: dict[str, Any] = json.load(file)["llmjepa"]
dummydata = get_dummy_data()

BATCH_SIZE = parameters["BATCH_SIZE"]
LR = parameters["LEARNING_RATE"]
EPOCHS = parameters["EPOCHS"]
MODEL_NAME = parameters["MODEL_NAME"]
NUM_PRED_TOKENS = parameters["PRED_TOKENS"]
LM_GAMMA = parameters["LM_GAMMA"]
JEPA_LAMBDA = parameters["JEPA_LAMBDA"]

device = "cuda" if torch.cuda.is_available() else "cpu"

class TextEncoder(Trainer):
    
    def __init__(self, *args, gamma=0.9, jp_lambda=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = gamma
        self.jp_lambda = jp_lambda
    
    def forward(self, encoder, inputs):
        
        with torch.set_grad_enabled(True):
            output = encoder(**inputs, output_hidden_states=True)
        
        batch_size = inputs["input_ids"].shape[0] // 3 # as we gave full, user, assistant in the same input
        user_hidden_states = output.hidden_states[-1][batch_size: batch_size*2] # data hidden states from 1/3 to 2/3 (corresponding to user)
        assistant_hidden_states = output.hidden_states[-1][batch_size*2:] # data hidden states from 2/3 to 3/3 (corresponding to assistant)
        
        return {
            "main_output": output,
            "user_hidden_states": user_hidden_states,
            "assistant_hidden_states": assistant_hidden_states,
        }
        
    def compute_loss(self, encoder, inputs, return_outputs=False, num_items_in_batch=False):
        first_dim = inputs["input_ids_user"].shape[0]
        
        model_input = self._prepare_input_data(inputs)
        
        last_user_index = self._last_token_index(inputs["attention_mask_user"])
        last_assistant_index = self._last_token_index(inputs["attention_mask_assistant"])
        
        results = self.forward(encoder, model_input)
        
        main_output = results["main_output"]
        lm_loss = main_output.loss
        
        user_pred_embed = results["user_hidden_states"][range(first_dim), last_user_index, :]
        assistant_last_token_embed = results["assistant_hidden_states"][range(first_dim), last_assistant_index, :]
        
        cos_sim = torch.cosine_similarity(user_pred_embed, assistant_last_token_embed)
        
        jepa_loss = 1 - torch.mean(cos_sim)
        
        total_loss = self.gamma * lm_loss + self.jp_lambda * jepa_loss
        
        return total_loss
        
    def _last_token_index(self, attention_mask):
        return (attention_mask.sum(dim=1) - 1).long() ## returns last index before padding starts
    
    def _prepare_input_data(self, inputs):
        return {
            "input_ids": torch.cat([
                inputs["input_ids"],
                inputs["input_ids_user"],
                inputs["input_ids_assistant"]
            ], dim=0),
            "labels": torch.cat([
                inputs["labels"],
                inputs["labels_user"],
                inputs["labels_assistant"]
            ], dim=0),
            "attention_mask": torch.cat([
                inputs["attention_mask"],
                inputs["attention_mask_user"],
                inputs["attention_mask_assistant"]
            ], dim=0)
        }
    
def create_labels(input_ids, attention_mask):
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    return labels
    
def init_tokenizer(tokenizer, dataset, max_length=60):
    full_input_ids = []
    full_att_masks = []
    full_labels = []
    input_ids_user = []
    attention_mask_users = []
    labels_user = []
    input_ids_assistant = []
    attention_mask_assistants = []
    labels_assistant = []
    
    for message in dataset["messages"]:
        
        # full_msg = message["user_msg"] + "<|perception|>" + message["assistant_msg"] + "<|eot_id|>" -> better for JEPA loss computing, will be implemented later on

        full_msg = message["user_msg"] + "<|eot_id|>" # lightweight solution for testing

        tokenized_full = tokenizer(
            full_msg,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        input_ids = tokenized_full["input_ids"]
        attention_mask = tokenized_full["attention_mask"]
        labels = create_labels(input_ids, attention_mask)
        full_input_ids.append(input_ids)
        full_att_masks.append(attention_mask)
        full_labels.append(labels)
        
        user_message = message["user_msg"]
        
        to_add = NUM_PRED_TOKENS
        while to_add > 0:
            user_message += f"<|predictor_{to_add}|>"
            to_add -= 1
            
        tokenized_user = tokenizer(
            user_message,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        input_ids_user.append(tokenized_user["input_ids"])
        attention_mask_users.append(tokenized_user["attention_mask"])
        labels_user.append(torch.full_like(tokenized_user["input_ids"], fill_value=-100)) # disregarding in LM loss
        
        assistant_message = message["assistant_msg"]
        tokenized_assistant = tokenizer(
            assistant_message,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        input_ids_assistant.append(tokenized_assistant["input_ids"])
        attention_mask_assistants.append(tokenized_assistant["attention_mask"])
        labels_assistant.append(torch.full_like(tokenized_assistant["input_ids"], fill_value=-100)) # disregarding in LM loss

    return {
        "input_ids": torch.cat(full_input_ids, dim=0),
        "attention_mask": torch.cat(full_att_masks, dim=0),
        "labels": torch.cat(full_labels, dim=0),
        "input_ids_user": torch.cat(input_ids_user, dim=0),
        "attention_mask_user": torch.cat(attention_mask_users, dim=0),
        "input_ids_assistant": torch.cat(input_ids_assistant, dim=0),
        "attention_mask_assistant": torch.cat(attention_mask_assistants, dim=0),
        "labels_user": torch.cat(labels_user, dim=0),
        "labels_assistant": torch.cat(labels_assistant, dim=0)
    }