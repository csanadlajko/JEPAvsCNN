from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    Trainer
)
import torch
import json
from typing import Any

file = open("././parameters.json")
paramters: dict[str, Any] = json.load(file)["llmjepa"]
MODEL_NAME = paramters["MODEL_NAME"]
NUM_PRED_TOKENS = paramters["PRED_TOKENS"]
LM_GAMMA = paramters["LM_GAMMA"]
JEPA_LAMBDA = paramters["JEPA_LAMBDA"]

device = "cuda" if torch.cuda.is_available() else "cpu"

class TextEncoder(Trainer):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(kwargs.pop('model'))
        new_tokens = [f"<|predictor_{i+1}|>" for i in range(kwargs.pop('pred_tokens'))]
        new_tokens += ["<|perception|>", "<|eot_id|>"]
        self.tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
        self.encoder = AutoModelForCausalLM.from_pretrained(kwargs.pop('model'))
        self.encoder.resize_token_embeddings(len(self.tokenizer))
        self.gamma = kwargs.pop('gamma')
        self.jp_lambda = kwargs.pop('jp_lambda')
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
    def tokenize_conversation(self, dataset, max_length=60):
        """
        Dataset should be:
        {
            "messages": [
                {
                    "user_msg": "user message", "assistant_msg": "assistant message"
                },
                ...
            ]
        }
        """
        full_input_ids = []
        full_att_masks = []
        full_labels = []
        user_input_ids = []
        user_attention_masks = []
        user_labels = []
        assistant_input_ids = []
        assistant_attention_masks = []
        assistant_labels = []
        
        for message in dataset["messages"]:
            
            # full_msg = message["user_msg"] + "<|perception|>" + message["assistant_msg"] + "<|eot_id|>" -> better for JEPA loss computing, will be implemented later on

            full_msg = message["user_msg"] + "<|eot_id|>" # lightweight solution for testing

            tokenized_full = self.tokenizer(
                full_msg,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt"
            )
            
            input_ids = tokenized_full["input_ids"]
            attention_mask = tokenized_full["attention_mask"]
            labels = self.create_labels(input_ids, attention_mask)
            full_input_ids.append(input_ids)
            full_att_masks.append(attention_mask)
            full_labels.append(labels)
            
            user_message = message["user_msg"]
            
            to_add = NUM_PRED_TOKENS
            while to_add > 0:
                user_message += f"<|predictor_{to_add}|>"
                to_add -= 1
                
            tokenized_user = self.tokenizer(
                user_message,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt"
            )
            
            user_input_ids.append(tokenized_user["input_ids"])
            user_attention_masks.append(tokenized_user["attention_mask"])
            user_labels.append(torch.full_like(tokenized_user["input_ids"], fill_value=-100)) # disregarding in LM loss
            
            assistant_message = message["assistant_msg"]
            tokenized_assistant = self.tokenizer(
                assistant_message,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt"
            )
            
            assistant_input_ids.append(tokenized_assistant["input_ids"])
            assistant_attention_masks.append(tokenized_assistant["attention_mask"])
            assistant_labels.append(torch.full_like(tokenized_assistant["input_ids"], fill_value=-100)) # disregarding in LM loss
            
        return {
            "input_ids": torch.cat(full_input_ids, dim=0),
            "attention_mask": torch.cat(full_att_masks, dim=0),
            "labels": torch.cat(full_labels, dim=0),
            "user_input_ids": torch.cat(user_input_ids, dim=0),
            "user_attention_mask": torch.cat(user_attention_masks, dim=0),
            "assistant_input_ids": torch.cat(assistant_input_ids, dim=0),
            "assistant_attention_mask": torch.cat(assistant_attention_masks, dim=0),
            "user_labels": torch.cat(user_labels, dim=0),
            "assistant_labels": torch.cat(assistant_labels, dim=0)
        }
        
    def create_labels(self, input_ids, attention_mask):
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return labels
    
    
    def forward(self, dataset):
        tokenized_conv = self.tokenize_conversation(dataset)
        
        last_user_index = self._last_token_index(tokenized_conv["user_attention_mask"])
        last_assistant_index = self._last_token_index(tokenized_conv["assistant_attention_mask"])
        
        llm_input = {
            "input_ids": torch.cat([
                tokenized_conv["input_ids"],
                tokenized_conv["user_input_ids"],
                tokenized_conv["assistant_input_ids"]
            ], dim=0),
            "labels": torch.cat([
                tokenized_conv["labels"],
                tokenized_conv["user_labels"],
                tokenized_conv["assistant_labels"]
            ], dim=0),
            "attention_mask": torch.cat([
                tokenized_conv["attention_mask"],
                tokenized_conv["user_attention_mask"],
                tokenized_conv["assistant_attention_mask"]
            ], dim=0)
        }
        
        first_dim = tokenized_conv["user_input_ids"].shape[0]
        
        output = self.encoder(**llm_input, output_hidden_states=True)
        
        batch_size = llm_input["input_ids"].shape[0] // 3 # as we gave full, user, assistant in the same input
        user_hidden_states = output.hidden_states[-1][batch_size: batch_size*2] # data hidden states from 1/3 to 2/3 (corresponding to user)
        assistant_hidden_states = output.hidden_states[-1][batch_size*2:] # data hidden states from 2/3 to 3/3 (corresponding to assistant)
        
        return {
            "main_output": output,
            "user_hidden_states": user_hidden_states,
            "assistant_hidden_states": assistant_hidden_states,
            "last_user_index": last_user_index,
            "last_assistant_index": last_assistant_index,
            "first_dim": first_dim
        }
        
    def compute_loss(self, dataset):
        encoded_data = self.forward(dataset)
        first_dim = encoded_data["first_dim"]
        
        main_output = encoded_data["main_output"]
        lm_loss = main_output.loss
        
        user_last_token_idx = encoded_data["last_user_index"]
        assistant_last_token_idx = encoded_data["last_assistant_index"]
        user_pred_embed = encoded_data["user_hidden_states"][range(first_dim), user_last_token_idx, :]
        assistant_last_token_embed = encoded_data["assistant_hidden_states"][range(first_dim), assistant_last_token_idx, :]
        
        cos_sim = torch.cosine_similarity(user_pred_embed, assistant_last_token_embed)
        
        jepa_loss = 1 - torch.mean(cos_sim)
        
        total_loss = self.gamma * lm_loss + self.jp_lambda * jepa_loss
        
        return total_loss
        
    def _last_token_index(self, attention_mask):
        return (attention_mask.sum(dim=1) - 1).long() ## returns last index before padding starts