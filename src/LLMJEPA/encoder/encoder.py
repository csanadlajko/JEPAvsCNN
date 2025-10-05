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

device = "cuda" if torch.cuda.is_available() else "cpu"

class TextEncoder(Trainer):
    
    def __init__(self, model_name=MODEL_NAME, pred_tokens=NUM_PRED_TOKENS):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        new_tokens = [f"<|predictor_{i+1}|>" for i in range(pred_tokens)]
        new_tokens += ["<|perception|>", "<|eot_id|>"]
        self.tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
        self.encoder = AutoModelForCausalLM.from_pretrained(model_name)
        self.encoder.resize_token_embeddings(len(self.tokenizer))
        
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
                return_tensors=None
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
                return_tensors=None
            )
            
            user_input_ids.append(tokenized_user["input_ids"])
            user_attention_masks.append(tokenized_user["attention_mask"])
            user_labels.append([-100] * len(tokenized_user["input_ids"])) # disregarding in LM loss
            
            assistant_message = message["assistant_msg"]
            tokenized_assistant = self.tokenizer(
                assistant_message,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors=None
            )
            
            assistant_input_ids.append(tokenized_assistant["input_ids"])
            assistant_attention_masks.append(tokenized_assistant["attention_mask"])
            assistant_labels.append([-100] * len(tokenized_assistant["input_ids"])) # disregarding in LM loss
            
        return {
            "input_ids": full_input_ids,
            "attention_mask": full_att_masks,
            "labels": full_labels,
            "user_input_ids": user_input_ids,
            "user_attention_mask": user_attention_masks,
            "assistant_input_ids": assistant_input_ids,
            "assistant_attention_mask": assistant_attention_masks,
            "user_labels": user_labels,
            "assistant_labels": assistant_labels
        }
        
    def create_labels(self, input_ids, attention_mask):
        labels = []
        for i, mask in enumerate(attention_mask):
            if mask == 0:
                labels.append(-100)
            else: labels.append(input_ids[i])
        return labels
    
    
    def forward(self, dataset):
        tokenized_conv = self.tokenize_conversation(dataset)
        
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
        
        output = self.encoder(**llm_input, output_hidden_states=True)
        
        batch_size = llm_input["input_ids"].shape[0] // 3 # as we gave full, user, assistant in the same input
        user_hidden_states = output[batch_size: batch_size*2] # data hidden states from 1/3 to 2/3 (corresponding to user)
        assistant_hidden_states = output[batch_size*2:] # data hidden states from 2/3 to 3/3 (corresponding to assistant)
        
        return {
            "main_output": output,
            "user_hidden_states": user_hidden_states,
            "assistant_hidden_states": assistant_hidden_states
        }

    
    # def _last_token_index(self, input_ids, attention_mask):
    #     last_token_id = 0
    #     for id, mask in zip(input_ids, attention_mask):
    #         if mask != 0:
    #             last_token_id = id
    #             break
    #     return last_token_id
                
    
    # def forward(self, dataset):
        
    #     tokenized_conv = self.tokenize_conversation(dataset)
    #     full_input_ids = tokenized_conv["input_ids"]
    #     full_att_masks = tokenized_conv["attention_mask"]
    #     user_input_ids = tokenized_conv["user_input_ids"]
    #     user_att_masks = tokenized_conv["user_attention_mask"]
    #     last_user_id = self._last_token_index(user_input_ids, user_att_masks)
    #     user_pred_hidden_state = tokenized_conv["tokenized_user"][last_user_id].hidden_state
    #     assistant_hidden_state = tokenized_conv["tokenized_assistant"].hidden_state