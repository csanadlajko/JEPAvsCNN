from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
from typing import Any

file = open("././parameters.json")
paramters: dict[str, Any] = json.load(file)["llmjepa"]
MODEL_NAME = paramters["MODEL_NAME"]
NUM_PRED_TOKENS = paramters["PRED_TOKENS"]

device = "cuda" if torch.cuda.is_available() else "cpu"

class TextTokenizer:
    
    def __init__(self, model_name=MODEL_NAME, pred_tokens=NUM_PRED_TOKENS):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        new_tokens = [f"<|predictor_{i+1}|>" for i in range(pred_tokens)]
        new_tokens.append("<|perception|>")
        new_tokens.append("<|eot_id|>")
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
        user_input_ids = []
        user_attention_masks = []
        assistant_input_ids = []
        assistant_attention_masks = []
        
        for message in dataset["messages"]:
            
            full_msg = message["user_msg"] + "<|perception|>" + message["assistant_msg"] + "<|eot_id|>"

            tokenized_full = self.tokenizer(
                full_msg,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors=None
            )
            
            input_ids = tokenized_full["input_ids"]
            attention_mask = tokenized_full["attention_mask"]
            full_input_ids.append(input_ids)
            full_att_masks.append(attention_mask)
            
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
            
        return {
            "input_ids": full_input_ids,
            "attention_mask": full_att_masks,
            "labels": None, ## TODO return labels
            "user_input_ids": user_input_ids,
            "user_attention_mask": user_attention_masks,
            "assistant_input_ids": assistant_input_ids,
            "assistant_attention_mask": assistant_attention_masks
        }