import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

class MiniLLMJEPA:
    
    def __init__(self, model_name="microsoft/DialoGPT-small"):
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def encode_text(self, text):
        print(f"INPUT TEXT IS: {text}")
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        print("----INPUTS TENSOR START----")
        print(inputs["input_ids"][0])
        print("----INPUTS TENSOR END----")
        with torch.no_grad():
            print("----OUTPUTS TENSOR START----")
            outputs = self.encoder(**inputs)
            print(outputs)
            print("----OUTPUTS TENSOR END----")
            embeddings = outputs.last_hidden_state[:, -1, :]
        
        return embeddings
    
    def compute_jepa_loss(self, user_text, assistant_text):
        user_embed = self.encode_text(user_text)
        
        assistant_embed = self.encode_text(assistant_text)
        
        cos_loss = F.cosine_similarity(user_embed, assistant_embed, dim=1)
        
        jepa_loss = 1.0 - torch.mean(cos_loss)
        
        return jepa_loss, user_embed, assistant_embed
    
user_text = "Mennyi 2+2?"
assistant_text = "2+2=4"

testmodel = MiniLLMJEPA()

loss, u_enbed, a_embed = testmodel.compute_jepa_loss(user_text, assistant_text)