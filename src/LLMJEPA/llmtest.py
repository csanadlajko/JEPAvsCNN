import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

class MiniLLMJEPA:
    
    def __init__(self, model_name="gpt2", gamma_lm=1.0, lambda_jepa = 0.1):
        self.encoder = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.gamma_lm = gamma_lm
        self.lambda_jepa = lambda_jepa
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def tokenize_input(self, user_text, assistant_text):
        full_msg = f"User: {user_text}\nAssistant: {assistant_text}"
        user_msg = f"User: {user_text}"
        assistant_msg = f"Assistant: {assistant_text}"
        
        full_msg_tokens = self.tokenizer(
            full_msg,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        user_tokens = self.tokenizer(
            user_msg,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        assistant_tokens= self.tokenizer(
            assistant_msg,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        return {
            'full': full_msg_tokens,
            'user': user_tokens,
            'assistant': assistant_tokens
        }
        
    def get_last_embedding(self, hidden_states, attention_mask):
        
        batch_size = hidden_states.shape[0]
        last_token_indices = []
        
        for i in range(batch_size):
            non_padding_indices = torch.where(attention_mask[i] == 1)[0]
            if len(non_padding_indices) > 0:
                last_idx = non_padding_indices[-1].item()
            else:
                last_idx = 0
            last_token_indices.append(last_idx)
        
        last_token_indices = torch.tensor(last_token_indices, device=self.device)
        batch_indices = torch.arange(batch_size, device=self.device)
        embeddings = hidden_states[batch_indices, last_token_indices, :]
        
        return embeddings
    
    def compute_jepa_loss(self, user_embed, assistant_embed):
        cosine_sim = F.cosine_similarity(user_embed, assistant_embed, dim=-1)
        
        loss = 1.0 - torch.mean(cosine_sim)
        
        return loss
    
    def forward(self, user_text, assistant_text):
        
        inputs = self.tokenize_input(user_text, assistant_text)
        print
        full_msg_output = self.encoder(
            **inputs['full'],
            labels=inputs['full']['input_ids'],
            output_hidden_states=True
        )
        lm_loss = full_msg_output.loss
        
        user_output = self.encoder(
            **inputs['user'],
            output_hidden_states=True
        )
        last_user_embedding = self.get_last_embedding(
            user_output.hidden_states[-1],
            inputs['user']['attention_mask']
        )
        
        assistant_output = self.encoder(
            **inputs['assistant'],
            output_hidden_states=True
        )
        last_assistant_embedding = self.get_last_embedding(
            assistant_output.hidden_states[-1],
            inputs['assistant']['attention_mask']
        )
        
        jepa_loss = self.compute_jepa_loss(last_user_embedding, last_assistant_embedding)
        
        final_loss = self.gamma_lm * lm_loss + self.lambda_jepa * jepa_loss
        
        return {
            'total_loss': final_loss,
            'jepa_loss': jepa_loss,
            'lm_loss': lm_loss,
            'user_embed': last_user_embedding,
            'assistant_embed': last_assistant_embedding
        }
        
    
    def train(self, user_text, assistant_text):
        step_result = self.forward(user_text, assistant_text)        
        step_result['total_loss'].backward()
        
    
user_text = "Mi Magyarország fővárosa?"
assistant_text = "Magyarország fővárosa Budapest."

testmodel = MiniLLMJEPA()

result_dict = testmodel.forward(user_text, assistant_text)

print(result_dict)