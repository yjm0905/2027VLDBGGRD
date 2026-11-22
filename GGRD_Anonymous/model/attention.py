import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
class MultiHeadAttention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, d_model, num_heads, dropout, residual):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.residual = residual
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(value_dim)

        self.W_query = nn.Linear(in_features=query_dim, out_features=d_model, bias=True)
        self.W_key = nn.Linear(in_features=key_dim, out_features=d_model, bias=True)
        self.W_value = nn.Linear(in_features=key_dim, out_features=value_dim, bias=True)
 
    def forward(self, query, key, mask=None):
        

        querys = self.W_query(query)  
        keys = self.W_key(key)  
        values = self.W_value(key)
 
        split_size = self.d_model // self.num_heads
        split_size2 = self.value_dim // self.num_heads

        querys = torch.stack(torch.split(querys, split_size, dim=2), dim=0)  

        keys = torch.stack(torch.split(keys, split_size, dim=2), dim=0)  
       
        values = torch.stack(torch.split(values, split_size2, dim=2), dim=0)  
       
        scores = torch.matmul(querys, keys.transpose(2, 3))  #
        scores = scores / (self.key_dim ** 0.5)
 
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(0).repeat(self.num_heads,1,querys.shape[2],1)
            scores = scores.masked_fill(mask, -np.inf)
        scores = F.softmax(scores, dim=3)
 
        
        out = torch.matmul(scores, values)  
        out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0)  
    
        out = self.dropout(out)

        return self.norm(out)

class GGAttention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, d_model, num_heads, dropout, residual, window_size, num_patches):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.residual = residual
        self.dropout = nn.Dropout(dropout)
        self.window_size = window_size
        self.num_patches = num_patches

        self.W_query = nn.Linear(in_features=query_dim, out_features=d_model, bias=True)
        self.W_key = nn.Linear(in_features=key_dim, out_features=d_model, bias=True)
        self.W_value = nn.Linear(in_features=key_dim, out_features=value_dim, bias=True)

        self.ln1 = nn.Linear(in_features=self.num_patches, out_features=self.window_size, bias=True)
        self.ln2 = nn.Linear(in_features=self.window_size, out_features=num_patches, bias=True)

        self.norm = nn.LayerNorm(value_dim)

    def forward(self, query, key, S, mask=None):# query :B T model  key :B T D   
                                                # S : B P D D
        residual = query.clone()
        raw_key = key

        key = key.permute(0, 2, 1) # B 64 33 
       
        key = self.ln2(key).permute(0,2,1) # B 16 33   
                                # S : B P D D
        key_unsqueezed = key.unsqueeze(-1)
       
        result = torch.matmul(S, key_unsqueezed)
        
        out = result.squeeze(-1) 
        key = self.ln1(out.permute(0,2,1)).permute(0,2,1)   
        

        querys = self.W_query(query)  #B 16 64
        keys = self.W_key(key)  # B 16 64
        values = self.W_value(raw_key)# B  64  512

        split_size = self.d_model // self.num_heads
        split_size2 = self.value_dim // self.num_heads

        querys = torch.stack(torch.split(querys, split_size, dim=2), dim=0) 

        keys = torch.stack(torch.split(keys, split_size, dim=2), dim=0) 
       
        values = torch.stack(torch.split(values, split_size2, dim=2), dim=0)  
       
        # score = softmax(QK^T / (d_k ** 0.5))
        scores = torch.matmul(querys, keys.transpose(2, 3))  # [h, B, N, M]
        scores = scores / (self.key_dim ** 0.5) #  8 B  16 16
 
        
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(0).repeat(self.num_heads,1,querys.shape[2],1)
            scores = scores.masked_fill(mask, -np.inf)

        scores = F.softmax(scores, dim=3)
 
        ## out = score * V
        out = torch.matmul(scores, values)  # [h, B, N, value_dim/h]
        out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0)  # [B, N, value_dim]
        out = self.dropout(out)
        
        if self.residual:
            return self.norm(out + residual)
        else:
            return self.norm(out)
