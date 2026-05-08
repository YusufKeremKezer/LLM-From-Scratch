from torch import nn
import torch.nn.functional as F
import torch
from torch import optim
import numpy as np

class AttentionLayer(nn.Module):
    pass

class ClassicAttentionLayer(AttentionLayer): # Can see future to cheat
    def __init__(self, embedding_dim:int=512):
        super().__init__()
        self.Q = nn.Linear(embedding_dim,embedding_dim)
        self.K = nn.Linear(embedding_dim,embedding_dim)
        self.V = nn.Linear(embedding_dim,embedding_dim)

    def forward(self,X):
        Q = self.Q(X)
        K = self.K(X)
        V = self.V(X)
        scores = Q @ K.transpose(-2,-1) # How related each query with keys.
        attention_weights = F.softmax(scores,dim=-1) # How important affair between them

        return attention_weights @ V # How important that token


class MultiHeadAttention(nn.Module):
    def __init__(self,attention_layer,num_heads:int=8,embedding_dim:int=512):
        super().__init__()
        self.attention_layers = nn.ModuleList([attention_layer(embedding_dim/num_heads) for _ in range(num_heads)])

    def forward(self,X):
        head_outputs = [layer(X) for layer in self.attention_layers]
        return torch.cat(head_outputs,dim=-1) # Concatenate all heads


class CasualAttentionLayer(AttentionLayer):
    def __init__(self,embedding_dim:int=512):
        super().__init__()
        self.Q = nn.Linear(embedding_dim,embedding_dim)
        self.K = nn.Linear(embedding_dim,embedding_dim)
        self.V = nn.Linear(embedding_dim,embedding_dim)

    def forward(self,X):
        # X.shape = (batch_size,sequence_len,embedding_dim)
        # Batch size means how many sentences we are processing in parallel by predicting next word.

        Q = self.Q(X) # Q.shape = Q @ X.transpose() (batch_size,sequence_len,embedding_dim) @ (embedding_dim x embedding_dim).T
        K = self.K(X)  
        V = self.V(X) # output Q and K and Vshape = (batch_size,sequence_len,embedding_dim) 
        scores = Q @ K.transpose(-2,-1) # How related each query with keys.
        scores/=np.sqrt(Q.shape[-1])
        # K.Transpose shape = (batch_size,embedding_dim,sequence_len)

        #scores.shape = (batch_size,sequence_len,sequence_len)
        mask = torch.triu(torch.full(scores.shape, float('-inf')), diagonal=1) # Masking future tokens with -inf to prevent cheating.
        scores +=mask
        attention_weights = F.softmax(scores,dim=-1) # How important affair between them

        return attention_weights @ V.transpose(-2,-1) # How valuable that token is within the context batch_size, seq_len, embedding_dim

class TransformerBlock(nn.Module):
    def __init__(self,embedding_dim:int=512, attention_layer:nn.Module=CasualAttentionLayer(), num_heads:int=8):
        super().__init__()
        self.mha = MultiHeadAttention(attention_layer,num_heads,embedding_dim)
        self.ff_dim = embedding_dim*2
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim,self.ff_dim), 
            nn.GELU(),
            nn.Linear(self.ff_dim,embedding_dim),
            nn.Dropout(0.2)
        )
        self.norm1 = nn.LayerNorm(embedding_dim) #pre normalization
        self.norm2 = nn.LayerNorm(embedding_dim) #post normalization

    def forward(self,X):
        X = self.mha.forward(X)
        X = self.norm1.forward(X)
        X = self.feed_forward.forward(X)
        X = self.norm2.forward(X)
        return X


class PositionalEncoding(nn.Module):
    pass
