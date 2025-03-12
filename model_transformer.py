import torch
import torch.nn  as nn
import math

class inputEmbeddings(nn.Module):
    def __init__(self, dim_model: int, vocab_size: int):
        super().__init__()
        self.dim_model = dim_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, dim_model)
    
    def forward(self,x):
        return math.sqrt(self.dim_model) * self.embedding(x)

class positionalEncoding(nn.Module):
    def __init__(self, dim_model: int, seq_length: int, dropout: float) -> None:
        super().__init()
        self.dim_model = dim_model
        self.seq_length = seq_length
        self.dropout = nn.Dropout(dropout)

        
