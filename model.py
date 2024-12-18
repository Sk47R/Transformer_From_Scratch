import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    def __init__(self, d_model : int, vocab_size : int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model : int, seq_len : int, dropout : float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        
        pe = torch.zeros(seq_len, d_model) # CREATE A MATRIX OF SHAPE (seq_len, d_model)
        position = torch.arange(0, seq_len, dtype = torch.float).unsqueeze(1) # CREATE A VECTOR OF SHAPE (seq_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term) # APPLY SIN FUNCTION TO EVEN POSITION
        pe[:, 1::2] = torch.cos(position * div_term) # APPLY COS FUNCTION TO ODD POSITION

        pe = pe.unsqueeze(0) # SHAPE => (1, seq_len, d_model), FIRST DIMENSION FOR THE BATCH
        self.register_buffer("pe", pe) # SAVING 'pe' AS A CONSTANT TENSOR 

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # ADDING POSITIONAL EMBEDDING TO EACH TOKEN OF THE SENTENCE
        return self.dropout(x)



class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # MULTIPLIER
        self.bias = nn.Parameter(torch.zeros(1)) # ADDITIVE
        
    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim  = True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # W1 AND B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # W2 AND B2

    def forward(self, x):
        # (batch, seq_len, d_model) -> (batch, seq_len, d_ff) -> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    