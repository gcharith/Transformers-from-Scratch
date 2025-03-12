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

        #creating a matrix to store the positional encodings of size seq_length X dim_model
        pos_enc = torch.zeros(seq_length,dim_model)

        #creating a tensor vector of length seq_length, using unsqueeze to get a column vector
        position = torch.arange(0,seq_length,dtype = torch.float).unsqueeze(1)
        denominator = torch.exp(torch.arange(0,dim_model,2).float()*(-math.log(10000.0)/dim_model))
        
        #sin for even position and cos for odd positions
        pos_enc[:,0::2] = torch.sin(position*denominator)
        pos_enc[:,1::2] = torch.cos(position*denominator)

        pos_enc = pos_enc.unsqueeze(0) #adding batch dimension (1,seq_length, dim_model)

        self.register_buffer('pos_enc',pos_enc)

    def forward(self, x):
        x = x + (self.pos_enc[:,:x.shape[1],:]).requires_grad(False) # adding the positional encoding to the input embedding. making positional encoding values nonlearnable since they are fixed
        return self.dropout(x)

class layerNormalization(nn.Module):
    def __init__(self, epsilon: float = 10**-6) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.alpha = nn.Parameter(torch.ones(1)) #multiplication scaler
        self.beta = nn.Parameter(torch.zeros(1)) #addition scaler
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True) #mean of everything after batch
        std = x.std(dim=-1, keepdim=True) #standard deviation

        return self.alpha * (x-mean) / (std + self.epsilon) + self.beta

class feedForward(nn.Module):
    def __init__(self, d_model: int,d_ff:int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)  #getting W1 and b1 in the feed forward
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff,d_model) #getting W2 and b2

    def forward(self,x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


        