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

class multiHeadAttention(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = heads
        assert d_model % heads == 0, "D_model is not divisible by number of heads"

        self.d_k = self.d_model//self.h

        #defining Wq, Wk, Wv, Wo
        self.w_q = nn.Linear(d_model,d_model)  
        self.w_k = nn.Linear(d_model,d_model)
        self.w_v = nn.Linear(d_model,d_model)
        self.w_o = nn.Linear(d_model,d_model)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        attention_score = ((query @ key.transpose(-2,-1))/math.sqrt(d_k))

        if mask is not None:
            attention_score.masked_fill_(mask==0,-1e9)
        attention_score = torch.softmax(attention_score, dim=-1)
    
        return (attention_score @ value), attention_score # returning tuple for visualization
        

    def forward(self, q, k, v, mask):
        self.q = self.w_q(q)
        self.k = self.w_k(k)
        self.v = self.w_v(v) 

        query = query.view(query.shape[0],query.shape[1],self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0],key.shape[1],self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0],value.shape[1],self.h, self.d_k).transpose(1,2)
        
        x,self.attention_score = multiHeadAttention.attention(query,key,value,mask,self.dropout)

        x = x.transpose(1,2).contiguous().view(x.shape[0],x.shape[1], -1, self.h*self.d_k) #changing back to original dimensions

        return self.w_o(x)

class skipConnection(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = layerNormalization()
    
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))  # is norm before sublayer?
    
class encoderBlock(nn.Module):
    def __init__(self, attention_block: multiHeadAttention, feed_forward: feedForward, dropout: float) -> None:
        self.attention_block = attention_block
        self.feed_forward = feed_forward
        self.skip_connections = nn.ModuleList([skipConnection(dropout) for i in range(2)])

    def forward(self, x, src_mask): #src_mask masks the interaction of the padding words with our tokens in input of the encoder
        x = self.skip_connections[0](x,lambda x: self.attention_block(x,x,x,src_mask))

        x = self.skip_connections[1](x,self.feed_forward)

        return x
    
class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList)->None:
        super().__init__()
        self.layers = layers
        self.norm = layerNormalization()
    
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x,mask)
        
        return self.norm(x)
    
class decoderBlock(nn.Module):
    def __init__(self,masked_attention_block: multiHeadAttention, cross_attention_block: multiHeadAttention, feed_forward: feedForward, dropout: float) -> None:
        self.masked_attention_block = masked_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward =  feed_forward
        self.skip_connections = nn.ModuleList(skipConnection(dropout) for i in range(3))
    
    def forward(self, x, encoder_output, src_mask, target_mask):
        x = self.skip_connections[0](x, lambda x: self.masked_attention_block(x,x,x,target_mask))

        x = self.skip_connections[1](x, lambda x: self.cross_attention_block(x,encoder_output,encoder_output,src_mask))

        x = self.skip_connections[2](x, self.feed_forward)

        return x
    
class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = layerNormalization()
    
    def forward(self, x, encoder_output, src_mask, target_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, target_mask)
        return self.norm(x)
    
class Linear(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.linearProj = nn.Linear(d_model,vocab_size)
    
    def forward(self, x):
        return torch.log_softmax(self.linearProj(x),dim=-1)

class Transformer(nn.Module):
    def __init__(self, encoder:Encoder, decoder: Decoder, in_embed:inputEmbeddings, out_embed: inputEmbeddings, in_pos: positionalEncoding, out_pos: positionalEncoding, linear: Linear):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.in_embed = in_embed
        self.out_embed = out_embed
        self.in_pos = in_pos
        self.out_pos = out_pos
        self.linear = linear

    def encode(self, input, src_mask):
        input = self.in_embed(input)
        input = self.in_pos(input)

        return self.encoder(input, src_mask)
    
    def decode(self, encoder_output, src_mask, tgt_mask):
        tgt = self.out_embed(tgt)
        tgt = self.out_pos(tgt)

        return self.decode(tgt, encoder_output, src_mask, tgt_mask)
    
    def linear_project(self,x):
        return self.linear(x)
    

