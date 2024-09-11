import torch.nn as nn
import torch
import math

#TODO1
class MultiHeadAttention(nn.Module):
    def __init__(self, dim=768, num_heads=16, attn_drop=0.1):
        super(MultiHeadAttention, self).__init__()
        self.token_dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * self.num_heads == self.token_dim, 'Token dimension must be divisible by number of heads'
        
        # Linear transformation: Weight of matrices Q, K, V
        self.w_q = nn.Linear(in_features=self.token_dim, out_features=self.token_dim)
        self.w_k = nn.Linear(in_features=self.token_dim, out_features=self.token_dim)
        self.w_v = nn.Linear(in_features=self.token_dim, out_features=self.token_dim)
 
        self.dropout = nn.Dropout(attn_drop)

        # Linear transformation of concatenated tensor
        self.linear_output = nn.Linear(in_features=self.token_dim, out_features=self.token_dim)

    def attention(self, q, k ,v, drop=None):
        dim = q.size(-1)
        QK_T = torch.matmul(q, k.transpose(-1, -2))
        QK_T = nn.functional.softmax((QK_T / math.sqrt(dim)), dim=-1)
        if drop is not None:
            QK_T = drop(QK_T)
        output = torch.matmul(QK_T, v)
        return output


    def forward(self, x):
        ''' Hint: input x tensor shape is (batch_size, num_image_tokens, dim), 
            because the bidirectional transformer first will embed each token to dim dimension, 
            and then pass to n_layers of encoders consist of Multi-Head Attention and MLP. 
            # of head set 16
            Total d_k , d_v set to 768
            d_k , d_v for one head will be 768//16.
        ''' 
        batch_size, seq_len = x.size(0), x.size(1)

        # Linear transformation to multihead (batch_size, seq_len, dim) -> (batch_size, num_heads, seq_len, head_dim)
        q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        k = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        v = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)

        # Concat back (batch_size, num_heads, seq_len, head_dim) -> (batch_size, seq_len, dim)
        output = self.attention(q, k, v, self.dropout)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.token_dim)

        return self.linear_output(output)
    

class MLP(nn.Sequential):
    def __init__(self, dim=768, hidden_dim=3072, drop_rate=0.1):
        super(MLP, self).__init__(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=0.1)
        )
        
    def forward(self, input):
        return super().forward(input)
    
    
class TokenPredictor(nn.Sequential):
    def __init__(self, dim=768):
        super(TokenPredictor, self).__init__(
            nn.Linear(in_features=dim, out_features=dim),
            nn.GELU(),
            nn.LayerNorm(dim, eps=1e-12)
        )
        
    def forward(self, input):
        return super().forward(input)
    
    
class Encoder(nn.Module):
    def __init__(self, dim=768, hidden_dim=1536):
        super(Encoder, self).__init__()
        self.Attention = MultiHeadAttention(dim)
        self.LayerNorm1 = nn.LayerNorm(dim, eps=1e-12)
        self.LayerNorm2 = nn.LayerNorm(dim, eps=1e-12)
        self.MLP = MLP(dim, hidden_dim)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        attn = self.Attention(x)
        attn = self.dropout(attn)
        
        x = x + attn
        x = self.LayerNorm1(x)
        
        mlp = self.MLP(x)
        x = x + mlp
        return self.LayerNorm2(x)
    

if __name__ == "__main__":
    model = MultiHeadAttention(6,2)
    b, seq_len, dim = 1, 10, 6
    x = torch.ones(b, seq_len, dim)
    print(x.size(), x)
    output = model(x)
    print(output.size(), output)
    print('Parameters:', sum(p.numel() for p in model.parameters()))
    m = nn.MultiheadAttention(6,2)
    print(sum(p.numel() for p in m.parameters()))