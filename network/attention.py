import torch
import torch.nn as nn

"""
Implementation of self-attention refers to "Attention is All You Need"
"""

class MultiHeadAttentionLayer(nn.Module):
    """
    Multihead self-attention
    scaled dot-product attention
    
    Args: 
        hidden_dim (int): size of context vector
        n_heads (int): number of attention heads
        dropout_p (float): dropout ratio
    
    Inputs: query, key, value, mask 
    - ** query ** - : (batch size, query len, hidden dim) 
                      contains each target vector that need to find relevant 
                      context from other vectors
    - ** key ** - : (batch size, key len, hidde dim) other context vectors
    - ** value ** -: (batch size, value len, hidden dim) multipled with attention scores 
                     ultimately to derived the attention-weighted x

    Outputs: x, attention
    - ** x ** -: the vector weighted by attention score
    - ** attention **-: attention scores
    """
    def __init__(self, hidden_dim, n_heads, dropout_p, device):
        super().__init__()
        assert hidden_dim % n_heads == 0
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        # split hidden vector into h heads
        self.head_dim = hidden_dim // n_heads
        
        self.fc_q = nn.Linear(hidden_dim, hidden_dim)
        self.fc_k = nn.Linear(hidden_dim, hidden_dim)
        self.fc_v = nn.Linear(hidden_dim, hidden_dim)

        self.fc_o = nn.Linear(hidden_dim, hidden_dim)
       
        # dropout is applied to the attention (in offical implementation)
        self.dropout = nn.Dropout(dropout_p)
       
        # scaling factor |vector_dim| to prevent the dot product from growing large
        # causing gradients to become too small
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
       

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        # Q: (batch size, query len, hidden dim)
        Q = self.fc_q(query)
        # K: (batch size, key len, hidden dim)
        K = self.fc_k(key)
        # V: (batch size, value len, hidden dim)
        # keys dim = values dim
        V = self.fc_v(value)
        
        # permute in order to be multipled
        # Q: (batch size, n heads, query len, head dim)
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        # K: (batch size, n heads, key len, head dim)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        # V: (batch size, n heads, value len, head dim)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # energy: (batch size, n heads, query len, key len)
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
       
        # avoid paying attention to PAD
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        # attention: (batch size, n heads, query len, key len)
        attention = torch.softmax(energy, dim = -1)
        # x: (batch size, n heads, query len, head dim) 
        x = torch.matmul(self.dropout(attention), V)
        # x: (batch size, query len, n heads, head dim)
        # in order to perform .view(), we need first .contiguous() a permuted tensor
        x = x.permute(0, 2, 1, 3).contiguous()
        # x: (batch size, query len, hidden dim)
        x = x.view(batch_size, -1, self.hidden_dim)
        # x: (batch size, query len, hidden dim)        
        x = self.fc_o(x)
         
        return x, attention
