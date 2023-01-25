import torch
import torch.nn as nn

from .attention import MultiHeadAttentionLayer
from .feedforwardLayer import PositionwiseFeedforwardLayer



class Encoder(nn.Module):
    """
    encoding the input sequence
    Args:
        input_dim (int): size of vocab
        hidden_dim (int): size of context vectors
        n_layers (int): number of encoder layer
        n_head (int): number of attention heads
        pos_dim (int): size of positional embedding vector
        dropout_p (float): dropout ratio
        max_length (int): for making positional embedding

    Inputs: src, src_mask
    - ** src ** -: (batch size, src len) the input sequence
    - ** src_mask ** -: (batch size, 1, 1, src len) has value of 1 
                        when the token is not PAD otherwise 0
                        used when applying attention over the source sequence 
                        so that no attention is paid to PAD
    
    Outputs: src
    - ** src ** -: (batch size, src len, hid dim): the encoded input sequence
    """
    def __init__(self, input_dim, hidden_dim, n_layers, n_heads, 
                 posff_dim, dropout_p, device, max_length=50):
        super().__init__()
        self.device = device
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.pos_embedding = nn.Embedding(max_length, hidden_dim)
        self.layers = nn.ModuleList(
            [EncoderLayer(hidden_dim, n_heads, posff_dim, dropout_p, device)
                for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout_p)

        # scaling factor |vector_dim| to reduce variance 
        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim])).to(device)


    def forward(self, src, src_mask):
        batch_size = src.shape[0]
        src_len = src.shape[1]

        # generate input for positional embedding (batch size, src len)
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        # combine token embedding with positional embedding (batch size, src len, hidden dim)
        src = self.dropout((self.embedding(src) * self.scale) + self.pos_embedding(pos))
        for layer in self.layers:
            # src: (batch size, src len, hid dim)
            src = layer(src, src_mask)        

        return src



class EncoderLayer(nn.Module):
    """
    Conduct encoding with self-attention layer by layer
    -> pass the src, mask into multi-head attention layer
    -> perform dropout
    -> residual connection
    -> layer normalization
    -> position-wise feedforward layer
    -> dropout
    -> residual connection
    -> layer normalization
    Args:
        hidden_dim (int): size of context vectors
        n_head (int): number of attention heads
        pos_dim (int): size of positionwise feedforward layer
        dropout_p (float): dropout ratio

    Inputs: src, src_mask
    - ** src ** -: (batch size, src len, hidden dim) encoded src
    - ** src_mask ** -: (batch size, 1, 1, src len) has value of 1 
                        when the token is not PAD otherwise 0
                        used when applying attention over 
                        source sequence so that no attention is paid to PAD
    Outputs: src
    - ** src ** -: (batch size, src len, hid dim): the encoded input sequence
    """
    def __init__(self, hidden_dim, n_heads, posff_dim, dropout_p, device):
        super().__init__()
        self.attn_layer_norm = nn.LayerNorm(hidden_dim)
        self.ff_layer_norm = nn.LayerNorm(hidden_dim)
        self.attention = MultiHeadAttentionLayer(hidden_dim, n_heads, dropout_p, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hidden_dim, 
                                                                     posff_dim, 
                                                                     dropout_p)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, src, src_mask): 
        #self attention
        _src, _ = self.attention(src, src, src, src_mask)
        
        #dropout, residual connection and layer norm
        #src = (batch size, src len, hid dim)
        src = self.attn_layer_norm(src + self.dropout(_src))
        
        #positionwise feedforward
        _src = self.positionwise_feedforward(src)
        
        #dropout, residual and layer norm
        #src = [batch size, src len, hid dim]
        src = self.ff_layer_norm(src + self.dropout(_src))
         
        return src

