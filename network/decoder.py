import torch
import torch.nn as nn
from .attention import MultiHeadAttentionLayer
from .feedforwardLayer import PositionwiseFeedforwardLayer


class Decoder(nn.Module):
    """
    Generate sequence of tokens with 2 multi-head attentions
    decoder side attention: attend to itself
    encoder-decoder attention: use decoder representation as query and 
                               encoder representation as key, value
    positional embedding and token embedding combined via elementwise sum
    ->scaled->dropout->n decoder layers-> linear layer
    source mask: mask PADs
    target mask: mask future tokens
    Args:
        output_dim (int): size of vocab set
        hidden_dim (int): size of contextual vector
        n_layers (int): number of decoder layers
        n_heads (int): number of heads
        posff_dim (int): size of positionwise feedforward layer
        dropout_p (float): dropout ratio

    Inputs: trg, enc_src, trg_mask, src_mask
        - ** trg ** -: (batch size, trg len) tokenID sequence 
        - ** enc_src ** - : (batch size, src len, hidden dim) contextual vectors of src
        - ** trg_mask ** -: (batch size, 1, trg len, trg len) target mask. mask future tokens
        - ** src_mask ** -: (batch size, 1, 1, src len) source mask. mask PADs

    Outputs: output, attention
        - ** output ** -: predicted tokenIDs
        - ** attention ** - : attention weights
    """
    def __init__(self, output_dim, hidden_dim, n_layers, n_heads, 
                 posff_dim, dropout_p, device, max_length=50):
        super().__init__()
        self.device = device
        self.embedding = nn.Embedding(output_dim, hidden_dim)
        self.pos_embedding = nn.Embedding(max_length, hidden_dim)
        self.layers = nn.ModuleList(
            [DecoderLayer(hidden_dim, n_heads, posff_dim, dropout_p, device) 
                for _ in range(n_layers)]) 
        
        self.fc_out = nn.Linear(hidden_dim, output_dim) 
        self.dropout = nn.Dropout(dropout_p)
        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim])).to(device)
        

    def forward(self, trg, enc_src, trg_mask, src_mask):
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]

        # pos: (batch size, trg len)
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
                            
        # trg: (batch size, trg len, hidden dim) 
        trg = self.dropout((self.embedding(trg) * self.scale) + self.pos_embedding(pos))
       
        for layer in self.layers:
            # trg: (batch size, trg len, hidden dim)
            # attention: (batch size, n heads, trg len, src len)
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)
        
        # output: (batch size, trg len, output dim) 
        output = self.fc_out(trg)
            
        return output, attention



class DecoderLayer(nn.Module):
    """
    Decoder layer with
    decoder side attention and encoder-decoder attention
    decoder side attention-> dropout, residual connection, layer normalization
    encoder-decoder attention-> dropout, residual connection, layer normalization
    -> positionwise feedforward layer-> 
    dropout, residual connection, layer normalization
    Args:
        hidden_dim: size of contextual vector
        n_heads: number of heads
        posff_dim: size of positionwise feedforward layer
        dropout_p: dropout ratio

    Inputs: trg, enc_src, trg_mask, src_mask
        - ** trg ** -: (batch size, trg len, hidden dim) target side contextual vectors
        - ** enc_scr ** -: (batch size, src len, hidden dim) source side contextual vectors
        - ** trg_mask ** -: (batch size, 1, trg len, trg len) mask against future tokens
        - ** src_mask ** -: (batch size, 1, 1, src len) mask against PAD
    Outputs: trg, attention
        - ** trg ** -: (batch size, trg len, hidden dim) target side contextual vectors
        - ** attention ** -: (batch size, n heads, trg len, src len) 
                             encoder-decoder attention weights
    """
    def __init__(self, hidden_dim, n_heads, posff_dim, dropout_p, device):
        super().__init__()
        self.attn_layer_norm = nn.LayerNorm(hidden_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hidden_dim)
        self.ff_layer_norm = nn.LayerNorm(hidden_dim)

        self.attention = MultiHeadAttentionLayer(hidden_dim, n_heads, dropout_p, device)
        self.encoder_attention = MultiHeadAttentionLayer(hidden_dim, n_heads, dropout_p, device)
        
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(
                                        hidden_dim, posff_dim, dropout_p)
        self.dropout = nn.Dropout(dropout_p)


    def forward(self, trg, enc_src, trg_mask, src_mask):
        # decoder attention
        _trg, _ = self.attention(trg, trg, trg, trg_mask)

        #dropout, residual connection and layer norm
        trg = self.attn_layer_norm(trg + self.dropout(_trg))

        #encoder attention
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)

        #dropout, residual connection and layer norm
        trg = self.attn_layer_norm(trg + self.dropout(_trg))

        #positionwise feedforward
        _trg = self.positionwise_feedforward(trg)

        #dropout, residual and layer norm
        trg = self.ff_layer_norm(trg + self.dropout(_trg))

        return trg, attention
