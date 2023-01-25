import torch.nn as nn

from .network_util import pad_mask, and_mask, initialize_network
from .encoder import Encoder
from .decoder import Decoder


"""
Transformer architecture refers to
https://github.com/bentrevett/pytorch-seq2seq/blob/master/6%20-%20Attention%20is%20All%20You%20Need.ipynb
"""


class Seq2Seq(nn.Module):
    """
    Implementation of transformer based sequence to sequence model

    Args:   
        vocab (VocabFacExpan): token_ID alignment for both encoder and decoder side
        max_len (int): maximum length for deciding the dim of positional embedding 
        hid_dim (int): size of hidden state
        enc_layers (int): number of encoder layers
        dec_layers (int): number of decoder layers
        enc_heads (int): number of encoder's attention head
        dec_heads (int): number of decoder's attention head
        enc_pf_dim (int): size of encoder's positionwise feedforward layer
        enc_heads (int): size of decoder's positionwise feedforward layer
        enc_dropout (float): encoder's dropout rate
        dec_dropout (float): decoder's dropout rate
    
    Inputs: src, trg
    - ** src ** -: (batch size, src len) source sequence
    - ** trg ** -: (batch size, trg len) target sequence

    Outputs: output, attention
    - ** output ** - : predicted token ids 
    - ** attention ** -: attention score generated  during decoding
    """
    def __init__(self, vocab, device, max_len=32, hid_dim=256, enc_layers=3,
        dec_layers=3, enc_heads=8, dec_heads=8, enc_pf_dim=512,
        dec_pf_dim=512, enc_dropout=0.1, dec_dropout=0.1):
        super().__init__()
        
        self.vocab = vocab
        self.max_len = max_len

        self.encoder = Encoder(
            vocab.vocab_size, hid_dim, enc_layers, enc_heads,
            enc_pf_dim, enc_dropout, device)

        self.decoder = Decoder(
            vocab.vocab_size, hid_dim, dec_layers, dec_heads,
            dec_pf_dim, dec_dropout, device)

        self.criterion = nn.CrossEntropyLoss(ignore_index=self.vocab.PAD)
        initialize_network(self)
        self.to(device)


    def forward(self, src, trg):
        # src_mask: (batch size, 1, 1, src len)
        # trg_mask: (batch size, 1, 1, src len)
        src_mask = pad_mask(src, self.vocab.PAD)
        trg_mask = and_mask(trg, self.vocab.PAD)
            
        # enc_src: (batch size, src len, hid dim)
        enc_src = self.encoder(src, src_mask)

        # output: (batch size, trg len, output dim)
        # attention: (batch size, n heads, trg len, src len)
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)

        return output, attention
