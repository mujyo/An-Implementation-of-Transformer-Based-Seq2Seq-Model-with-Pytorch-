import torch
import torch.nn as nn



def initialize_network(model):
    """
    Initialize parameters

    Args:
        model (seq2seq): the model for conducting sequence generation
    """
    def _initialize_w(m):
            if hasattr(m, "weight") and m.weight.dim() > 1:
                nn.init.xavier_uniform_(m.weight.data)

    model.encoder.apply(_initialize_w)
    model.decoder.apply(_initialize_w)
    


def pad_mask(indx_seq, pad_index):
    """
    generate mask for src
    
    Args:
        indx_seq: (batch size, indx_seq len) tokenIDs in input sentence
        pad_index (int): PAD's id in vocab

    Return:
        mask: (batch size, 1, 1, indx_seq len) the mask where
        1 for non-PAD element 0 for PAD
    """
    mask = (indx_seq != pad_index).unsqueeze(1).unsqueeze(2)
    return mask



def and_mask(indx_seq, pad_index):
    """
    perform AND operation on 2 masks
    PAD mask & future token mask
    
    Args:
        indx_seq: (batch size, indx_seq len) tokenIDs in target sentence
        pad_index (int): PAD's id in vocab

    Return:
        mask: (batch size, 1, indx_seq len, indx_seq len)
    """
    # pad_mask: (batch size, 1, 1, indx_seq len)
    mask = pad_mask(indx_seq, pad_index)
    seq_len = indx_seq.shape[1]

    # generate matrix where elements in upper diagonal are 0,
    # elements below are 1
    # future_tokens_mask: (seq len, seq len)
    future_tokens_mask = torch.tril(torch.ones((seq_len, seq_len)).type_as(indx_seq)).bool()

    # (batch size, 1, seq len, seq len)
    return mask & future_tokens_mask
