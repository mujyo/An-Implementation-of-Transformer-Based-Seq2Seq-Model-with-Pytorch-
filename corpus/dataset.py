import torch.nn as nn
from torch.utils.data import Dataset


class Fac2expDataset(Dataset):
    """
    Customerized Dataset
    
    Args:
        data: dataset to be processed
    """
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)   



class Collater:
    """
    Data collater. 
    Conduct padding on a batch of sequences
    
    Args:
        vocab (VocabFacExpan): token-ID alignment
        predict (boolean): if predict, batch only contains src
    """
    def __init__(self, vocab, predict=False):
        self.vocab = vocab
        self.predict = predict

    def __call__(self, batch):
        # TODO: try pack_padded_sequence for faster processing
        if self.predict:
            # batch = src_tensors in predict mode
            return nn.utils.rnn.pad_sequence(
                batch, batch_first=True, padding_value=self.vocab.PAD
            )

        src_tensors, trg_tensors = zip(*batch)
        src_tensors = nn.utils.rnn.pad_sequence(
            src_tensors, batch_first=True, padding_value=self.vocab.PAD
        )
        trg_tensors = nn.utils.rnn.pad_sequence(
            trg_tensors, batch_first=True, padding_value=self.vocab.PAD
        )
        return src_tensors, trg_tensors



def preprocess_src_trg(src_trg, vocab):
    """
    Transform a (src, trg) pairs into the form that can be
    processed by model
    First tokenize the string, then transform it into id
    Last, convert the list into tensor 

    Args:
        src_trg (tuple): contains src sequences and trg sequences
        vocab (VocabFacExpan): token-ID alignment
    """
    tensors = [(vocab.transform_tok2tensor(src), vocab.transform_tok2tensor(trg)) 
                for src, trg in zip(src_trg[0], src_trg[1])]

    return tensors


