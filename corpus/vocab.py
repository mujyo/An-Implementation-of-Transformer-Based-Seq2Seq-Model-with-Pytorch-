import re

import torch

class VocabBase:
    """
    Base token_ID alignments containing common attributes and funcs
    """
    PAD = 0
    SOS = 1
    EOS = 2
    UNK = 3
    def __init__(self):
        self.indx2tok = {self.PAD:"PAD", self.SOS:"SOS", self.EOS:"EOS", self.UNK:"UNK"}
        self.tok2indx = {v:k for k, v in self.indx2tok.items()}
        self.vocab_size = 4
    
    def add_tokens(self, tokens):
        """
        Add each token within a sentence into the vocab
        
        Args:
            tokens (list): list of tokens to be added into vocabulary set
        """
        for tok in tokens:
            if tok not in self.tok2indx:
                self.tok2indx[tok] = self.vocab_size
                self.indx2tok[self.vocab_size] = tok
                self.vocab_size += 1

    def tokenize_sentence(self, sentence):
        """
        split sentence into tokens
        need to be overwritten 
        
        Args: 
            sentence (str): target string to be tokenized
            need to be overrided 
        """
        raise NotImplementedError()

    def build_vocab(self, parallel_corpus):
        """
        Add tokens from parallel corpus into the vocabulary set

        Args:
            parallel_corpus (tuple): contains src sentence(tuple) and trg sentence (tuple)
        """
        src, trg = parallel_corpus
        for s in src:
            self.add_tokens(self.tokenize_sentence(s))
        for t in trg:
            self.add_tokens(self.tokenize_sentence(t))
    
    def transform_tok2index(self, seq):
        """
        Transform a sequence of tokens into corresponding ids
        
        Args:
            seq (list): list of tokens

        Return:
            ids (list): list of ids with SOS at head, EOS at tail
        """
        ids = [self.tok2indx.get(tok, self.UNK) for tok in self.tokenize_sentence(seq)]
        ids = [self.SOS] + ids + [self.EOS]
        return ids

    def transform_tok2tensor(self, seq):
        return torch.LongTensor(self.transform_tok2index(seq))


class VocabFacExpan(VocabBase):
    """
    token_ID alignments for factorized polynomial expansion
    """
    def tokenize_sentence(self, sentence):
        """
        tokenized a string into function names, digits, and alphabet
        according to the regexp
        Args:
            sentence: string

        Return:
            a string splitted into tokens
        """
        return re.findall(r"sin|cos|tan|\w|\(|\)|\+|-|\*+", sentence.lower().strip())
