import os
import pickle
import argparse


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint


from CONST import TRAIN_DIR, VALID_DIR, TEST_DIR
from network import Seq2Seq



class Fac2Exp(pl.LightningModule):
    """
    A sequence to sequence model for polynomial expansion
    encoder and decoder are based on Transformer  

    Args:
        vocab (VocabFacExpan): token-ID alignments
        max_len (int): max sequence length for specifying size
                       of positional embedding in seq2seq 
        hidden_dim (int): size of hidden state 
        n_layers (int): number of layers
        n_heads (int): number of heads
        posff_dim (int): size of positionwise embedding layer, 
        dropout_p (float): dropout rate
        lr (float): learning rate 
    """
    def __init__(self, vocab, device, max_len=32, hidden_dim=256, 
                 n_layers=3, n_heads=8, posff_dim=512, 
                 dropout_p=0.1, lr=0.0005):
        super().__init__()
        self.seq2seq = Seq2Seq(
            vocab, device=device, max_len=max_len, 
            hid_dim=hidden_dim, enc_layers=n_layers, 
            dec_layers=n_layers, enc_heads=n_heads, 
            dec_heads=n_heads, enc_pf_dim=posff_dim, 
            dec_pf_dim=posff_dim, enc_dropout=dropout_p, 
            dec_dropout=dropout_p)
        
        self.save_hyperparameters()
        del self.hparams["vocab"]



    def load_model(self, exper_dir, vocab_pickle="vocab.pickle", 
                    model_ckpt="fac2exp_model.ckpt"):
        """
        Load a checkpoint and vocabulary
        Args:
            exper_dir (str): root dir for experiment
            vocab_pickle (str): name of vocabulary file
            model_ckpt (str): path, combined with root dir to find checkpoint

        Return:
            model (FacExp): model for conducting polynomial expansion
        """
        with open(os.path.join(exper_dir, vocab_pickle), "rb") as f:
            vocab = pickle.load(f)

        model = self.load_from_checkpoint(
            os.path.join(exper_dir, model_ckpt),
            vocab=vocab).to(device)
        return model



    def forward(self, src, trg):
        """
        Call forward func of seq2seq
        Args:
            src: (batch size, src len) factorized polynomial
            trg: (batch size, trg len) expansion
        """
        self.seq2seq(src, trg)


    def configure_optimizers(self):
        """
        Optimizer and lr scheduler
        Override configure_optimizers() of pytorch lightning
        Return:
            Optimizer for seq2seq model
        """
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


    def training_step(self, batch, batch_idx):
        """
        To activate training Loop
        Override training_step() of pytorch lightning
        Args:
            batch: batch of src, trg pairs
            batch_idx: index of current batch

        Return:
            loss: training loss for current batch
        """
        src, trg = batch
        output, _ = self.seq2seq(src, trg[:, :-1])
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)
        loss = self.seq2seq.criterion(output, trg)
        self.log("train_loss", loss)
        return loss


    def validation_step(self, batch, batch_idx):
        """
        To activate validation loop
        Override validation_step() of pytorch lightning
        Args: 
            batch: batch of src, trg pairs
            batch_idx: index of current batch

        Return:
            loss: validation loss for current batch
        """
        src, trg = batch
        output, _ = self.seq2seq(src, trg[:, :-1])
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)
        loss = self.seq2seq.criterion(output, trg)
        self.log("val_loss", loss, prog_bar=True)
        return loss


    def test_step(self, batch, batch_idx):
        """
        To activate test loop
        Override test_step() of pytorch lightning
        Args:
            batch: batch of src, trg pairs
            batch_idx: index of current batch

        Return:
            loss: validation loss for current batch
        """
        return self.validation_step(batch, batch_idx)  

