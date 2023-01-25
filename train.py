import os
import pickle
import argparse


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint


from CONST import TRAIN_DIR, VALID_DIR, TEST_DIR, DEVICE
from network import Seq2Seq, Fac2Exp
from trainer import get_device, set_seed
from corpus import VocabFacExpan, load_file, preprocess_src_trg, Collater, Fac2expDataset
from run_evaluate import evaluate



def train(exper_dir, train_dir, valid_dir, test_dir=None, 
          batch_size=500, max_len=32, hidden_dim=256, 
          n_layers=3, n_heads=8, posff_dim=512, dropout_p=0.1,
          lr=0.0005, num_workers=8, seed=1234):
    """
    Run training and validation on datasets

    Args:
        exper_dir (str): path specifying the directory for 
                         dumping checkpoints, vocab
        train_dir (str): path specifying the directory of trainset
        valid_dir (str): path specifying the directory of validation set
        test_dir (str): path specifying the directory of test set
        batch_size (int): size of mini batch
        max_len (int): max length of sentences in the dataset
        hidden_dim (int): size of hidden state of encoder and decoder
        n_layers (int): number of layers of encoder and decoder
        n_heads (int): number of heads of attention
        posff_dim (int): size of position-wise feedforward layer
        dropout_p (float): dropout rate
        lr (float): learning rate
        num_workers (int): number of workers for DataLoader
        seed (int): generate random number

    Return:
        model (Fac2Exp): a trained model
    """
    set_seed(seed)

    print("Load dataset...")
    trainset = load_file(train_dir)
    validset = load_file(valid_dir)
    if test_dir:
        testset = load_file(test_dir)

    print("Create vocab...")
    vocab = VocabFacExpan()
    vocab.build_vocab(trainset)

    print("Create tensors...")
    train_tensor = preprocess_src_trg(trainset, vocab)
    valid_tensor = preprocess_src_trg(validset, vocab)

    print("Create dataloader")
    collate_fn = Collater(vocab)
    train_dataloader = DataLoader(
        Fac2expDataset(train_tensor),
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    valid_dataloader = DataLoader(
        Fac2expDataset(valid_tensor),
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    with open(os.path.join(exper_dir, "vocab.pickle"), "wb") as f:
            pickle.dump(vocab, f)
    
    model = Fac2Exp(
        vocab=vocab, device=DEVICE, max_len=max_len, hidden_dim=hidden_dim, 
        n_layers=n_layers, n_heads=n_heads, posff_dim=posff_dim, 
        dropout_p=dropout_p, lr=lr).to(DEVICE)

    #trainer = pl.Trainer(gpus=1, max_epochs=3, default_root_dir=exper_dir)
    trainer = pl.Trainer(max_epochs=3, default_root_dir=exper_dir)

    trainer.fit(model, train_dataloader, valid_dataloader)
    model.to(DEVICE)
    if test_dir:
        test_score = evaluate(model.seq2seq, testset, batch_size=batch_size)
        with open(os.path.join(exper_dir, "test.txt"), "w") as f:
            f.write(f"{test_score:.4f}\n")

    return model





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exper_dir", type=str, default="experiment/")
    parser.add_argument("-t", "--train_dir", type=str, default="data/train_split_debug.txt")
    parser.add_argument("-v", "--valid_dir", type=str, default="data/valid_split_debug.txt")
    parser.add_argument("-i", "--test_dir", type=str, default="data/test_split_debug.txt")

    parser.add_argument("-l", "--max_len", type=int, default=32)
    parser.add_argument("-s", "--hidden_dim", type=int, default=256)
    parser.add_argument("-y", "--n_layers", type=int, default=3)
    parser.add_argument("-d", "--n_heads", type=int, default=8)
    parser.add_argument("-p", "--posff_dim", type=int, default=512)
    
    parser.add_argument("--dropout_p", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    
    args = parser.parse_args()


    train(
        exper_dir=args.exper_dir, 
        train_dir=args.train_dir, 
        valid_dir=args.valid_dir, 
        test_dir=args.test_dir, 
        batch_size=args.batch_size, 
        max_len=args.max_len, 
        hidden_dim=args.hidden_dim, 
        n_layers=args.n_layers, 
        n_heads=args.n_heads, 
        posff_dim=args.posff_dim, 
        dropout_p=args.dropout_p, 
        lr=args.lr, 
        num_workers=args.num_workers, 
        seed=args.seed)

