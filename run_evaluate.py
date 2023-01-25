import tqdm
import argparse

import numpy as np

import torch
from torch.utils.data import DataLoader

from corpus import Fac2expDataset
from network import pad_mask, and_mask
from evaluation import score, load_model
from corpus import load_file
from trainer import get_device
from CONST import TEST_DIR, DEVICE
from corpus import Collater
from corpus import VocabFacExpan



def predict_batch(seq2seq, src_tensor):
    """
    Conduct batch prediction
    Implementation refers to https://github.com/jaymody/seq2seq-polynomial/blob/master/train.py

    Args:
        model (Fac2Exp): trained model
        src_tensor: (batch size, src len) input sequences

    Return:
        pred_sentences (string): predicted sequence
        pred_words (list): tokens
        pred_attention (list): attention score during decoding
    """
    batch_size = len(src_tensor)

    # src_mask: (batch size, 1, 1, src len)
    src_mask = pad_mask(src_tensor, seq2seq.vocab.PAD)

    # encoder_hidden: (batch size, src len, hid dim)
    encoder_hidden = seq2seq.encoder(src_tensor, src_mask)

    # decoder_input = (batch_size, 1)
    decoder_input = torch.LongTensor([[seq2seq.vocab.SOS]
                                       for _ in range(batch_size)]
                                    ).to(DEVICE)

    for _ in range(seq2seq.max_len):
        # trg_mask: (batch size, 1, trg len, trg len)
        trg_mask = and_mask(decoder_input, seq2seq.vocab.PAD)

        # output: ([batch size, trg len, output dim)
        output, attention = seq2seq.decoder(decoder_input, encoder_hidden,
                                            trg_mask, src_mask)

        # preds: (batch_size, 1)
        preds = output.argmax(2)[:, -1].reshape(-1, 1)

        # decoder_input: (batch_size, trg len)
        decoder_input = torch.cat((decoder_input, preds), dim=-1)

    src_tensor = src_tensor.detach().cpu().numpy()
    decoder_input = decoder_input.detach().cpu().numpy()
    attention = attention.detach().cpu().numpy()

    pred_words = []
    pred_sentences = []
    pred_attention = []
    for src_indexes, trg_indexes, attn in zip(src_tensor, decoder_input, attention):
        # trg_indexes = [trg len = max len (filled with eos if max len not needed)]
        # src_indexes = [src len = len of longest sentence (padded if not longest)]
        # indexes where first eos tokens appear
        src_eosi = np.where(src_indexes == seq2seq.vocab.EOS)[0][0]
        _trg_eosi_arr = np.where(trg_indexes == seq2seq.vocab.EOS)[0]
        if len(_trg_eosi_arr) > 0:  # check that an eos token exists in trg
            trg_eosi = _trg_eosi_arr[0]
        else:
            trg_eosi = len(trg_indexes)

        # cut target indexes up to first eos token and also exclude sos token
        trg_indexes = trg_indexes[1:trg_eosi]

        # attn = [n heads, trg len=max len, src len=max len of sentence in batch]
        # we want to keep n heads, but we'll cut trg len and src len up to
        # their first eos token
        attn = attn[:, :trg_eosi, :src_eosi]  # cut attention for trg eos tokens

        words = [seq2seq.vocab.indx2tok[index] for index in trg_indexes]
        sentence = "".join(words)
        pred_words.append(words)
        pred_sentences.append(sentence)
        pred_attention.append(attn)

    return pred_sentences, pred_words, pred_attention


def predict(seq2seq, test_src, batch_size=120):
    """
    Efficiently predict a list of sentences

    Args:
        model (Fac2Exp): trained model
        test_src (list): factorized polyinomals
        batch_size (int): batch size

    Return:
        preds (list): generated expansions
    """
    pred_tensors = [
            seq2seq.vocab.transform_tok2tensor(s)
            for s in test_src
        ]

    collate_fn = Collater(seq2seq.vocab, predict=True)
    test_dataloader = DataLoader(
        Fac2expDataset(pred_tensors),
        batch_size=batch_size,
        collate_fn=collate_fn,
    )

    preds = []
    for batch in tqdm.tqdm(test_dataloader, total=len(test_dataloader)):
        res = predict_batch(seq2seq, batch.to(DEVICE))
        p, _, _ = res
        preds.extend(p)
    return preds



def evaluate(seq2seq, testset, batch_size=100):
    """
    Run evaluation of trained model on testset

    Args:
        seq2seq (Seq2Seq): trained model
        testset (tuple): a list of (src, trg) pairs
        batch_size (int): size of batch

    Return:
        final_score: performance of seq2seq based on a given
                     evaluation func
    """
    src, trg = testset
    pred = predict(seq2seq, src, batch_size=batch_size)
    assert len(pred) == len(trg)

    total_score = 0
    for i, (s, t, p) in enumerate(zip(src, trg, pred)):
        pred_score = score(t, p)
        total_score += pred_score
        if i % 10 == 0:
            print(f"\n\n\n---- Example {i} ----")
            print(f"src = {s}")
            print(f"trg = {t}")
            print(f"prd = {p}")
            print(f"score = {pred_score}")

    final_score = total_score / len(pred)
    print(f"{total_score}/{len(pred)} = {final_score:.4f}")
    return final_score






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exper_dir", type=str, default="experiment/")
    parser.add_argument("-t", "--test_dir", type=str, default="data/test_split.txt")
    parser.add_argument("-v", "--vocab_dir", type=str, default="vocab.pickle")
    parser.add_argument("-c", "--checkpoint_dir", type=str, required=True)
    args = parser.parse_args()

    testset = load_file(args.test_dir)
    model = load_model(
        DEVICE,
        exper_dir=args.exper_dir,
        vocab_pickle=args.vocab_dir,
        model_ckpt=args.checkpoint_dir)
    evaluate(model.seq2seq, testset)
