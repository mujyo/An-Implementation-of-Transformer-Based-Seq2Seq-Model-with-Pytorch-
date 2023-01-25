import os
import pickle

from network import Fac2Exp


def load_model(device, exper_dir, vocab_pickle="vocab.pickle", 
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

    model = Fac2Exp.load_from_checkpoint(
        os.path.join(exper_dir, model_ckpt),
        device=device,
        vocab=vocab).to(device)
    return model
