import sys
import numpy as np
from typing import Tuple
from tqdm import tqdm

from CONST import EXPERIMENT_ROOT_DIR, VOCAB_PICKLE_DIR, CHECKPOINT_DIR, BATCH_SIZE, DEVICE
from trainer import get_device
import run_evaluate
from evaluation import load_model
from corpus import VocabFacExpan



def load_file(file_path: str) -> Tuple[Tuple[str], Tuple[str]]:
    """ A helper functions that loads the file into a tuple of strings

    :param file_path: path to the data file
    :return factors: (LHS) inputs to the model
            expansions: (RHS) group truth
    """
    data = open(file_path, "r").readlines()
    factors, expansions = zip(*[line.strip().split("=") for line in data])
    return factors, expansions


def score(true_expansion: str, pred_expansion: str) -> int:
    """ the scoring function - this is how the model will be evaluated

    :param true_expansion: group truth string
    :param pred_expansion: predicted string
    :return:
    """
    return int(true_expansion == pred_expansion)


# --------- START OF IMPLEMENT THIS --------- #
def predict(factors: str, seq2seq):
    """
    Perform batch prediction on factors

    Args:
        factors (tuple): input sequences 
        seq2seq (Seq2Seq): a trained sequence to sequence model

    Return:
        preds: expansions for each factor
    """
    try:
        preds = run_evaluate.predict(seq2seq,  factors, batch_size=BATCH_SIZE)
    except:
        preds = ""
    
    #return factors
    return preds
# --------- END OF IMPLEMENT THIS --------- #


def main(filepath: str):
    """
    Driven function for running prediction
    
    Args:
        filepath: directory to the file containing input sentences
    """
    factors, expansions = load_file(filepath)

    # Load checkpoint
    model = load_model(
        DEVICE,
        exper_dir=EXPERIMENT_ROOT_DIR,
        vocab_pickle= VOCAB_PICKLE_DIR,
        model_ckpt=CHECKPOINT_DIR)

    # Change to batch prediction in order to accelerate the process
    preds = predict(factors, model.seq2seq)

    scores = [score(te, pe) for te, pe in zip(expansions, preds)]
    print(np.mean(scores))


if __name__ == "__main__":
    main("test.txt" if "-t" in sys.argv else "train.txt")
    
    # tested with 100 samples, it works well
    #main("data/test_split_100.txt")
