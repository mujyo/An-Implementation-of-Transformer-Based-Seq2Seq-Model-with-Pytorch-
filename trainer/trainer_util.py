import os
import pickle
import random

import torch
import numpy as np


def set_seed(seed):
    """
    Set seed for random number generation
    to achieve reproductivity

    Args:
        seed (int): the id of seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)



def get_device():
    """
    Get the current device 

    Return: "cuda" if GPU is available, otherwise "cpu" 
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
