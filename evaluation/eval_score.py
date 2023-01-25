def score(true_expansion, pred_expansion):
    """
    Provided evaluation func for polynomial expansion
    Args:
        true_expansion (str): ground truth
        pred_expansion (str): predicted expansion 

    Retur:
        1: matched 0: not matched
    """
    return int(true_expansion == pred_expansion)
