def load_file(file_path):
    """ A helper functions that loads the file into a tuple of strings
    :param file_path: path to the data file
    :return factors: (LHS) inputs to the model
            expansions: (RHS) group truth
    """
    data = open(file_path, "r").readlines()
    factors, expansions = zip(*[line.strip().split("=") for line in data])
    return factors, expansions


def read_txt(file_path):
    """
    Read a .txt file
    Args:
        file_path: directory of file

    Return:
        list of sentences stripped of leading and trailing space
    """
    with open(file_path) as f:
        lines = f.readlines()
    return [line.strip() for line in lines]


def write_txt(file_path, content):
    """
    Write lines of sentences into .txt file
    Args:
        file_path: directory of file
    """
    with open(file_path, 'w') as f:
        for line in content:
            f.write("%s\n" % line)
