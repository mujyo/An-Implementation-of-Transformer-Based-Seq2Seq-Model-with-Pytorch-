from trainer import get_device

# set device
DEVICE = get_device()

# Datasets
DATA_ROOT = "data/"
RAW_DATA_DIR = DATA_ROOT + "train.txt"
TRAIN_DIR = DATA_ROOT + "train_split.txt"
VALID_DIR = DATA_ROOT + "valid_split.txt"
TEST_DIR = DATA_ROOT + "test_split.txt"

# Build dataset
MAX_SEQUENCE_LENGTH = 29
TRAIN_URL = "https://scale-static-assets.s3-us-west-2.amazonaws.com/ml-interview/expand    /train.txt"

# experiment
EXPERIMENT_ROOT_DIR = "experiment"
VOCAB_PICKLE_DIR = "vocab.pickle"
CHECKPOINT_DIR = "lightning_logs/version_0/checkpoints/epoch=49-step=80000.ckpt"
BATCH_SIZE = 120
