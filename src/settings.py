from pathlib import Path

DATA_PATH = Path('../data')
TIMIT_PATH = DATA_PATH / 'timit'
TRAIN_PATH = DATA_PATH / 'train'
TEST_PATH = DATA_PATH / 'test'
SAVED_MODEL_PATH = Path('saved_model')
BATCH_SIZE = 32
INPUT_SIZE = (299, 299)