import os

from gensim.models import Word2Vec
from gensim import models

print(os.path.dirname(os.path.realpath(__file__)))

with open("data.directory.txt") as f:
    # with open(os.path.join("..", "data.directory.txt")) as f:
    DATA_DIRECTORY = f.read().strip()

if not os.path.isdir(DATA_DIRECTORY):
    raise Exception(
        "Data directory: {} does not exist.".format(DATA_DIRECTORY))


_MODEL_NAME = os.path.join(
    DATA_DIRECTORY, 'GoogleNews-vectors-negative300.bin')
_EMBEDDING_MODEL = models.KeyedVectors.load_word2vec_format(
    _MODEL_NAME, binary=True)


def vec(word):
    """The vector of a word"""
    return _EMBEDDING_MODEL.word_vec(word)


def similarity(word_01, word_02):
    """Similarity between two words"""
    return _EMBEDDING_MODEL.similarity(word_01, word_02)
