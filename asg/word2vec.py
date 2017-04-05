import os

from gensim.models import Word2Vec
from gensim import models
from .datadirectory import data_directory


_mode_name = os.path.join(
    data_directory, 'GoogleNews-vectors-negative300.bin')
_embedding_model = models.KeyedVectors.load_word2vec_format(
    _mode_name, binary=True)


def vec(word):
    """The vector of a word"""
    return _embedding_model.word_vec(word)


def similarity(word_01, word_02):
    """Similarity between two words"""
    return _embedding_model.similarity(word_01, word_02)
