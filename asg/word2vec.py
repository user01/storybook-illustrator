import os
import re

import numpy as np
from gensim.models import Word2Vec
from gensim import models
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

from .datadirectory import data_directory


_tokenizer = RegexpTokenizer(r'\w+')
_stopset = set(stopwords.words('english'))


_mode_name = os.path.join(
    data_directory, 'GoogleNews-vectors-negative300.bin')
_embedding_model = models.KeyedVectors.load_word2vec_format(
    _mode_name, binary=True)


def vec(word):
    """The vector of a word"""
    try:
        return _embedding_model.word_vec(word)
    except KeyError:
        # word not found
        # TODO: Handle this invalid word properly
        return np.zeros((300,), dtype=np.float32)


def similarity(word_01, word_02):
    """Similarity between two words"""
    try:
        return _embedding_model.similarity(word_01, word_02)
    except KeyError:
        # a word not found
        # TODO: Handle this invalid word properly
        return 1


_punct_regexp = re.compile(r"^\W+$", re.IGNORECASE)


def tokenize(text):
    """Reduce a string to non-stopword tokens"""
    return [word for word in _tokenizer.tokenize(text.lower())
            if word not in _stopset and _punct_regexp.match(word) is None]


def sentence_embedding(text):
    """Convert a sentence to text embedding"""
    return [vec(word) for word in tokenize(text)]


def _find_min_distance(word_source, tokens_target):
    """Find the minimum similarity between source word and all tokens"""
    return min([similarity(word_target, word_source) for word_target in tokens_target])


def word_mover_distance(text_source, text_target):
    """
    Measures the distance from the source text to the target text

    Minimum cumulative distance that all words in document 1 need to travel to exactly match document 2.

    http://jmlr.org/proceedings/papers/v37/kusnerb15.pdf
    """
    tokens_source = tokenize(text_source)
    tokens_target = tokenize(text_target)

    scores = [_find_min_distance(token, tokens_target)
              for token in tokens_source]

    return sum(scores) / len(tokens_source)
