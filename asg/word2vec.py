# -*- coding: utf-8 -*-
"""Word2Vec Utilities"""

import os
import re

import numpy as np
# from gensim.models import Word2Vec as word2vec
from gensim import models
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

from .datadirectory import data_directory


class Word2Vec(object):
    """Word2Vect Utility class"""

    def __init__(self):

        self._tokenizer = RegexpTokenizer(r'\w+')
        self._stopset = set(stopwords.words('english'))

        self._mode_name = os.path.join(
            data_directory, 'GoogleNews-vectors-negative300.bin')
        self._embedding_model = models.KeyedVectors.load_word2vec_format(
            self._mode_name, binary=True)
        self._punct_regexp = re.compile(r"^\W+$", re.IGNORECASE)

    def vec(self, word):
        """The vector of a word"""
        try:
            return self._embedding_model.word_vec(word)
        except KeyError:
            # word not found
            # TODO: Handle this invalid word properly
            return np.zeros((300,), dtype=np.float32)

    def similarity(self, word_01, word_02):
        """Similarity between two words"""
        try:
            return self._embedding_model.similarity(word_01, word_02)
        except KeyError:
            # a word not found - defaults to distance of 1
            # TODO: Handle this invalid word properly
            return 1

    def tokenize(self, text):
        """Reduce a string to non-stopword tokens"""
        return [word for word in self._tokenizer.tokenize(text.lower())
                if word not in self._stopset and
                self._punct_regexp.match(word) is None]

    def sentence_embedding(self, text):
        """Convert a sentence to text embedding"""
        return [self.vec(word) for word in self.tokenize(text)]

    def _find_min_distance(self, word_source, tokens_target):
        """Find the minimum similarity between source word and all tokens"""
        return min([self.similarity(word_target, word_source) for
                    word_target in tokens_target])

    def word_mover_distance(self, text_source, text_target):
        """
        Measures the distance from the source text to the target text

        Minimum cumulative distance that all words in document 1 need to
        travel to exactly match document 2.

        http://jmlr.org/proceedings/papers/v37/kusnerb15.pdf
        """
        tokens_source = self.tokenize(text_source)
        tokens_target = self.tokenize(text_target)

        scores = [self._find_min_distance(token, tokens_target)
                  for token in tokens_source]

        return sum(scores) / len(tokens_source)
