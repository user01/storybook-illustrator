"""Tests for word2vec"""

import unittest

from asg.word2vec import Word2Vec


class TestBasic(unittest.TestCase):
    """Test the embedding tools. This requires the main data file and will take a long time."""
    _word2vec = Word2Vec()

    def test_basic_embedding(self):
        """Read a common embedding"""
        result = TestBasic._word2vec.vec("king")
        self.assertEqual(len(result), 300)
        self.assertAlmostEqual(result[0], 0.12597656, 5)

    def test_basic_similarity(self):
        """Test similarity between words"""
        result = TestBasic._word2vec.similarity("king", "queen")
        self.assertAlmostEqual(result, 0.651095683538665, 5)

    def test_basic_tokenize(self):
        """Test tokenize of sentence"""
        result = TestBasic._word2vec.tokenize(
            "The quick brown fox jumped over the lazy dog?")
        self.assertListEqual(
            result, ['quick', 'brown', 'fox', 'jumped', 'lazy', 'dog'])

    def test_basic_wmd(self):
        """Test word move distance"""
        result = TestBasic._word2vec.word_mover_distance(
            "Obama speaks to the media in Illinois.",
            "The President greets the press in Chicago.")
        self.assertAlmostEqual(result, 0.042881972611898585, 5)


if __name__ == '__main__':
    unittest.main()
