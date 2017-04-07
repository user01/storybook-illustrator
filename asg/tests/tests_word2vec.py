"""Tests for word2vec"""

import unittest

from asg.word2vec import vec


class TestEmbeddings(unittest.TestCase):
    """Test the embedding tools. This requires the main data file and will take a long time."""

    def test_basic_embedding(self):
        """Read a common embedding"""
        result = vec("king")
        self.assertEqual(len(result), 300)
        self.assertAlmostEqual(result[0], 0.12597656, 5)


if __name__ == '__main__':
    unittest.main()
