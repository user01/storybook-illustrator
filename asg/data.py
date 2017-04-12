# -*- coding: utf-8 -*-
"""Utilities to load a custom version of the VIST dataset"""


import os
import re
import random

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from .datadirectory import data_directory
from .labels import Annotations
# from .word2vec import word_mover_distance, sentence_embedding, tokenize


class Loader(object):
    """Load image data"""

    def __init__(self, word2vec):
        self._word2vec = word2vec

        image_folder_train = Loader._image_folder(
            os.path.join(data_directory, 'train', 'images'))
        image_folder_test = Loader._image_folder(
            os.path.join(data_directory, 'test', 'images'))

        a_train = Annotations.annotations_train()
        a_test = Annotations.annotations_test()
        text_values_train = [Loader._img_path_to_text(path, a_train)
                             for path, _ in image_folder_train.imgs]
        text_values_test = [Loader._img_path_to_text(path, a_test)
                            for path, _ in image_folder_test.imgs]

        self.data_train = DataLoader(
            image_folder_train, text_values_train,
            word2vec, self._sentence_to_tensor,
            3, 451)
        self.data_test = DataLoader(
            image_folder_test, text_values_test,
            word2vec, self._sentence_to_tensor,
            3, 452)

    _NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
    _REGEXP_ID = re.compile(r'^(.+)\.\D+$', re.IGNORECASE)

    @staticmethod
    def _image_folder(path):
        """Image folder from a path"""
        return datasets.ImageFolder(
            path,
            transforms.Compose([
                transforms.RandomSizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                Loader._NORMALIZE
            ]))

    @staticmethod
    def _path_leaf(path):
        """
        Returns filename

        http://stackoverflow.com/a/8384788/2601448
        """
        head, tail = os.path.split(path)
        return tail or os.path.basename(head)

    @staticmethod
    def _img_path_to_text(path, annotations):
        filename = Loader._path_leaf(path)
        results = Loader._REGEXP_ID.match(filename)
        if results is None:
            return False

        groups = results.groups()
        if len(groups) != 1:
            return False

        return annotations[groups[0]] if groups[0] in annotations else False

    @staticmethod
    def _vector_to_tensor(vec):
        return torch.unsqueeze(torch.from_numpy(vec), 0)

    @staticmethod
    def _vectors_to_tensor(vectors):
        return torch.stack(list(map(Loader._vector_to_tensor, vectors)))

    def _sentence_to_tensor(self, sentence):
        return Loader._vectors_to_tensor(self._word2vec.sentence_embedding(sentence))

    def train(self):
        """Return the training data DataLoader"""
        return self.data_train

    def test(self):
        """Return the testing data DataLoader"""
        return self.data_test


class DataLoader(object):
    """
    Iterator to step though image/text pairs with associated distance

    Each iteration returns a tuple of (torch_image, text, distance)

    This generates mismatched image/text pairs with the correct distance.
    mismatched_passes indicates how many times an image will be mismatched with
    a different text
    """

    def __init__(self, images, texts, word2vec, sentence_to_tensor, mismatched_passes=3, seed=451):
        self._idx = -1
        self._images = images
        self._texts = texts
        self._word2vec = word2vec
        self._sentence_to_tensor = sentence_to_tensor

        self._valid_texts = [
            text for text in texts if self._valid_text(text)]
        self._pass = 0
        self._mismatched_passes = mismatched_passes
        self._seed = seed

    def _valid_text(self, text):
        if text is False:
            return False
        tokens = len(self._word2vec.tokenize(text))
        return tokens > 1 and tokens < 20

    def __iter__(self):
        return self

    def __len__(self):
        return len(self._valid_texts) * self._mismatched_passes

    def __next__(self):

        while self._pass < self._mismatched_passes:
            while self._idx < len(self._texts) - 1:
                self._idx += 1
                if self._valid_text(self._texts[self._idx]):
                    return self._current_image()
            self._pass += 1
            self._idx = -1

        raise StopIteration()

    def _current_image(self):
        text_actual = self._texts[self._idx]
        image_raw, _ = self._images[self._idx]

        # resnet requires this shape of [1, 3, 224, 224]
        image = torch.unsqueeze(image_raw, 0)

        # one of the passes, return the correct with no distance
        if (self._idx + self._mismatched_passes) % self._mismatched_passes == 0:
            return (image, self._sentence_to_tensor(text_actual), 0)

        # mismatch the text
        possible_texts = [
            text for text in self._valid_texts if text != text_actual]
        random.seed(self._idx + self._mismatched_passes + self._seed)
        new_text = random.choice(possible_texts)
        distance = self._word2vec.word_mover_distance(text_actual, new_text)
        return (image, self._sentence_to_tensor(new_text), distance)
