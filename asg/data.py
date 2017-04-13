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


class DataLoader(object):
    """
    Iterator to step though image/text pairs with associated distance, sized
    by the batch size

    Each iteration returns a tuple of (torch_image, text, distance)

    image is of form: `batch_size*color*width*height`

    text is of form: `time_step_size*batch_size*hidden_size`

    distance is of form: `batch_size*1`

    This generates mismatched image/text pairs with the correct distance.
    mismatched_passes indicates how many times an image will be mismatched with
    a different text
    """

    def __init__(self,
                 group,
                 word2vec,
                 mismatched_passes=3,
                 max_tokens=15,
                 seed=451):
        self._idx = -1
        self._max_tokens = max_tokens
        self._word2vec = word2vec
        self._mismatched_passes = mismatched_passes
        self._seed = seed

        self._images = DataLoader._image_folder(
            os.path.join(data_directory, group, 'images'))

        annotations = Annotations.annotations_train() if \
            group == 'train' else Annotations.annotations_test()
        self._texts = [DataLoader._img_path_to_text(path, annotations)
                       for path, _ in self._images.imgs]

        self._valid_texts = [
            text for text in self._texts if self._valid_text(text)]

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
                DataLoader._NORMALIZE
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
        filename = DataLoader._path_leaf(path)
        results = DataLoader._REGEXP_ID.match(filename)
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
        return torch.stack(list(map(DataLoader._vector_to_tensor, vectors)))

    def _sentence_to_tensor(self, sentence):
        return DataLoader._vectors_to_tensor(self._word2vec.sentence_embedding(sentence))

    def _valid_text(self, text):
        if text is False:
            return False
        tokens = len(self._word2vec.tokenize(text))
        return tokens > 1 and tokens < self._max_tokens

    def _actual_length(self):
        """Actual number of image/text pairs"""
        return len(self._valid_texts) * self._mismatched_passes

    def __iter__(self):
        return self

    def __len__(self):
        """Number of valid pairs, including mismatches"""
        return self._actual_length()

    def __next__(self):
        while self._idx < self._actual_length() - 1:
            self._idx += 1
            current_idx = self._idx % len(self._valid_texts)
            if self._valid_text(self._texts[current_idx]):
                return self._current_image(current_idx)

        raise StopIteration()

    def _current_image(self, current_idx):
        text_actual = self._texts[current_idx]
        image_raw, _ = self._images[current_idx]

        # resnet requires this shape of [1, 3, 224, 224]
        image = torch.unsqueeze(image_raw, 0)

        # one of the passes (0th), return the correct text with no distance
        if (self._idx // len(self._valid_texts)) == 0:
            return (image, self._sentence_to_tensor(text_actual), 0)

        # mismatch the text
        possible_texts = [
            text for text in self._valid_texts if text != text_actual]
        random.seed(self._idx + self._mismatched_passes + self._seed)
        new_text = random.choice(possible_texts)
        distance = self._word2vec.word_mover_distance(text_actual, new_text)
        return (image, self._sentence_to_tensor(new_text), distance)

