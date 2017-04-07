import torch
import math
import os
import re
import time
import random


import numpy as np

from torch.autograd import Variable
from torch.utils.data import TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from .datadirectory import data_directory
from .labels import annotations_train
from .word2vec import word_mover_distance, sentence_embedding

_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])

_image_folder = datasets.ImageFolder(
    os.path.join(data_directory, 'images'),
    transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        _normalize
    ]))


_id_regexp = re.compile('^(.+)\.\D+$', re.IGNORECASE)


def _path_leaf(path):
    """
    Returns filename

    http://stackoverflow.com/a/8384788/2601448
    """
    head, tail = os.path.split(path)
    return tail or os.path.basename(head)


def _img_path_to_text(path):
    filename = _path_leaf(path)
    results = _id_regexp.match(filename)
    if results is None:
        return False

    groups = results.groups()
    if len(groups) != 1:
        return False

    return annotations_train[groups[0]] if groups[0] in annotations_train else False


_text_values = [_img_path_to_text(path)
                for path, _ in _image_folder.imgs]


class DataLoader:
    """
    Iterator to step though image/text pairs with associated distance

    Each iteration returns a tuple of (torch_image, text, distance)

    This generates mismatched image/text pairs with the correct distance.
    mismatched_passes indicates how many times an image will be mismatched with
    a different text
    """

    def __init__(self, images, texts, mismatched_passes=3, seed=451):
        self._idx = -1
        self._images = images
        self._texts = texts
        self._valid_texts = [text for text in texts if text != False]
        self._pass = 0
        self._mismatched_passes = mismatched_passes

    def __iter__(self):
        return self

    def __next__(self):

        while self._pass < self._mismatched_passes:
            while self._idx < len(self._texts) - 1:
                self._idx += 1
                if self._texts[self._idx] != False:
                    return self._current_image()
            self._pass += 1
            self._idx = -1

        raise StopIteration()

    def _current_image(self):
        actual_text = self._texts[self._idx]
        # one of the passes, return the correct with no distance
        if (self._idx + self._mismatched_passes) % self._mismatched_passes == 0:
            return (self._images[self._idx], sentence_embedding(actual_text), 0)

        # mismatch the text
        possible_texts = [
            text for text in self._valid_texts if text != actual_text]
        random.seed(self._idx + self._mismatched_passes)
        new_text = random.choice(possible_texts)
        distance = word_mover_distance(actual_text, new_text)
        return (self._images[self._idx], sentence_embedding(new_text), distance)


data_train = DataLoader(_image_folder, _text_values)
