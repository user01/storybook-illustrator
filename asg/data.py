import torch
import math
import os
import re
import time

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
    """Iterator to step though image and text pairs"""

    def __init__(self, images, texts):
        self._idx = -1
        self._images = images
        self._texts = texts

    def __iter__(self):
        return self

    def __next__(self):
        while self._idx < len(self._texts) - 1:
            self._idx += 1
            if self._texts[self._idx] != False:
                return (self._images[self._idx], self._texts[self._idx])
        raise StopIteration()


data_train = DataLoader(_image_folder, _text_values)
