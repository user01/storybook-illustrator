# -*- coding: utf-8 -*-
"""Utilities to load a custom version of the VIST dataset"""


import os
import re
import random
from PIL import Image

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data

from .datadirectory import data_directory
from .labels import Annotations


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def _vector_to_tensor(vec):
    return torch.unsqueeze(torch.from_numpy(vec), 0)

def _vectors_to_tensor(vectors, tensor_size_min, tensor_size_max):
    tensors = list(map(_vector_to_tensor, vectors))
    if len(tensors) < tensor_size_min or len(tensors) > tensor_size_max:
        return False
    return torch.stack(tensors)


def _sentence_to_tensor(sentence, word2vec, tensor_size_min, tensor_size_max):
    text = _vectors_to_tensor(word2vec.sentence_embedding(
        sentence), tensor_size_min, tensor_size_max)
    if not torch.is_tensor(text):
        return False, 0
    print(text.size())
    if text.size()[0] >= tensor_size_max:
        return text, tensor_size_max
    text_size = text.size()[0]
    texts_padded = torch.cat([text,
                              torch.unsqueeze(
                                  torch.zeros(tensor_size_max - text_size, 300),
                                  1)
                              ])
    return texts_padded, text_size


_REGEXP_ID = re.compile(r'^(.+)\.\D+$', re.IGNORECASE)


def _img_path_to_text(filename, annotations):
    results = _REGEXP_ID.match(filename)
    if results is None:
        return False

    groups = results.groups()
    if len(groups) != 1:
        return False

    return annotations[groups[0]] if groups[0] in annotations else False


def make_dataset(dir, class_to_idx):
    images = []
    for target in os.listdir(dir):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageLoader(data.Dataset):

    def __init__(self,
                 group,
                 word2vec,
                 mismatched_passes=3,
                 max_tokens=15,
                 seed=451,
                 transform=None,
                 target_transform=None,
                 loader=default_loader):

        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

        self._mismatched_passes = mismatched_passes
        self._seed = seed

        self._image_path = os.path.join(data_directory, group)

        annotations = Annotations.annotations_train() if \
            group == 'train' else Annotations.annotations_test()
        texts = [(d, _img_path_to_text(d, annotations))
                 for d in os.listdir(self._image_path)]
        texts_clean = [(d, text) for d, text in texts if text != False]

        text_tensors = [(d, _sentence_to_tensor(text, word2vec, 1, max_tokens))
                        for d, text in texts_clean if text != False]
        text_tensors_clean = [(d, tensor, text_size) for d, (tensor, text_size) in text_tensors if
                              torch.is_tensor(tensor)]

        self._valid_values = text_tensors_clean

    def __getitem__(self, index):
        idx = index % len(self._valid_values)

        filename, text, text_size = self._valid_values[idx]
        path = os.path.join(self._image_path, filename)
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        # test if this is a mismatch pass
        real_pass = index // len(self._valid_values) == idx % self._mismatched_passes
        if not real_pass:
            random.seed(self._seed + index)
            idx_different = (random.randint(0,
                                            len(self._valid_values) - 1) + idx) % len(self._valid_values)
            _, text, text_size = self._valid_values[idx_different]

        target = torch.Tensor([[1 if real_pass else -1]])

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, text, text_size, target

    def __len__(self):
        return len(self._valid_values) * self._mismatched_passes
