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


# transforms.Compose([
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                       std=[0.229, 0.224, 0.225])
# ])


# def find_classes(dir):
#     classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
#     classes.sort()
#     class_to_idx = {classes[i]: i for i in range(len(classes))}
#     return classes, class_to_idx
#
# find_classes(os.path.join(data_directory,'train'))
#
# word2vec = Word2Vec()
#
# dd = {
#     "a": 1,
#     "b": 2
# }
# ll = [1,2,3,4,45,1]
#
# {ll[i]: i for i in range(len(ll))}
#
# group = 'train'
# image_path = os.path.join(data_directory, group)
#
# ids = [(d, _img_path_to_text(d, Annotations.annotations_train())) for d in \
#         os.listdir(image_path) if \
#         is_image_file(d) and \
#         _img_path_to_text(d, Annotations.annotations_train()) != False]
#
# ids[0:5]
#
# texts = [_img_path_to_text(d, Annotations.annotations_train()) for d in \
#         os.listdir(image_path)]
# texts[0:5]
# texts = [text for text in texts if text != False]
# texts[0:5]
# ss = [_sentence_to_tensor(text, word2vec) for text in texts if text != False]
#
# ff = [s for s in ss if torch.is_tensor(s)]
#
# len(ff)
#
# list(enumerate(texts))
#
# word2vec.tokenize(texts[92])


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

    # @staticmethod
    # def _path_leaf(path):
    #     """Returns filename"""
    #     # http://stackoverflow.com/a/8384788/2601448
    #     head, tail = os.path.split(path)
    #     return tail or os.path.basename(head)
    #
    # @staticmethod
    # def _img_path_to_text(path, annotations):
    #     filename = ImageLoader._path_leaf(path)
    #     results = ImageLoader._REGEXP_ID.match(filename)
    #     if results is None:
    #         return False
    #
    #     groups = results.groups()
    #     if len(groups) != 1:
    #         return False
    #
    #     return annotations[groups[0]] if groups[0] in annotations else False

# class DataLoader(object):
#     """
#     Iterator to step though image/text pairs with associated distance, sized
#     by the batch size
#
#     Each iteration returns a tuple of (torch_image, text, distance)
#
#     image is of form: `batch_size*color*width*height`
#
#     text is of form: `time_step_size*batch_size*hidden_size`
#
#     distance is of form: `batch_size*1`
#
#     This generates mismatched image/text pairs with the correct distance.
#     mismatched_passes indicates how many times an image will be mismatched with
#     a different text
#     """
#
#     def __init__(self,
#                  group,
#                  word2vec,
#                  mismatched_passes=3,
#                  max_tokens=15,
#                  seed=451):
#         self._idx = -1
#         self._max_tokens = max_tokens
#         self._word2vec = word2vec
#         self._mismatched_passes = mismatched_passes
#         self._seed = seed
#
#         self._images = DataLoader._image_folder(
#             os.path.join(data_directory, group, 'images'))
#
#         annotations = Annotations.annotations_train() if \
#             group == 'train' else Annotations.annotations_test()
#         self._texts = [DataLoader._img_path_to_text(path, annotations)
#                        for path, _ in self._images.imgs]
#
#         self._valid_texts = [
#             text for text in self._texts if self._valid_text(text)]
#
#     _NORMALIZE =
#     _REGEXP_ID = re.compile(r'^(.+)\.\D+$', re.IGNORECASE)
#
#     @staticmethod
#     def _image_folder(path):
#         """Image folder from a path"""
#         return datasets.ImageFolder(
#             path,
#
#             )
#
#     @staticmethod
#     def _path_leaf(path):
#         """
#         Returns filename
#
#         http://stackoverflow.com/a/8384788/2601448
#         """
#         head, tail = os.path.split(path)
#         return tail or os.path.basename(head)
#
#     @staticmethod
#     def _img_path_to_text(path, annotations):
#         filename = DataLoader._path_leaf(path)
#         results = DataLoader._REGEXP_ID.match(filename)
#         if results is None:
#             return False
#
#         groups = results.groups()
#         if len(groups) != 1:
#             return False
#
#         return annotations[groups[0]] if groups[0] in annotations else False
#
#     @staticmethod
#     def _vector_to_tensor(vec):
#         return torch.unsqueeze(torch.from_numpy(vec), 0)
#
#     @staticmethod
#     def _vectors_to_tensor(vectors):
#         return torch.stack(list(map(DataLoader._vector_to_tensor, vectors)))
#
#     def _sentence_to_tensor(self, sentence):
#         return DataLoader._vectors_to_tensor(self._word2vec.sentence_embedding(sentence))
#
#     def _valid_text(self, text):
#         if text is False:
#             return False
#         tokens = len(self._word2vec.tokenize(text))
#         return tokens > 1 and tokens < self._max_tokens
#
#     def _idx_length(self):
#         """Actual number of image/text pairs"""
#         return len(self._texts) * self._mismatched_passes
#
#     def __iter__(self):
#         return self
#
#     def __len__(self):
#         """Number of valid pairs, including mismatches"""
#         return len(self._valid_texts) * self._mismatched_passes
#
#     def __next__(self):
#         while self._idx < self._idx_length() - 1:
#             self._idx += 1
#             current_idx = self._idx % len(self._valid_texts)
#             if self._valid_text(self._texts[current_idx]):
#                 return self._current_image(current_idx)
#
#         raise StopIteration()
#
#     def _current_image(self, current_idx):
#         text_actual = self._texts[current_idx]
#         image_raw, _ = self._images[current_idx]
#         # resnet requires this shape of [1, 3, 224, 224]
#         image = torch.unsqueeze(image_raw, 0)
#
#         # one of the passes (0th), return the correct text with no distance
#         if (self._idx // len(self._valid_texts)) == current_idx % self._mismatched_passes:
#             return (image, self._sentence_to_tensor(text_actual), 1)
#
#         # mismatch the text
#         possible_texts = [
#             text for text in self._valid_texts if text != text_actual]
#         random.seed(self._idx + self._mismatched_passes + self._seed)
#         new_text = random.choice(possible_texts)
#         # distance = self._word2vec.word_mover_distance(text_actual, new_text)
#         return (image, self._sentence_to_tensor(new_text), -1)
