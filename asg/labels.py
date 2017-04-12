# -*- coding: utf-8 -*-
"""Annotation Labels"""

import os
import json

from .datadirectory import data_directory


class Annotations:
    
    _json_subsets = ['test', 'train', 'val']
    _json_groups = [('dii', 'description-in-isolation'),
                    ('sis', 'story-in-sequence')]
    _labels = None

    _annotations_train = None
    _annotations_test = None

    @staticmethod
    def _label_data():
        if Annotations._labels is not None:
            return Annotations._labels

        Annotations._labels = {}
        for directory, label in Annotations._json_groups:
            Annotations._labels[directory] = {}
            for subset in Annotations._json_subsets:
                path = os.path.join(data_directory, directory,
                                    '{}.{}.json'.format(subset, label))
                with open(path) as data_file:
                    Annotations._labels[directory][subset] = json.load(
                        data_file)

        return Annotations._labels

    @staticmethod
    def _annotation_to_dict(label_data, subset, group):
        """
        Reduce an annotation to only the relevant data
        """
        annotations_ids = [a[0]["photo_flickr_id"]
                           for a in label_data[group][subset]['annotations']]
        annotations_texts = [a[0]["text"]
                             for a in label_data[group][subset]['annotations']]
        return dict(zip(annotations_ids, annotations_texts))

    @staticmethod
    def _annotations(label_data, subset):
        """
        Gather annotation into dictionary

        key - string - value that matches [image filename].jpg
        value - string of sanitized text description
        """

        annotations_dii = Annotations._annotation_to_dict(label_data, subset, "dii")
        annotations_sis = Annotations._annotation_to_dict(label_data, subset, "sis")
        return {**annotations_sis, **annotations_sis}

    @staticmethod
    def annotations_train():
        """Returns the training annotations dictionary"""
        if Annotations._annotations_train is None:
            Annotations._annotations_train = Annotations._annotations(Annotations._label_data(), 'train')
        return Annotations._annotations_train

    @staticmethod
    def annotations_test():
        """Returns the testing annotations dictionary"""
        if Annotations._annotations_test is None:
            Annotations._annotations_test = Annotations._annotations(Annotations._label_data(), 'test')
        return Annotations._annotations_test

