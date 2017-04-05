
import os
import json

from .datadirectory import data_directory


_json_subsets = ['test', 'train', 'val']
_json_groups = [('dii', 'description-in-isolation'),
                ('sis', 'story-in-sequence')]

_label_data = {}
for directory, label in _json_groups:
    _label_data[directory] = {}
    for subset in _json_subsets:
        path = os.path.join(data_directory, directory,
                            '{}.{}.json'.format(subset, label))
        with open(path) as data_file:
            _label_data[directory][subset] = json.load(data_file)


def _annotation_to_dict(label_data, subset, group):
    """
    Reduce an annotation to only the relevant data
    """
    annotations_ids = [a[0]["photo_flickr_id"] for a in label_data[group][subset]['annotations']]
    annotations_texts = [a[0]["text"] for a in label_data[group][subset]['annotations']]
    return dict(zip(annotations_ids, annotations_texts))


def _annotations(label_data, subset):
    """
    Gather annotation into dictionary

    key - string - value that matches [image filename].jpg
    value - string of santanized text description
    """

    annotations_dii = _annotation_to_dict(label_data, subset, "dii")
    annotations_sis = _annotation_to_dict(label_data, subset, "sis")
    return {**annotations_sis, **annotations_sis}


annotations_train = _annotations(_label_data, 'train')
annotations_test = _annotations(_label_data, 'test')
