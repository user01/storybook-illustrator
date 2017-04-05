
import os
import json

from .datadirectory import DATA_DIRECTORY


_json_subsets = ['test', 'train', 'val']
_json_groups = [('dii', 'description-in-isolation'),
                ('sis', 'story-in-sequence')]

_label_data = {}
for directory, label in _json_groups:
    _label_data[directory] = {}
    for subset in _json_subsets:
        path = os.path.join(DATA_DIRECTORY, directory,
                            '{}.{}.json'.format(subset, label))
        with open(path) as data_file:
            _label_data[directory][subset] = json.load(data_file)


def _modify_annotation(annotation, label_data, group, subset):
    """
    Reduce an annotation to only the relevant data
    """
    album = list(filter(lambda a: a["id"] == annotation[
                 'album_id'], label_data[group][subset]['albums']))[0]
    return {
        'album_id': annotation['album_id'],
        'album_title': album['title'],
        'album_description': album['description'],
        'original_text': annotation['original_text'],
        'text': annotation['text'],
        'photo_order_in_story': annotation['photo_order_in_story']
    }


def _annotations(label_data, group, subset):
    """
    Gather annotation into dictionary

    key - int - value that matches [image filename].jpg
    value - dictionary of annotation data. Of form:
    {
    'album_id': '481598',
    'original_text': 'a shiny black car resting on the pavement in front of a crowd of people',
    'photo_flickr_id': '20643962',
    'photo_order_in_story': 4,
    'text': 'a shiny black car resting on the pavement in front of a crowd of people',
    'tier': 'descriptions-in-isolation',
    'worker_id': 'K9458ZXUSNHEL4D'
    }
    """
    annotations_single = map(lambda a: a[0], label_data[
                             group][subset]['annotations'])
    # annotations = map(lambda a: _modify_annotation(a, label_data, group, subset), annotations_single)
    annotation_ids = map(lambda a: int(
        a['photo_flickr_id']), annotations_single)

    return dict(zip(annotation_ids, annotations_single))
    # return dict(zip(annotation_ids, annotations))


annotations_train = _annotations(_label_data, 'dii', 'train')
annotations_test = _annotations(_label_data, 'dii', 'test')
