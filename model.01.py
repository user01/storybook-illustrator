
# coding: utf-8

# # Automated Storyboard Generation
# 
# #### Model 01

# In[1]:

import os
import json
import sys
import numpy as np

from __future__ import print_function
import torch

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[34]:

with open("vist.directory.txt") as file:
    VIST_DIRECTORY = file.read().strip()

print("VIST data directory: {}".format(VIST_DIRECTORY))
if not os.path.isdir(VIST_DIRECTORY):
    raise Exception("VIST Data Directory does not exist.")


# In[3]:

JSON_SUBSETS = ['test', 'train', 'val']
JSON_GROUPS = [('dii', 'description-in-isolation'), ('sis', 'story-in-sequence')]

LABEL_DATA = {}
for directory, label in JSON_GROUPS:
    LABEL_DATA[directory] = {}
    for subset in JSON_SUBSETS:
        path = os.path.join(VIST_DIRECTORY, directory, '{}.{}.json'.format(subset, label))
        with open(path) as data_file:
            LABEL_DATA[directory][subset] = json.load(data_file)


# In[4]:

test_img_id = LABEL_DATA['dii']['train']['images'][0]['id']
test_img_path = os.path.join(VIST_DIRECTORY, 'train', '{}.jpg'.format(test_img_id))
test_img = image.load_img(test_img_path, target_size=(224, 224))
plt.imshow(np.asarray(test_img))


# In[5]:

LABEL_DATA['dii']['train']['images'][:2]


# In[ ]:



