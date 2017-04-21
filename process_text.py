# coding: utf-8

import re
import operator
import argparse
from functools import reduce
import multiprocessing

import numpy as np
from scipy import spatial
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable

from asg.datadirectory import data_directory
from asg.model import Net
from asg.logger import Logger


parser = argparse.ArgumentParser(
    description='Automated Storyboard Generator')

parser.add_argument('--model', type=str,
                    help='Path to model')
parser.add_argument('--embedding', type=str,
                    help='Path to image embedding')
parser.add_argument('--text', type=str,
                    help='Path to input text')
parser.add_argument('--output', type=str,
                    help='Path to output directory')

parser.add_argument('--seed', type=int, default=451,
                    help='Random seed. Default=451')
parser.add_argument('--batch', type=int, default=32,
                    help='Batch size. Default=32')
parser.add_argument('--workers', type=int, default=multiprocessing.cpu_count(),
                    help='Number of workers for data loader. Defaults to system cores')

# opt = parser.parse_args()
opt = parser.parse_args(([
    '--model',
    'models/model_a532783_epoch_0000018.pth',
    '--embedding',
    'models/model_a532783_epoch_0000018.pth',
    '--text',
    'a.bell.rings.txt'
]))

### Storyboard Selection
#### 1. Parse text corpus into sentences
#### 2. Create dummy image and sentence embeddings
#### 3. Get closest k = 5 dummy images for each dummy sentence
#### 4. Create final storyboard by selecting image closest to previous sentence's image (first sentence's image is closest image to sentence)

Logger.log("Init")

with open(opt.text, 'r') as f:
    raw_lines = f.read() \
                .split('\n\n')

raw_lines = list(map(lambda line: line.replace('\n', ' '), raw_texts))

Logger.log("Read Text")


with open('temp.json') as f:
    image_dict = json.load(f)

Logger.log("Read Image Embeddings")

# parse corpus into sentences

# http://stackoverflow.com/questions/4576077/python-split-text-on-sentences
caps = "([A-Z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"

def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + caps + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(caps + "[.]" + caps + "[.]" + caps + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(caps + "[.]" + caps + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + caps + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip(" .") for s in sentences]
    return sentences



# split_into_sentences(raw_text)

map(split_into_sentences, raw_lines)
list(_)



# get closest k images for each sentence
def top_images(sentence_embedding, image_dict, k=5):
    similarities = {}
    for key, value in image_dict.items():
        similarities[key] = 1 - spatial.distance.cosine(sentence_embedding, value)
    sorted_sim = sorted(similarities.items(), key=operator.itemgetter(1), reverse=True)
    top = []
    for i in range(k):
        top.append((sorted_sim[i][0], image_dict[sorted_sim[i][0]]))
    return(top)


# In[187]:

# k nearest images
k = 5


# In[194]:

# k image names and 300d embedding arrays for each sentence (10 sentences in this example)
image_pics = map(lambda x: top_images(x, image_dict, k), sentence_embeddings)


# In[195]:

# all storyboard candidates (k=5 for each sentence)
candidates = list(image_pics)


# In[190]:

# contains final storyboard images
storyboard = []

for i in range(len(candidates)):

    if i == 0:
        storyboard.append((candidates[0][0][0], candidates[0][0][1])) # closest image to first sentence
        continue

    temp = []

    for j in range(0,k):
        temp.append(1 - spatial.distance.cosine(candidates[i][j][1], storyboard[i-1][1]))

    idx = temp.index(max(temp)) # closest image to top image for previous sentence
    storyboard.append((candidates[i][idx][0], candidates[i][idx][1]))  # add to final storyboard array


# In[191]:

# Top image for each sentence (10 images for 10 sentences)
print(storyboard)
