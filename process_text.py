# coding: utf-8

import os
import re
import json
import operator
import argparse
import shutil
from functools import reduce
import multiprocessing

import numpy as np
from jinja2 import Template
from scipy import spatial
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable

from asg.datadirectory import data_directory
from asg.model import Net
from asg.logger import Logger
from asg.word2vec import Word2Vec
from asg.data import sentence_to_tensor


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
    'image.embeddings.json',
    '--text',
    'a.bell.rings.txt',
    '--output',
    'output'
]))


if not os.path.isdir(opt.output):
    raise Exception("Output path is not a directory")

### Storyboard Selection
#### 1. Parse text corpus into sentences
#### 2. Create dummy image and sentence embeddings
#### 3. Get closest k = 5 dummy images for each dummy sentence
#### 4. Create final storyboard by selecting image closest to previous sentence's image (first sentence's image is closest image to sentence)

Logger.log("Init")

with open(opt.text, 'r') as f:
    raw_lines = f.read() \
                .split('\n\n')
raw_lines = list(map(lambda line: line.replace('\n', ' '), raw_lines))

Logger.log("Read Text")


with open(opt.embedding) as f:
    image_dict = json.load(f)

Logger.log("Read Image Embeddings")

word2vec = Word2Vec()
Logger.log("Read Word2Vec")

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

Logger.log("Sentence Creation")
raw_sentences = list(map(split_into_sentences, raw_lines))


Logger.log("Loading Network")
CUDA_AVAILABLE = torch.cuda.is_available()

torch.manual_seed(opt.seed)
if CUDA_AVAILABLE:
    torch.cuda.manual_seed(opt.seed)


def variable(target):
    """Convert tensor to a variable"""
    if CUDA_AVAILABLE:
        target = target.cuda()
    return Variable(target)


net = Net()
if CUDA_AVAILABLE:
    net = net.cuda()

net.load_state_dict(torch.load(opt.model))
net.eval()
# split_into_sentences(raw_text)


def text_size_to_variables(text_sizes):
    return [variable(torch.LongTensor([text_size])) for text_size in text_sizes]


image_static = variable(torch.randn(1,3,224,244))
def sentence_to_embedding(sentence):
    tensor = sentence_to_tensor(sentence, word2vec, 2, 15)
    if tensor[0] is False:
        return False
    text = variable(tensor[0].unsqueeze(0))
    text_sizes_var = text_size_to_variables([tensor[1]])
    _, output_text_var = net(image_static, text, text_sizes_var)
    sentence_embedding = output_text_var.data.cpu().numpy()
    return sentence_embedding


# get closest k images for each sentence
def top_images(sentence_embedding, image_dict, k=5):
    similarities = {}
    for key, value in image_dict.items():
        similarities[key] = 1 - spatial.distance.cosine(sentence_embedding, value)
    sorted_sim = sorted(similarities.items(), key=operator.itemgetter(1), reverse=True)
    top = []
    for i in range(k):
        top.append((sorted_sim[i][0], sorted_sim[i][1], image_dict[sorted_sim[i][0]]))
    return(top)


def top_images_simple(sentence_embedding, image_dict, k=5):
    return [(filename, score) for filename, _, score in top_images(sentence_embedding, image_dict, k)]


def sentences_top_images(sentences, image_dict, k=5):
    embeddings = map(sentence_to_embedding, sentences)
    embeddings_and_sentences = zip(embeddings, sentences)
    canidates = [(s, top_images(e, image_dict, k)) if e is not False else (s, []) for e, s in embeddings_and_sentences]
    return canidates

results = list(map(lambda sentences: sentences_top_images(sentences, image_dict), raw_sentences))


len(results)

len(results[2])
len(results[2][3])

# List<List<Tuple<Sentence,List<Tuple<filename, score>>>>>
# Results List<Paragraphs>
# Paragraph List<Line>
# Line Tuple(Text, List<Images>)
# Images List<Tuple(Filename, Score)>


html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <title>Story Output</title>
    <link href="https://fonts.googleapis.com/css?family=Libre+Baskerville" rel="stylesheet">
    <link rel="stylesheet" href="https://unpkg.com/purecss@0.6.2/build/pure-min.css" integrity="sha384-UQiGfs9ICog+LwheBSRCt1o5cbyKIHbwjWscjemyBMT9YCUMZffs6UqUTd0hObXD" crossorigin="anonymous">
    <style type="text/css">
    .canidate {
      border-radius: 0.1em;
      margin-left: 0.2em;
      margin-right: 0.2em;
      max-width: 100%;
      max-height: 100%;
    }
    .text {
        font-family: 'Libre Baskerville', serif;
    }
    .line:nth-child(even) {
        background-color: rgba(116, 123, 131, 0.13);
    }
    .paragraph:nth-child(even) {
        background-color: rgba(0, 191, 255, 0.07);
    }
    .score {
        text-align: center;
        font-family: monospace;
    }
    </style>
</head>
<body>
    <div id="all">
    {% for paragraph in paragraphs %}
        <div class="paragraph">
            {% for line in paragraph %}
            <div class="line">
                <p class="text">
                    {{ line[0] }}
                </p>
                <div class="pure-g">
                    {% for canidate in line[1] %}
                    <div class="pure-u-1-5">
                        <p class="score">{{ canidate[1]|round(2) }}</p>
                        <img class="canidate" src="assets/{{ canidate[0] }}" alt="{{ canidate[0] }}" />
                    </div>
                    {% endfor %}
                </div>
            </div>
            {% endfor %}
        </div>
    {% endfor %}
    </div>
</body>
</html>
"""


template = Template(html_template)
render = template.render(paragraphs=results)


with open(os.path.join(opt.output, "demo.html"), "w") as html_file:
    html_file.write(render)


files_referenced = [f for f, _ in sum([l for _, l in sum(results, [])], [])]
asset_path = os.path.join(opt.output,'assets')


if not os.path.isdir(asset_path):
    os.mkdir(asset_path)

source_directory = os.path.join(data_directory, 'images_full')

for filename in files_referenced:
    shutil.copy(os.path.join(source_directory, filename), asset_path)




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
