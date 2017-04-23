# coding: utf-8

import os
import re
import json
import operator
import argparse
import shutil
from subprocess import call
import multiprocessing
from functools import reduce

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
    'awakening.txt',
    '--output',
    'output'
]))


if not os.path.isdir(opt.output):
    raise Exception("Output path is not a directory")

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

def cosine_distance(embedding_01, embedding_02):
    return 1 - spatial.distance.cosine(embedding_01, embedding_02)

# get closest k images for each sentence
def top_images(sentence_embedding, image_dict, k=5):
    similarities = {}
    for key, value in image_dict.items():
        similarities[key] = cosine_distance(sentence_embedding, value)
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
    candidates = [(s, top_images(e, image_dict, k)) if e is not False else (s, []) for e, s in embeddings_and_sentences]
    return candidates

def top_images_results(paragraphs_sentences):
    return list(map(lambda sentences: sentences_top_images(sentences, image_dict), paragraphs_sentences))

results = top_images_results(raw_sentences)

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
                    {{loop.index0}} {{ line[0] }}
                </p>
                <div class="pure-g">
                    {% for canidate in line[1] %}
                    <div class="pure-u-1-5">
                        <p class="score">{{ canidate[1]|round(2) }}</p>
                        <img class="canidate" src="assets/{{ prefix }}{{ canidate[0] }}" alt="{{ canidate[0] }}" />
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

def render_template(paragraphs, prefix, filename):
    render = template.render(paragraphs=paragraphs, prefix=prefix)

    with open(os.path.join(opt.output, "{}.html".format(filename)), "w") as html_file:
        html_file.write(render)

render_template(results, '', 'demo')

# Write assets to output
files_referenced = [f for f, _, _ in sum([l for _, l in sum(results, [])], [])]

def ensure_images(files_referenced, ensure_primitive=False):
    asset_path = os.path.join(opt.output,'assets')
    if not os.path.isdir(asset_path):
        os.mkdir(asset_path)

    source_directory = os.path.join(data_directory, 'images_full')

    for idx, filename in enumerate(files_referenced):
        source_path = os.path.join(source_directory, filename)
        target_path = os.path.join(asset_path, filename)
        primitive_path = os.path.join(asset_path, "prim.{}".format(filename))
        if not os.path.isfile(target_path):
            shutil.copy(source_path, asset_path)
        if not os.path.isfile(primitive_path) and ensure_primitive:
            Logger.log("Running Primitive {:0>3}/{:0>3} {:.1f} {}".format(idx,
                                                                          len(files_referenced),
                                                                          100 * idx / len(files_referenced),
                                                                          filename))
            response_code = call(["primitive", "-n", "500", "-i", target_path, "-o", primitive_path])

ensure_images(files_referenced)

Logger.log("Demo Written")

def paragraph_to_images(paragraph, topN=10, threshold=0.85):
    image_tuples = sum([images for _, images in paragraph], [])
    image_tuples_sorted = sorted(image_tuples, key=operator.itemgetter(1), reverse=True)[:topN]
    return [t for t in image_tuples_sorted if t[1] >= threshold]


def image_score(image_embedding, past_images, current_position, gamma=0.9):
    values = [gamma ** (abs(current_position - past_position) / 250) \
                * spatial.distance.cosine(image_embedding, past_embedding)
                for _, past_embedding, past_position in past_images]
    penalty = sum(values) / len(past_images) if len(past_images) > 0 else 0
    print('penalty', penalty)
    return penalty

MAX_IMAGE_HISTORY = 5
def paragraph_reduce(acc, paragraph_and_length, score_threshold):
    running_images, pressure, position = acc
    paragraph, length = paragraph_and_length
    candidates = paragraph_to_images(paragraph)
    if len(candidates) < 1:
        return running_images + [(False, False, False)], pressure, position + length

    past_images = [img for img in running_images if img[0] is not False][-MAX_IMAGE_HISTORY:]
    candidates_scored = [(filename, score - image_score(embedding, \
                            past_images, position), embedding) for \
                            filename, score, embedding in candidates]
    filename_top, score_top, embedding_top = sorted(candidates_scored, \
                            key=operator.itemgetter(1), \
                            reverse=True)[0]

    new_image = filename_top, embedding_top, position
    new_images = running_images + [new_image]

    if len(past_images) > 0:
        threshold = score_threshold(position - past_images[-1][2])
        print("past_images", past_images[-1][2], type(past_images[-1][2]))
        print("position", position, type(position))
        print("score_top", score_top, type(score_top))
        print("threshold", threshold, type(threshold))
    # threshold = score_threshold(position - running_images[-1][2]) if len(running_images) > 1 else 0
        print("{:.4f} vs {:.4f}".format(float(score_top), float(threshold)))
        if score_top > threshold:
            print("PICKED")
            return new_images, pressure, position + length
        else:
            print("IGNORED")
            return running_images + [(False, False, False)], pressure, position + length
    else:
        print("IGNORED FORCED")
    return new_images, pressure, position + length



def choose_images(top_images_results, raw_lines, distance=500):
    lengths = [len(l) for l in raw_lines]
    top_images_results_flatten = sum(top_images_results, [])
    image_candidates = sum([lines for _, lines in top_images_results_flatten], [])
    image_scores = [score for _, score, _ in image_candidates]
    score_max = max(image_scores)
    score_average = sum(image_scores) / len(image_scores)
    score_threshold = lambda location: -((score_max - score_average) / distance) * location + score_max

    results = reduce(lambda acc, elm: paragraph_reduce(acc, elm, score_threshold), \
        list(zip(top_images_results, lengths)), ([], 0, 0))
    return [f for f, _, _ in results[0]]

result_filenames = choose_images(results, raw_lines)


def float_left_elm(acc, elm):
    return acc + ([acc[-1]] if elm is False else [not acc[-1]])
float_left = reduce(float_left_elm, result_filenames, [False])[1:]

text_and_images = list(zip(raw_lines, result_filenames, float_left))
ensure_images([f for f in result_filenames if f is not False], True)


html_template_final = """
<!DOCTYPE html>
<html lang="en">
<head>
    <title>Story Output</title>
    <link href="https://fonts.googleapis.com/css?family=Libre+Baskerville|Playfair+Display+SC" rel="stylesheet">
    <link rel="stylesheet" href="https://unpkg.com/purecss@0.6.2/build/pure-min.css" integrity="sha384-UQiGfs9ICog+LwheBSRCt1o5cbyKIHbwjWscjemyBMT9YCUMZffs6UqUTd0hObXD" crossorigin="anonymous">
    <style type="text/css">
    .canidate {
        border-radius: 0.1em;
        margin: 0.8em;
        margin-top: 0.2em;
        max-width: 20em;
        max-height: 20em;
    }
    .text {
        font-family: 'Libre Baskerville', serif;
    }
    .shadow {
        -moz-box-shadow:    0.1em 0.1em 0.3em 0.4em #ccc;
        -webkit-box-shadow: 0.1em 0.1em 0.3em 0.4em #ccc;
        box-shadow:         0.1em 0.1em 0.3em 0.4em #ccc;
    }
    .title {
        font-family: 'Playfair Display SC', serif;
        text-align: right;
        margin-right: 1em;
        border-bottom: 3px grey solid;
        padding-right: 2em;
    }
    </style>
</head>
<body>
    <div class="pure-g">
        <div class="pure-u-1-8 pure-u-md-1-3"></div>
        <div class="pure-u-3-4 pure-u-md-1-3">
        <h2 class="title">{{ title }}</h2>
        {% for paragraph in paragraphs %}
            <div class="paragraph">
                <p class="text">
                {{ paragraph[0] }}
                </p>
                {% if paragraph[1] %}
                <img class="canidate shadow" style="float:{% if paragraph[2] %}left{% else %}right{% endif %}"
                    src="assets/{{ prefix }}{{ paragraph[1] }}" alt="{{ paragraph[1] }}" />
                {% endif %}
            </div>
        {% endfor %}
        </div>
        <div class="pure-u-1-8 pure-u-md-1-3"></div>
    </div>
</body>
</html>
"""


template_final = Template(html_template_final)

def render_template_final(paragraphs, prefix, filename, title):
    render = template_final.render(paragraphs=paragraphs, prefix=prefix, title=title)

    with open(os.path.join(opt.output, "{}.html".format(filename)), "w") as html_file:
        html_file.write(render)

render_template_final(text_and_images, '', 'clear')
render_template_final(text_and_images, 'prim.', 'index')

Logger.log("Story Written")
