
# coding: utf-8

# In[65]:

# import os
# import argparse
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.autograd import Variable
# 
# from asg.model import Net
# from asg.logger import Logger
# from asg.word2vec import Word2Vec
# from asg.data import DataLoader


# In[66]:

import numpy as np
from scipy import spatial
import operator
from functools import reduce


# ### Storyboard Selection
# #### 1. Parse text corpus into sentences
# #### 2. Create dummy image and sentence embeddings
# #### 3. Get closest k = 5 dummy images for each dummy sentence
# #### 4. Create final storyboard by selecting image closest to previous sentence's image (first sentence's image is closest image to sentence)
# 

# In[69]:

# parse corpus into sentences

# http://stackoverflow.com/questions/4576077/python-split-text-on-sentences
import re
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


# In[70]:

text = "I like pigs. I like pigs that can fly. I like cats. I like cats that can fetch."


# In[71]:

split_into_sentences(text)


# In[72]:

# sample images
images = ["000.jpg", "100.jpg", "200.jpg", "300.jpg", "400.jgp", "500.jpg", "600.jpg", "700.jpg", "800.jpg", "900.jpg"]

# dict that holds image name and embedding
image_dict = {}

# produce 300-d embedding for each image (what CNN will do)
for i in range(len(images)):
    image_dict[images[i]] = np.random.randint(5, size=(1,300))


# In[87]:

# produce 300-d embedding for 10 sentences (what LSTM will do)
sentence_embeddings = []

for i in range(10):
    sentence_embeddings.append(np.random.randint(5, size=(1,300)))


# In[186]:

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

