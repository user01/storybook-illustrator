# Automated Storyboard Generation

## Abstract

### Summary
Our automated storyboard generation project will explore visual storytelling by annotating a narrative with images. Ultimately, the images will form a cohesive flow of events described in the text. We plan to use Microsoft Research’s Visual Storytelling Dataset [1] to build a two part network relating images and text descriptions. Once trained, this architecture will create a storyboard of images for unseen text narratives.


### Related Work

Previous work in the area of mapping image and caption embeddings to a vector space has been used to expand classification tasks for unseen images. For example, the DeViSE model [2] used an embedding vector of semantically rich text and a visual model trained on ImageNet [3] to make inferences about unobserved visual objects. Researchers from Google [4] built a CNN to maximize the likelihood of producing a target sequence of words to generate an image description. In their model’s learned embedding space, learned representations captured semantics of the text. For example, the proximity of the words "horse", "pony", and "donkey" encouraged their CNN to extract images with horse-like animals. In another work, researchers from the University of Toronto and MIT demonstrated how a model can align visual content from book-to-movie clips and their respective book sentences [5]. In order to learn visual-semantic embeddings they used a dot produced based score computed from both fixed vectors of book sentences and linearly mapped vectors of movie clip image features. Movie clips were represented as vectors corresponding to mean-pooled features across each frame in the clip. In a similar vein, our project intends to map image features and text to vectors to choose sequentially appropriate images that capture the text’s semantics.

### Data
Our primary data source will be the Visual Storytelling Dataset (VIST) from Microsoft Research. This dataset contains 81, 743 images in 20,211 sequences with corresponding text annotations.  The two types of text annotations include: descriptions of images-in-isolation (DII) and stories of images-in-sequence (SIS). The images are separated into training, validation, and testing sets with approximately 40,000, 5,000, and 5,000 instances respectively.

### Objective
Two main objectives drive our project: (1) Train a model that maximizes cosine similarity between images and text in order to generate image candidates for an unseen text narrative (2) Generate a visually consistent and thematic storyboard from image candidates using cosine similarity maximization. To accomplish the first objective, a convolutional neural network and a recurrent neural network are used to reduce images and text to vectors, allowing the network to be trained to maximize the cosine similarity.

Once the networks are trained to maximize similarity between related text and images, vectors can be cached for all of the available images. Unseen text (i.e. the new story) are then reduced to another vector, allowing calculation of cosine similarity between the text and all images. The closest N images are collected as storyboard candidates for the text.

The second objective of maintaining a consistent visual tone and flow for the storyboard will be accomplished by maximizing cosine similarity for the chain of candidate images for each text. The intention is to pick the storyboard from candidate images that both maximize cosine similarity and thematic consistency.

### References

[1] Huang, Ting-Hao Kenneth, et al. "Visual storytelling." (2016).

[2] Frome, Andrea, et al. "Devise: A deep visual-semantic embedding model." Advances in neural information processing systems. 2013.

[3] Olga Russakovsky*, Jia Deng*, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg and Li Fei-Fei. (* = equal contribution) ImageNet Large Scale Visual Recognition Challenge. IJCV, 2015

[4] Vinyals, Oriol, et al. "Show and tell: A neural image caption generator." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2015.

[5] Zhu, Yukun, et al. "Aligning books and movies: Towards story-like visual explanations by watching movies and reading books." Proceedings of the IEEE International Conference on Computer Vision. 2015.

## Build Notes

### Jupyter

Modifying the [Jupyter](https://jupyter.org/) installation to produce `.py` and `.html` on Jupyter saves will simplify running and updating of all models. The following instructions are pulled from [Jupyter Notebook Best Practices for Data Science](https://www.svds.com/jupyter-notebook-best-practices-for-data-science/).

First, if the `~/.jupyter/jupyter_notebook_config.py` configuration file does not exist, run `jupyter notebook --generate-config`.

Prepend this python to the configuration file at `~/.jupyter/jupyter_notebook_config.py`.

```python
c = get_config()
### If you want to auto-save .html and .py versions of your notebook:
# modified from: https://github.com/ipython/ipython/issues/8009
import os
from subprocess import check_call

def post_save(model, os_path, contents_manager):
    """post-save hook for converting notebooks to .py scripts"""
    if model['type'] != 'notebook':
        return # only do this for notebooks
    d, fname = os.path.split(os_path)
    check_call(['jupyter', 'nbconvert', '--to', 'script', fname], cwd=d)
    check_call(['jupyter', 'nbconvert', '--to', 'html', fname], cwd=d)

c.FileContentsManager.post_save_hook = post_save
```

### Dependencies

Python dependencies are in the pip3 file `requirements.txt`. Note that nltk requires data files, which are loaded by `python -m nltk.downloader all`. Pytorch will automatically retrieve resnet preloadings on the first run.

### Data

The [Visual Storytelling Dataset (VIST)](http://visionandlanguage.net/VIST/) needs to be downloaded to the local disk and extracted. The location folder must be stored in a plain text file in the project root called `data.directory.txt`.

Word embeddings can be downloaded from Google [here](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing) and must be extracted in the root of the **DATA_DIRECTORY**.

The **DATA_DIRECTORY** needs to have the following structure:

```
DATA_DIRECTORY/
┣━━GoogleNews-vectors-negative300.bin
┗━┳━dii/
  ┣━sis/
  ┣━train━images━images┳━image.0.jpg
  ┃                    ┣━image.1.jpg
  ┃                    ┣━...
  ┃                    ┗━image.12.jpg
  ┗━test━━images━test━━┳━image.0.jpg
                       ┣━image.1.jpg
                       ┣━...
                       ┗━image.12.jpg
```

 * `dii` contains the Description in Isolation JSON
 * `sis` contains the Story in Sequence JSON
 * `train` contains the training images, in two subfolders (extracted train_split.*.tar.gz)
 * `test` contains the testing images, in two subfolders (extracted test_split.tar.gz)

The deep folder structure is an artifact of how pytorch's Image Folder considers the assets.

#### ImageMagick Command

```bash
mogrify -path . -resize "224x224^" -gravity center -crop 224x224+0+0 *.*
```
