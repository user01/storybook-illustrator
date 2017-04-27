
import os
import argparse
import glob
import datetime
import time
import re
import csv
import multiprocessing
import subprocess

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.autograd import Variable

from asg.model import Net
from asg.logger import Logger
from asg.word2vec import Word2Vec
from asg.data import ImageLoader

# Argument settings
parser = argparse.ArgumentParser(
    description='PyTorch Automated Storyboard Generator')
parser.add_argument('--epochs', type=int, default=2,
                    help='Number of epochs for training')
parser.add_argument('--learningrate', type=float,
                    default=0.01, help='Learning Rate. Default=0.01')
parser.add_argument('--seed', type=int, default=451,
                    help='Random seed. Default=451')
parser.add_argument('--batch', type=int, default=32,
                    help='Batch size. Default=32')
parser.add_argument('--workers', type=int, default=multiprocessing.cpu_count(),
                    help='Number of workers for data loader. Defaults to system cores')
parser.add_argument('--report', type=int, default=200,
                    help='Rate of reporting images. Default=200')
opt = parser.parse_args()
# opt = parser.parse_args(([
#     '--epochs',
#     '10',
#     '--batch',
#     '32',
#     '--workers',
#     '4',
#     '--learningrate',
#     '0.01',
#     '--seed',
#     '451'
# ]))

git_head = subprocess.check_output(['git',
                                    'rev-parse',
                                    '--short',
                                    'HEAD']).strip().decode('UTF-8')
Logger.log("Starting with model for {}".format(git_head))

Logger.log("Loading Word2Vec")
word2vec = Word2Vec()

Logger.log("Loading Training")
loader_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

image_loader_train = ImageLoader('train',
                                 word2vec,
                                 transform=loader_transforms)

loader_train = data.DataLoader(image_loader_train,
                               batch_size=opt.batch, shuffle=True,
                               num_workers=opt.workers, pin_memory=True)

Logger.log("Loading Testing")
image_loader_test = ImageLoader('test',
                                word2vec,
                                transform=loader_transforms)

loader_test = data.DataLoader(image_loader_test,
                              batch_size=opt.batch, shuffle=True,
                              num_workers=opt.workers, pin_memory=True)


Logger.log("Loading Network")
CUDA_AVAILABLE = torch.cuda.is_available()

torch.manual_seed(opt.seed)
if CUDA_AVAILABLE:
    torch.cuda.set_device(1)
    torch.cuda.manual_seed(opt.seed)


def variable(target):
    """Convert tensor to a variable"""
    if CUDA_AVAILABLE:
        target = target.cuda()
    return Variable(target)


net = Net()
if CUDA_AVAILABLE:
    net = net.cuda()

criterion = nn.CosineEmbeddingLoss()
optimizer = optim.SGD(net.parameters(), lr=opt.learningrate)


def text_size_to_variables(text_sizes):
    return [variable(torch.LongTensor([text_size])) for text_size in text_sizes]


starting_epoch = 0
past_models = sorted(glob.glob(os.path.join('models', 'model_{}*.pth'.format(git_head))))
if len(past_models) > 0:
    past_model = past_models[-1]
    epoch_regexp = re.compile(r".*epoch_(\d+)\.pth$")

    regexp_result = epoch_regexp.match(past_model)
    starting_epoch = int(regexp_result.groups()[0])

    Logger.log("Found past model at epoch {} of {}".format(
        starting_epoch, past_model))
    net.load_state_dict(torch.load(past_model))


csv_path = os.path.join('models', 'results_{}.csv'.format(git_head))


def write_line(arr):
    with open(csv_path, 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(arr)


if not os.path.isfile(csv_path):
    write_line(['epoch', 'timestamp', 'loss'])


def train(epoch):
    epoch_loss = 0
    start_time = time.time()
    net.train()
    for idx, (image, text, text_sizes, match) in enumerate(loader_train):
        optimizer.zero_grad()

        image = variable(image)
        text = variable(text)
        target = variable(match)
        text_sizes_var = text_size_to_variables(text_sizes)

        output_image_var, output_text_var = net(image, text, text_sizes_var)
        loss = criterion(output_image_var, output_text_var, target)
        epoch_loss += loss.data[0]
        loss.backward()
        optimizer.step()

        if idx % opt.report == 0:
            Logger.log("Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch,
                                                               idx,
                                                               len(loader_train),
                                                               loss.data[0] * 1000))

    end_time = time.time()
    Logger.log("Epoch {} Complete: Avg. Loss: {:.4f} over {}".format(
        epoch, 1000 * epoch_loss / len(loader_train),
        Logger.human_seconds(end_time - start_time)))


def test():
    epoch_loss = 0
    net.eval()
    for idx, (image, text, text_sizes, match) in enumerate(loader_test):

        image = variable(image)
        text = variable(text)
        target = variable(match)
        text_sizes_var = text_size_to_variables(text_sizes)

        output_image_var, output_text_var = net(image, text, text_sizes_var)
        loss = criterion(output_image_var, output_text_var, target)
        epoch_loss += loss.data[0]


    full_loss = 1000 * epoch_loss / len(loader_test)
    Logger.log("Avg. Test Loss: {:.4f}".format(full_loss))
    write_line([epoch, datetime.datetime.now().isoformat(), full_loss])


def checkpoint(epoch):
    model_out_path = os.path.join(
        "models", "model_{}_epoch_{:0>8}.pth".format(git_head, epoch))
    torch.save(net.state_dict(), model_out_path)
    Logger.log("Checkpoint saved to {}".format(model_out_path))


Logger.log("Starting ...")
for epoch in range(starting_epoch + 1, starting_epoch + opt.epochs + 1):
    train(epoch)
    test()
    checkpoint(epoch)
