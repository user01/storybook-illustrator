
import os
import argparse
import glob
import datetime
import time
import re
import csv

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
parser.add_argument('--report', type=int, default=200,
                    help='Rate of reporting images. Default=200')
# opt = parser.parse_args()
opt = parser.parse_args(([
    '--epochs',
    '10',
    '--learningrate',
    '0.01',
    '--seed',
    '451'
]))

Logger.log("Loading Word2Vec")
word2vec = Word2Vec()

Logger.log("Loading Training")
image_loader_train = ImageLoader('train',
                 word2vec,
                 transform=transforms.Compose([
                     transforms.RandomHorizontalFlip(),
                     transforms.ToTensor(),
                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])
                 ]))

train_loader = data.DataLoader(image_loader_train,
   batch_size=4, shuffle=True,
   num_workers=4, pin_memory=True)

Logger.log("Loading Testing")
image_loader_train = ImageLoader('test',
                 word2vec,
                 transform=transforms.Compose([
                     transforms.RandomHorizontalFlip(),
                     transforms.ToTensor(),
                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])
                 ]))

train_loader = data.DataLoader(image_loader_train,
   batch_size=4, shuffle=True,
   num_workers=4, pin_memory=True)


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

criterion = nn.CosineEmbeddingLoss()
optimizer = optim.SGD(net.parameters(), lr=opt.learningrate)



net.train()
# loader = DataLoader('train', word2vec, seed=epoch)

def text_size_to_variables(text_sizes):
    return [variable(torch.LongTensor([text_size])) for text_size in text_sizes]

for idx, (image, text, text_sizes, match) in enumerate(train_loader):
    optimizer.zero_grad()

    image = variable(image)
    text = variable(text)
    target = variable(match)
    text_sizes_var = text_size_to_variables(text_sizes)

    output_image_var, output_text_var = net(image, text, text_sizes_var)
    loss = criterion(output_image_var, output_text_var, target)
    loss.backward()
    optimizer.step()
    break


output_image_var

res = output_text_var.data.cpu()
res

ts = text_size.type(torch.LongTensor)

lst = ts.squeeze().tolist()


def gg(idx, elm):
    return idx + elm

[idx + elm for  idx, elm in enumerate(range(3))]




list(map(lambda (idx, elm): torch.index_select(res, 1, torch.LongTensor([elm]))[idx], enumerate(lst)))
torch.stack([torch.index_select(res, 1, torch.LongTensor([elm]))[idx] for idx, elm in enumerate(lst)])



pp = torch.stack([torch.index_select(res, 1, torch.LongTensor([elm]))[idx] for idx, elm in enumerate(lst)])

torch.squeeze(pp)

ts.tolist()

res[:]

torch.index_select(res, 1, torch.LongTensor([1]))
rd = torch.index_select(res, 1, ts.squeeze())
rd


torch.index_select(rd, 1, ts.squeeze())


res[[1]]


starting_epoch = 0
past_models = sorted(glob.glob(os.path.join('models', '*.pth')))
if len(past_models) > 0:
    past_model = past_models[-1]
    epoch_regexp = re.compile(r".*epoch_(\d+)\.pth$")

    regexp_result = epoch_regexp.match(past_model)
    starting_epoch = int(regexp_result.groups()[0])

    Logger.log("Found past model at epoch {} of {}".format(
        starting_epoch, past_model))
    net.load_state_dict(torch.load(past_model))


csv_path = os.path.join('models', 'results.csv')


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
    loader = DataLoader('train', word2vec, seed=epoch)
    for idx, (image, text, distance) in enumerate(loader):
        optimizer.zero_grad()

        image = variable(image)
        text = variable(text)
        target = variable(torch.FloatTensor([distance]))

        output_image_var, output_text_var = net(image, text)
        loss = criterion(output_image_var, output_text_var, target)
        epoch_loss += loss.data[0]
        loss.backward()
        optimizer.step()

        if idx % opt.report == 0:
            Logger.log("Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch,
                                                               idx,
                                                               len(loader),
                                                               loss.data[0] * 1000))

    end_time = time.time()
    Logger.log("Epoch {} Complete: Avg. Loss: {:.4f} over {}".format(
        epoch, 1000 * epoch_loss / len(loader),
        Logger.human_seconds(end_time - start_time)))


def test():
    epoch_loss = 0
    net.eval()
    loader = DataLoader('test', word2vec)
    for idx, (image, text, distance) in enumerate(loader):

        image = variable(image)
        text = variable(text)
        target = variable(torch.FloatTensor([distance]))

        output_image_var, output_text_var = net(image, text)
        loss = criterion(output_image_var, output_text_var, target)
        epoch_loss += loss.data[0]

        if idx > 2 * opt.report:
            break  # large testing currently not useful

    full_loss = 1000 * epoch_loss / len(loader)
    Logger.log("Avg. Test Loss: {:.4f}".format(full_loss))
    write_line([epoch, datetime.datetime.now().isoformat(), full_loss])


def checkpoint(epoch):
    model_out_path = os.path.join(
        "models", "model_epoch_{:0>8}.pth".format(epoch))
    torch.save(net.state_dict(), model_out_path)
    Logger.log("Checkpoint saved to {}".format(model_out_path))


Logger.log("Starting ...")
for epoch in range(starting_epoch + 1, starting_epoch + opt.epochs + 1):
    train(epoch)
    test()
    checkpoint(epoch)
