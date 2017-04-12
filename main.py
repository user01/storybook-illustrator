
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from asg.model import Net
from asg.logger import Logger
from asg.word2vec import Word2Vec
from asg.data import Loader

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
opt = parser.parse_args()
# opt = parser.parse_args(([
#     '--epochs',
#     '1',
#     '--learningrate',
#     '0.01',
#     '--seed',
#     '451'
# ]))

Logger.log("Loading Word2Vec")
word2vec = Word2Vec()
Logger.log("Loading Data Loader")
loader = Loader(word2vec)

Logger.log("Main Network")
CUDA_AVAILABLE = torch.cuda.is_available()

torch.manual_seed(opt.seed)
if CUDA_AVAILABLE:
    torch.cuda.manual_seed(opt.seed)

net = Net()
number_one = torch.FloatTensor([1])
if CUDA_AVAILABLE:
    net = net.cuda()
    number_one = number_one.cuda()

net.number_one = Variable(number_one)


criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=opt.learningrate)

Logger.log("Loading datasets")


def variable(target):
    """Convert tensor to a variable"""
    if CUDA_AVAILABLE:
        target = target.cuda()
    return Variable(target)


def train(epoch):
    epoch_loss = 0
    net.train()
    for idx, (image, text, distance) in enumerate(loader.data_train):
        optimizer.zero_grad()
        if CUDA_AVAILABLE:
            image = image.cuda()
            text = text.cuda()

        image = variable(image)
        text = variable(text)
        output = net(image, text)
        target = variable(torch.FloatTensor([distance]))
        loss = criterion(output, target)
        epoch_loss += loss.data[0]
        loss.backward()
        optimizer.step()

        if idx % opt.report == 0:
            Logger.log("Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch,
                                                               idx,
                                                               len(loader.data_train),
                                                               loss.data[0]))

    Logger.log("Epoch {} Complete: Avg. Loss: {:.4f}".format(
        epoch, epoch_loss / len(loader.data_train)))


def test():
    epoch_loss = 0
    net.eval()
    for idx, (image, text, distance) in enumerate(loader.data_test):

        image = variable(image)
        text = variable(text)
        target = variable(torch.FloatTensor([distance]))
        prediction = net(image, text)
        loss = criterion(prediction, target)
        epoch_loss += loss.data[0]

        if idx % opt.report == 0:
            Logger.log("Current Avg. Test Loss: {:.4f}".format(
                epoch_loss / idx))

    Logger.log("Avg. Test Loss: {:.4f}".format(
        epoch_loss / len(loader.data_test)))


def checkpoint(epoch):
    model_out_path = "model_epoch_{}.pth".format(epoch)
    torch.save(net, model_out_path)
    Logger.log("Checkpoint saved to {}".format(model_out_path))


for epoch in range(1, opt.epochs + 1):
    train(epoch)
    test()
    checkpoint(epoch)
