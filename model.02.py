# -*- coding: utf-8 -*-
"""Second version of Storyboard Model generation"""


import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torchvision.models as models

from asg.data import data_train

cuda_available = torch.cuda.is_available()

torch.manual_seed(451)
if cuda_available:
    torch.cuda.manual_seed_all(451)

def variable(target):
    """Convert tensor to a variable"""
    if cuda_available:
        target = target.cuda()
    return Variable(target)

EPOCHS_TO_TRAIN = 1


class Net(nn.Module):
    """Image and Text module"""
    _number_one = variable(torch.FloatTensor([1]))

    def __init__(self):
        super(Net, self).__init__()

        self._cnn = models.resnet18(pretrained=True)
        self._cnn.fc = nn.Linear(512, 300, True)

        self._model_lstm = nn.LSTM(300, 300, 2)

    def forward(self, image_var, text_var):
        output_image_var = self._cnn(image_var)
        output_text_seq, _ = self._model_lstm(text_var)

        output_text_var = output_text_seq[-1]
        output_distance = Net.cosine_distance(
            output_image_var, output_text_var)

        return output_distance

    @staticmethod
    def cosine_distance(tensor_1, tensor_2):
        """Measure cosine distance of tensor variables"""
        numerator = tensor_1.mul(tensor_2).sum()

        denominator_1 = tensor_1.pow(2).sum().pow(0.5)
        denominator_2 = tensor_2.pow(2).sum().pow(0.5)
        denominator = denominator_1.mul(denominator_2)

        div_result = numerator.div(denominator)
        result = Net._number_one.sub(div_result)

        return result

net = Net()

if cuda_available:
    net = net.cuda()

criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)


print("Training loop:")
total_per_epoch = len(data_train)
net.train()

for epoch in range(EPOCHS_TO_TRAIN):
    for idx, (image, text, distance) in enumerate(data_train):
        optimizer.zero_grad()   # zero the gradient buffers
        image = variable(image)
        text = variable(text)
        output = net(image, text)
        target = variable(torch.FloatTensor([distance]))
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()    # Does the update
        if idx % (total_per_epoch // 25) == 0:
            print("Epoch: {: >6} | Index: {: >6} | Complete {: >7.2%} | Loss {:.8f}".format(epoch, idx, idx / total_per_epoch, loss.data.cpu().numpy()[0]))
    if epoch % 1 == 0:
        print("Epoch {: >8} Loss: {}".format(
            epoch, loss.data.cpu().numpy()[0]))
