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
#     dtype = torch.cuda.FloatTensor
# else:
#     dtype = torch.FloatTensor

EPOCHS_TO_TRAIN = 1


class Net(nn.Module):
    """Image and Text module"""
    _number_one = Variable(torch.FloatTensor([1]).cuda())

    def __init__(self):
        super(Net, self).__init__()
        self._cnn = models.resnet18(pretrained=True).cuda()
        self._cnn.fc = nn.Linear(512, 300, True).cuda()

        self._model_lstm = nn.LSTM(300, 300, 2).cuda()

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

net._cnn.cuda()

net._cnn(image)

criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)


print("Training loop:")
total_per_epoch = len(data_train)
net.train()

for epoch in range(EPOCHS_TO_TRAIN):
    for idx, (image, text, distance) in enumerate(data_train):
        print("Epoch: {: >6} | Index: {: >6} | Complete {:.2%}".format(epoch, idx, idx / total_per_epoch))
        optimizer.zero_grad()   # zero the gradient buffers
        image = Variable(image.cuda())
        text = Variable(text.cuda())
        output = net(image, text)
        target = Variable(torch.FloatTensor([distance]).cuda())
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()    # Does the update
    if epoch % 1 == 0:
        print("Epoch {: >8} Loss: {}".format(
            epoch, loss.data.numpy()[0]))
