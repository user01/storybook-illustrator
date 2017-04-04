import torch
import math
import os
import time


from torch.autograd import Variable
from torch.utils.data import TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models


EPOCHS_TO_TRAIN = 25000

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 3, True)
        self.fc2 = nn.Linear(3, 1, True)

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x

net = Net()

# note the shape here. Each must be 2d tensor (in this case, of size 1x2)
dataset = TensorDataset(
    torch.Tensor([
        [[0, 0]],
        [[0, 1]],
        [[1, 0]],
        [[1, 1]]
    ]),
    torch.Tensor([
        [[0]],
        [[1]],
        [[1]],
        [[0]]
    ])
)



criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)

print("Training loop:")
for idx in range(0, EPOCHS_TO_TRAIN):
    for input, target in dataset:
        optimizer.zero_grad()   # zero the gradient buffers
        # print(idx)
        output = net(Variable(input))
        loss = criterion(output, Variable(target))
        loss.backward()
        optimizer.step()    # Does the update
    if idx % 5000 == 0:
        print("Epoch {: >8} Loss: {}".format(idx, loss.data.numpy()[0]))



print("")
print("Final results:")
for input, target in dataset:
    input_variable = Variable(input)
    target_variable = Variable(target)
    output = net(input_variable)
    print("Input:[{},{}] Target:[{}] Predicted:[{}] Error:[{}]".format(
        int(input_variable.data.numpy()[0][0]),
        int(input_variable.data.numpy()[0][1]),
        int(target_variable.data.numpy()[0]),
        round(float(output.data.numpy()[0]), 4),
        round(float(abs(target_variable.data.numpy()[0] - output.data.numpy()[0])), 4)
    ))
