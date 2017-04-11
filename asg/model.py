# -*- coding: utf-8 -*-
"""Storyboard Model"""


import torch
import torch.nn as nn
import torch.autograd as autograd
import torchvision.models as models

CUDA_AVAILABLE = torch.cuda.is_available()

class Variable(autograd.Variable):
    """Variable that makes use of GPU if available"""

    def __init__(self, data, *args, **kwargs):
        if CUDA_AVAILABLE:
            data = data.cuda()
        super(Variable, self).__init__(data, *args, **kwargs)


class Net(nn.Module):
    """Image and Text module"""
    _number_one = Variable(torch.FloatTensor([1]))

    def __init__(self):
        super(Net, self).__init__()

        self._cnn = models.resnet18(pretrained=True)
        self._cnn.fc = nn.Linear(512, 300, True)

        self._model_lstm = nn.LSTM(300, 300, 2)

    def forward(self, image, text):
        """Overridden forward method"""
        image_var = Variable(image)
        text_var = Variable(text)

        output_image_var = self._cnn(image_var)
        output_text_seq, _ = self._model_lstm(text_var)

        output_text_var = output_text_seq[-1]
        output_distance = Net.cosine_distance(output_image_var, output_text_var)

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
