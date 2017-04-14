# -*- coding: utf-8 -*-
"""Storyboard Model"""


import torch.nn as nn
import torchvision.models as models


class Net(nn.Module):
    """Image and Text module"""
    number_one = None

    def __init__(self):
        super(Net, self).__init__()

        self._cnn = models.resnet18(pretrained=True)
        self._cnn.fc = nn.Linear(512, 300, True)

        self._model_lstm = nn.LSTM(300, 300, 2)

    def forward(self, image_var, text_var):
        """Overridden forward method"""
        output_image_var = self._cnn(image_var)
        output_text_seq, _ = self._model_lstm(text_var)

        output_text_var = output_text_seq[-1]
        # output_distance = self.cosine_distance(output_image_var, output_text_var)

        return output_image_var, output_text_var

    def cosine_distance(self, tensor_1, tensor_2):
        """Measure cosine distance of tensor variables"""
        numerator = tensor_1.mul(tensor_2).sum(1)

        denominator_1 = Net._cosine_denominator(tensor_1)
        denominator_2 = Net._cosine_denominator(tensor_2)
        denominator = denominator_1.mul(denominator_2)

        div_result = numerator.div(denominator)
        result = self.number_one.sub(div_result)

        return result

    @staticmethod
    def _cosine_denominator(tensor):
        return tensor.pow(2).sum(1).pow(0.5)
