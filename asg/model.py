# -*- coding: utf-8 -*-
"""Storyboard Model"""

import torch
import torch.nn as nn
import torchvision.models as models


class Net(nn.Module):
    """Image and Text module"""

    def __init__(self):
        super(Net, self).__init__()

        self._cnn = models.resnet18(pretrained=True)
        self._cnn_final_01 = nn.Linear(1000, 512, True)
        self._cnn_final_02 = nn.Linear(512, 300, True)

        self._model_lstm = nn.LSTM(300, 300, 4, dropout=0.2, batch_first=True)

    def forward(self, image_var, text_var, text_sizes):
        """Overridden forward method"""
        output_cnn_var = self._cnn(image_var)
        output_final_01_var = self._cnn_final_01(output_cnn_var)
        output_image_var = self._cnn_final_02(output_final_01_var)
        output_text_seq, _ = self._model_lstm(text_var)

        output_text_var = output_text_seq
        output_text_var = Net._select_from_lstm(output_text_seq, text_sizes)

        return output_image_var, output_text_var

    @staticmethod
    def _select_from_lstm(tensors_full, text_sizes):
        tensors_selected = torch.stack([torch.index_select(tensors_full, 1, elm)[idx]
                                        for idx, elm in enumerate(text_sizes)])
        return torch.squeeze(tensors_selected)
