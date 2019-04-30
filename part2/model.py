# MIT License
#
# Copyright (c) 2017 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2018
# Date Created: 2018-09-04
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch

class TextGenerationModel(nn.Module):

    def __init__(self, batch_size, seq_length, vocabulary_size,
                 lstm_num_hidden=256, lstm_num_layers=2, device='cuda:0'):

        super(TextGenerationModel, self).__init__()
        # Initialization here...
        self.lstm = nn.LSTM(vocabulary_size, lstm_num_hidden, lstm_num_layers)
        self.linear = nn.Linear(lstm_num_hidden, vocabulary_size)


        self.vocabulary_size = vocabulary_size


    def forward(self, x):
        # Implementation here...

        # Step through the sequence one element at a time.
        # after each step, hidden contains the hidden state.
        out, _ = self.lstm(x.view(1, 1, -1))

        linear_out = self.linear(out, self.vocabulary_size)
        return linear_out
