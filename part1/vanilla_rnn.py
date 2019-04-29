
################################################################################
# MIT License
#
# Copyright (c) 2018
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

import torch
import torch.nn as nn


################################################################################

class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(VanillaRNN, self).__init__()
        # Initialization here ...

        self.W_hx = torch.nn.Parameter(torch.zeros(num_hidden, seq_length))
        torch.nn.init.xavier_uniform_(self.W_hx)

        self.W_hh = nn.Parameter(torch.zeros(num_hidden, num_classes))
        torch.nn.init.xavier_uniform_(self.W_hh)

        self.b_h = nn.Parameter(torch.zeros(batch_size))

        self.W_ph = torch.nn.Parameter(torch.zeros(num_classes, num_hidden))
        torch.nn.init.xavier_uniform_(self.W_ph)

        self.b_p = torch.nn.Parameter(torch.zeros(num_classes))

        self.seq_length = seq_length
        self.batch_size = batch_size
        self.num_hidden = num_hidden
        self.device = device
        self.num_classes = num_classes

    def forward(self, x):
        # Implementation here ...
        y_prev = torch.zeros(self.batch_size, self.num_classes)

        for i in range(self.seq_length):

            a = torch.mm(self.W_hx, x.transpose(1, 0))

            b = torch.mm(self.W_hh, y_prev.transpose(1, 0))

            linear = a + b

            linear2 = linear + self.b_h

            tanh = torch.nn.Tanh()
            h = tanh(linear2)

            p = torch.mm(self.W_ph, h).transpose(1, 0) + self.b_p

            softmax = torch.nn.Softmax()
            y = softmax(p)

            y_prev = y

        return y_prev



