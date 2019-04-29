
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

        self.W_hx = torch.nn.Parameter(torch.zeros(1, num_hidden))
        torch.nn.init.xavier_uniform_(self.W_hx)

        self.W_hh = nn.Parameter(torch.zeros(num_hidden, num_hidden))
        torch.nn.init.xavier_uniform_(self.W_hh)

        self.b_h = nn.Parameter(torch.zeros(num_hidden))

        self.W_ph = torch.nn.Parameter(torch.zeros(num_hidden, num_classes))
        torch.nn.init.xavier_uniform_(self.W_ph)

        self.b_p = torch.nn.Parameter(torch.zeros(num_classes))

        self.seq_length = seq_length
        self.batch_size = batch_size
        self.num_hidden = num_hidden
        self.device = device
        self.num_classes = num_classes
        self.input_dim = input_dim

    def forward(self, x):
        # Implementation here ...
        y_prev = torch.zeros(self.batch_size, self.num_hidden)

        for i in range(self.seq_length):

            input = x.narrow(1, i, 1)

            a = torch.mm(input, self.W_hx)

            b = torch.mm(y_prev, self.W_hh)

            linear = a + b + self.b_h

            tanh = torch.nn.Tanh()
            y_prev = tanh(linear)

        p = torch.mm(y_prev, self.W_ph) + self.b_p

        return p



