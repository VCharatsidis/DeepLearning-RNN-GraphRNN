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

class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(LSTM, self).__init__()
        # g gate
        self.W_gx = torch.nn.Parameter(torch.zeros(input_dim, num_hidden))
        torch.nn.init.xavier_uniform_(self.W_gx)

        self.W_gh = nn.Parameter(torch.zeros(num_hidden, num_hidden))
        torch.nn.init.xavier_uniform_(self.W_gh)

        self.b_g = nn.Parameter(torch.zeros(num_hidden))
        
        # i gate
        self.W_ix = torch.nn.Parameter(torch.zeros(input_dim, num_hidden))
        torch.nn.init.xavier_uniform_(self.W_ix)

        self.W_ih = torch.nn.Parameter(torch.zeros(num_hidden, num_hidden))
        torch.nn.init.xavier_uniform_(self.W_ih)

        self.b_i = nn.Parameter(torch.zeros(num_hidden))

        # f gate

        self.W_fx = torch.nn.Parameter(torch.zeros(input_dim, num_hidden))
        torch.nn.init.xavier_uniform_(self.W_fx)

        self.W_fh = torch.nn.Parameter(torch.zeros(num_hidden, num_hidden))
        torch.nn.init.xavier_uniform_(self.W_fh)

        self.b_f = nn.Parameter(torch.zeros(num_hidden))

        # o gate

        self.W_ox = torch.nn.Parameter(torch.zeros(input_dim, num_hidden))
        torch.nn.init.xavier_uniform_(self.W_ox)

        self.W_oh = torch.nn.Parameter(torch.zeros(num_hidden, num_hidden))
        torch.nn.init.xavier_uniform_(self.W_oh)

        self.b_o = nn.Parameter(torch.zeros(num_hidden))

        # p

        self.W_ph = torch.nn.Parameter(torch.zeros(num_hidden, num_classes))
        torch.nn.init.xavier_uniform_(self.W_ph)

        self.b_p = torch.nn.Parameter(torch.zeros(num_classes))

        # config
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.num_hidden = num_hidden
        self.device = device
        self.num_classes = num_classes

    def forward(self, x):
        # Implementation here ...
        y_prev = torch.zeros(self.batch_size, self.num_hidden)
        c_prev = torch.zeros(self.batch_size, self.num_hidden)

        for i in range(self.seq_length):
            input = x.narrow(1, i, 1)

            # g
            linear_g = torch.mm(input, self.W_gx) + torch.mm(y_prev, self.W_gh) + self.b_g
            tanh = torch.nn.Tanh()
            g = tanh(linear_g)

            # i
            linear_i = torch.mm(input, self.W_ix) + torch.mm(y_prev, self.W_ih) + self.b_i
            sigmoid = torch.nn.Sigmoid()
            i = sigmoid(linear_i)

            # f
            linear_f = torch.mm(input, self.W_fx) + torch.mm(y_prev, self.W_fh) + self.b_f
            f = sigmoid(linear_f)

            # o
            linear_o = torch.mm(input, self.W_ox)+ torch.mm(y_prev, self.W_oh) + self.b_o
            o = sigmoid(linear_o)

            c = g * i + c_prev * f
            c_prev = c

            h = tanh(c) * o
            y_prev = h

        p = torch.mm(y_prev, self.W_ph) + self.b_p

        return p

