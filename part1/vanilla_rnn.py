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
        self.W_hx = torch.nn.Parameter(torch.zeros(num_hidden, num_classes))
        self.W_hh = nn.Parameter(torch.zeros(num_hidden, num_hidden))
        self.b_h = nn.Parameter(torch.zeros(num_hidden))

        self.W_ph = torch.nn.Parameter(torch.zeros(num_classes, num_hidden))
        self.b_p = torch.nn.Parameter(torch.zeros(num_classes))

        self.seq_length = seq_length
        self.batch_size = batch_size
        self.num_hidden = num_hidden
        self.device = device

    def forward(self, x):
        # Implementation here ...
        y_prev = torch.zeros(self.num_hidden, self.batch_size)

        for i in range(self.seq_length):
            print("Whx shape")
            print(self.W_hx.shape)
            print("x shape")
            print(x.shape)
            print("")


            #x = x[:, i].view(1, -1)
            a = torch.mm(self.W_hx, x.transpose(1,0))

            print("Whh shape")
            print(self.W_hh.shape)
            print("y_prev shape")
            print(y_prev.shape)
            print("b_h shapes")
            print(self.b_h.shape)
            b = torch.mm(self.W_hh, y_prev)

            print("b shape")
            print(b.shape)
            print("a shape")
            print(a.shape)
            linear = a + b

            linear2 = linear + self.b_h
            print("linear shape")
            print(linear2.shape)
            tanh = torch.nn.Tanh()
            h = tanh(linear2)

            #p = h.transpose(1, 0) @ self.W_ph.transpose(1, 0) + self.b_p  # (h, b)T (o, h)T  + (o) = (b, o)

            p = torch.mm(self.W_ph, h).transpose(1, 0) + self.b_p

            print("p shape")
            print(p.shape)
            softmax = torch.nn.Softmax()
            y = softmax(p)

            y_prev = y


        return y_prev