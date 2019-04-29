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
        # Initialization here ...
        self.W_gx = torch.nn.Parameter(torch.zeros(num_hidden, seq_length))
        torch.nn.init.xavier_uniform_(self.W_gx)

        self.W_gh = nn.Parameter(torch.zeros(num_hidden, num_classes))
        torch.nn.init.xavier_uniform_(self.W_gh)

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
        pass