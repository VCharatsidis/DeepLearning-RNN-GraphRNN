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

import argparse
import time
from datetime import datetime
import numpy as np
import pickle
import torch
from torch.utils.data import DataLoader

from part1.dataset import PalindromeDataset
from part1.vanilla_rnn import VanillaRNN
from part1.lstm import LSTM
import matplotlib.pyplot as plt

# You may want to look into tensorboardX for logging
# from tensorboardX import SummaryWriter

################################################################################

def train(config):

    assert config.model_type in ('RNN', 'LSTM')

    # Initialize the device which to run the model on
    device = torch.device(config.device)
    print('Running on {}'.format(device))
    # Initialize the model that we are going to use
    if config.model_type is 'RNN':
        model = VanillaRNN(config.input_length, config.input_dim, config.num_hidden, config.num_classes, config.batch_size, device)
    else:
        model = LSTM(config.input_length, config.input_dim, config.num_hidden, config.num_classes, config.batch_size, device)

    # Initialize the dataset and data loader (note the +1)
    dataset = PalindromeDataset(config.input_length+1)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Setup the loss and optimizer
    optimizer = torch.optim.RMSprop(model.parameters(), config.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    accuracies = []
    losses = []
    prev_accuracy = 0
    for step, (batch_inputs, batch_targets) in enumerate(data_loader):

        # Only for time measurement of step through network
        t1 = time.time()
        predictions = model.forward(batch_inputs)
        optimizer.zero_grad()
        loss = criterion(predictions, batch_targets)
        loss.backward()
        ############################################################################
        # QUESTION: what happens here and why?
        # we clip the gardients to prevent them from exploding, so to not have a numeric overflow.
        ############################################################################
        torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=config.max_norm)
        ############################################################################

        optimizer.step()


        loss = loss.item()

        preds = np.argmax(predictions.cpu().detach().numpy(), axis=1)

        targets = batch_targets.cpu().detach().numpy()
        accuracy = np.sum(preds == targets) / float(targets.shape[0])

        if accuracy > 0.98 and prev_accuracy > 0.98:
            break
        prev_accuracy = accuracy

        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size/(float(t2-t1)+0.000000000001)


        if step % 500 == 0:
            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                 "Accuracy = {:.2f}, Loss = {:.3f}".format(
                   datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                   config.train_steps, config.batch_size, examples_per_second,
                   accuracy, loss
            ))
        if step == config.train_steps:
            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break


    for j, (batch_inputs, batch_targets) in enumerate(data_loader):
        model.eval()
        predictions = model.forward(batch_inputs)
        loss = criterion(predictions, batch_targets)
        loss.backward()

        loss = loss.item()

        preds = np.argmax(predictions.cpu().detach().numpy(), axis=1)

        targets = batch_targets.cpu().detach().numpy()
        accuracy = np.sum(preds == targets) / float(targets.shape[0])
        accuracies.append(accuracy)

        if j > 29:
            break

    print('Done training.')
    print("Length {} max. accuracy: {}".format(config.input_length, max(accuracies)))

    return accuracies, losses
 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    # parser = argparse.ArgumentParser()
    #
    # # Model params
    # parser.add_argument('--model_type', type=str, default="LSTM", help="Model type, should be 'RNN' or 'LSTM'")
    # parser.add_argument('--input_length', type=int, default=19, help='Length of an input sequence')
    # parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence')
    # parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')
    # parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')
    # parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    # parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    # parser.add_argument('--train_steps', type=int, default=10000, help='Number of training steps')
    # parser.add_argument('--max_norm', type=float, default=10.0)
    # parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")
    #
    # config = parser.parse_args()
    # train(config)

    # Train the model
    total_accur = []
    lengths = []
    for i in range(5, 55, 2):

        parser = argparse.ArgumentParser()

        # Model params
        parser.add_argument('--model_type', type=str, default="LSTM", help="Model type, should be 'RNN' or 'LSTM'")
        parser.add_argument('--input_length', type=int, default=i, help='Length of an input sequence')
        parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence')
        parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')
        parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')
        parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
        parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
        parser.add_argument('--train_steps', type=int, default=10000, help='Number of training steps')
        parser.add_argument('--max_norm', type=float, default=10.0)
        parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")

        config = parser.parse_args()

        accuracies, losses = train(config)

        total_accur.append(sum(accuracies)/30)
        lengths.append(i)

    plt.plot(total_accur, lengths)
    plt.ylabel('lengths')
    plt.xlabel('accuracy')
    plt.show()

