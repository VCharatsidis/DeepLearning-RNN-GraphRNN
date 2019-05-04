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

import os
import time
from datetime import datetime
import argparse

import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from part2.dataset import TextDataset
from part2.model import TextGenerationModel

################################################################################

def generator(model, seed, length, dataset):

    with torch.no_grad():

        text_fifteen = seed.view(-1).tolist()
        text = seed.view(-1).tolist()
        text_beta_nine = seed.view(-1).tolist()
        text_beta_five = seed.view(-1).tolist()
        text_beta_one = seed.view(-1).tolist()


        size = list(seed.shape)
        size.append(dataset.vocab_size)
        one_hot_seed = torch.zeros(size, device=seed.device)
        one_hot_seed.scatter_(2, seed.unsqueeze(-1), 1)

        predictions = model(one_hot_seed)

        # sample
        distribution = torch.softmax(predictions[:, -1, :].squeeze(), dim=0)
        next_char = torch.multinomial(distribution, 1).item()

        text_fifteen.append(next_char)
        text.append(next_char)
        text_beta_nine.append(next_char)
        text_beta_five.append(next_char)
        text_beta_one.append(next_char)

        beta = [1.5, 0.9, 0.5, 0.2]

        for l in range(length - 1):

            t = torch.tensor(next_char, dtype=torch.long).view(1, -1)
            size = list(t.shape)
            size.append(dataset.vocab_size)
            one_hot_t = torch.zeros(size, device=t.device)
            one_hot_t.scatter_(2, t.unsqueeze(-1), 1)

            predictions = model(one_hot_t)

            distribution_fifteen = torch.softmax(predictions[:, -1, :].squeeze(), dim=0)
            next_char_fifteen = torch.multinomial(distribution_fifteen, 1).item()

            distribution = torch.softmax(predictions[:, -1, :].squeeze(), dim=0)
            next_char = torch.multinomial(distribution, 1).item()

            distribution_nine = torch.softmax(predictions[:, -1, :].squeeze() * beta[0], dim=0)
            next_char_nine = torch.multinomial(distribution_nine, 1).item()

            distribution_five = torch.softmax(predictions[:, -1, :].squeeze() * beta[1], dim=0)
            next_char_five = torch.multinomial(distribution_five, 1).item()

            distribution_one = torch.softmax(predictions[:, -1, :].squeeze() * beta[2], dim=0)
            next_char_one = torch.multinomial(distribution_one, 1).item()

            text_fifteen.append(next_char_fifteen)
            text.append(next_char)
            text_beta_nine.append(next_char_nine)
            text_beta_five.append(next_char_five)
            text_beta_one.append(next_char_one)

        text_fifteen = dataset.convert_to_string(text_fifteen)
        text = dataset.convert_to_string(text)
        text_beta_nine = dataset.convert_to_string(text_beta_nine)
        text_beta_five = dataset.convert_to_string(text_beta_five)
        text_beta_one = dataset.convert_to_string(text_beta_one)

        return text_fifteen, text, text_beta_nine, text_beta_five, text_beta_one


def train(config):
    print(config.train_steps)
    device = torch.device(config.device)

    dataset = TextDataset(config.txt_file, config.seq_length)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)


    model = TextGenerationModel(config.batch_size, config.seq_length, dataset.vocab_size, config.lstm_num_hidden,
                                config.lstm_num_layers,
                                device)


    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate)

    generated_text = []
    for epochs in range(10):
        for step, (batch_inputs, batch_targets) in enumerate(data_loader):

            # Only for time measurement of step through network
            t1 = time.time()

            x = torch.stack(batch_inputs, dim=1).to(device)

            # one hot
            encodded_size = list(x.shape)
            encodded_size.append(dataset.vocab_size)
            one_hot = torch.zeros(encodded_size, device=x.device)
            one_hot.scatter_(2, x.unsqueeze(-1), 1)

            targets = torch.stack(batch_targets, dim=1).to(device)

            #######################################################
            predictions = model.forward(one_hot)
            loss = criterion(predictions.transpose(2, 1), targets)
            loss.backward()
            #######################################################

            optimizer.step()
            optimizer.zero_grad()

            loss = loss.item()

            size = targets.shape[0] * targets.shape[1]
            accuracy = torch.sum(predictions.argmax(dim=2) == targets).to(torch.float32) / size
            # Just for time measurement
            t2 = time.time()
            examples_per_second = config.batch_size/float(t2-t1)

            if step % config.print_every == 0:

                print("examples per sec " + str(examples_per_second)+" step "+str(step)+" accuracy "+str(accuracy.item()) +" loss "+str(loss))
                # print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                #       "Accuracy = {:.2f}, Loss = {:.3f}".format(
                #         datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                #         config.train_steps, config.batch_size, examples_per_second,
                #         accuracy, loss
                # ))

                # Generate some sentences by sampling from the model
                random_seed = torch.randint(low=0, high=dataset.vocab_size, size=(1, 1), dtype=torch.long, device=device)

                text_fifteen, text, temp_nine, temp_five, temp_one = generator(model=model, seed=random_seed, length=config.seq_length, dataset=dataset)

                generated_text.append(text_fifteen)
                generated_text.append(text)
                generated_text.append(temp_nine)
                generated_text.append(temp_five)
                generated_text.append(temp_one)

                print("temp 1.5: " + generated_text[-5])
                print("temp 1: " + generated_text[-4])
                print("temp 0.9: " + generated_text[-3])
                print("temp 0.5: " + generated_text[-2])
                print("temp 0.2: " + generated_text[-1])
                print("")

                file = open("generated.txt", "a")
                file.write("beta 1.5: " + generated_text[-5] + "\n")
                file.write("beta 1: " + generated_text[-4] + "\n")
                file.write("beta 0.9: " + generated_text[-3] + "\n")
                file.write("beta 0.5: " + generated_text[-2] + "\n")
                file.write("beta 0.2: " + generated_text[-1] + "\n")
                file.write("")
                file.close()

            if step == config.sample_every:
                # Generate some sentences by sampling from the model
                pass

            if step == 30000:
                # If you receive a PyTorch data-loader error, check this bug report:
                # https://github.com/pytorch/pytorch/pull/9655
                break


    print('Done training.')


 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, required=False, default='Book.txt', help="Path to a .txt file to train on")
    parser.add_argument('--output_file', type=str, required=False, default='generated.txt', help="Path to a .txt file to generate text")
    parser.add_argument('--seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')

    # It is not necessary to implement the following three params, but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000, help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0, help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=30000, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=2000, help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=100, help='How often to sample from the model')

    parser.add_argument('--device', type=str, default="cpu", help="Training device 'cpu' or 'cuda:0'")

    config = parser.parse_args()

    # Train the model
    train(config)
