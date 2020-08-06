#!/usr/bin/env python3
"""
student.py

UNSW COMP9444 Neural Networks and Deep Learning

You may modify this file however you wish, including creating
additional variables, functions, classes, etc., so long as your code
runs with the hw2main.py file unmodified, and you are only using the
approved packages.

You have been given some default values for the variables stopWords,
wordVectors(dim), trainValSplit, batchSize, epochs, and optimiser.
You are encouraged to modify these to improve the performance of your model.

The variable device may be used to refer to the CPU/GPU being used by PyTorch.

You may only use GloVe 6B word vectors as found in the torchtext package.
"""

import torch
import torch.nn as tnn
import torch.optim as toptim
from torchtext.vocab import GloVe
# import numpy as np
# import sklearn

###########################################################################
### The following determines the processing of input data (review text) ###
###########################################################################

def preprocessing(sample):
    """
    Called after tokenising but before numericalising.
    """
    import re
    import string
    input = " ".join(sample)
    text = re.sub(r"[^\x00-\x7F]+", " ", input)
    regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]')
    nopunct = regex.sub(" ", text.lower())
    result = nopunct.split(" ")
    sample = list(filter(lambda x: x != '', result))

    return sample

def postprocessing(batch, vocab):
    """
    Called after numericalisation but before vectorisation.
    """
    vocabCount = vocab.freqs
    vocabITOS = vocab.itos
    for sentence in batch:
        for j, word in enumerate(sentence):
            if vocabCount[vocabITOS[word]] < 3:
                sentence[j] = -1
    return batch

stopWords = {}
embed_dim = 200
wordVectors = GloVe(name='6B', dim=embed_dim)

###########################################################################
##### The following determines the processing of label data (ratings) #####
###########################################################################

def convertLabel(datasetLabel):
    """
    Labels (product ratings) from the dataset are provided to you as
    floats, taking the values 1.0, 2.0, 3.0, 4.0, or 5.0.
    You may wish to train with these as they are, or you you may wish
    to convert them to another representation in this function.
    Consider regression vs classification.
    """
    datasetLabel = datasetLabel - 1

    return datasetLabel.long()

def convertNetOutput(netOutput):
    """
    Your model will be assessed on the predictions it makes, which
    must be in the same format as the dataset labels.  The predictions
    must be floats, taking the values 1.0, 2.0, 3.0, 4.0, or 5.0.
    If your network outputs a different representation or any float
    values other than the five mentioned, convert the output here.
    """
    netOutput = torch.argmax(netOutput, dim=1) + 1

    return netOutput.float()

###########################################################################
################### The following determines the model ####################
###########################################################################
class network(tnn.Module):
    """
    Class for creating the neural network.  The input to your network
    will be a batch of reviews (in word vector form).  As reviews will
    have different numbers of words in them, padding has been added to the
    end of the reviews so we can form a batch of reviews of equal length.
    """

    def __init__(self):
        super(network, self).__init__()
        self.classes = 5
        self.hidden_dim = 128
        self.hidden_layers = 3
        self.lstm = tnn.LSTM(embed_dim, hidden_size=self.hidden_dim, num_layers=self.hidden_layers)
        self.linear = tnn.Sequential(
            # tnn.Linear(self.hidden_dim, 256),
            # tnn.ReLU(),
            # tnn.Linear(256, 128),
            # tnn.ReLU(),
            tnn.Linear(128, self.classes),
        )

    def get_last_output(self, output, batch_seq_len):
        last_outputs = torch.zeros((output.shape[0], output.shape[2]))
        for i in range(len(batch_seq_len)):
            last_outputs[i] = output[i][batch_seq_len[i] - 1]
        last_outputs = last_outputs.to(output.device)
        return last_outputs

    def forward(self, input, length):
        embed_input_x_packed = tnn.utils.rnn.pack_padded_sequence(input, length, batch_first=True, enforce_sorted=False)
        encoder_outputs_packed, (hidden, _) = self.lstm(embed_input_x_packed)
        output = self.linear(hidden[-1])
        return output


class loss(tnn.Module):
    """
    Class for creating a custom loss function, if desired.
    You may remove/comment out this class if you are not using it.
    """

    def __init__(self):
        super(loss, self).__init__()
        self.loss = tnn.CrossEntropyLoss()

    def forward(self, output, target):
        return self.loss(output, target)

net = network()
"""
    Loss function for the model. You may use loss functions found in
    the torch package, or create your own with the loss class above.
"""
lossFunc = loss()

###########################################################################
################ The following determines training options ################
###########################################################################

trainValSplit = 0.8
batchSize = 32
epochs = 10
optimiser = toptim.SGD(net.parameters(), lr=0.1)
