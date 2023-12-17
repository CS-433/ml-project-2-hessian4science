import numpy as np
import torch.nn as nn


class NN(nn.Module):
    """
    Simple feedforward neural network.
    """

    def __init__(self, input_shape, hidden, activation='relu', sigmoid_output=True, conv_number=1):
        """
        Initialize the neural network.
        :param input_shape: The shape of the input.
        :param hidden: A list of hidden layer sizes.
        :param activation: The activation function to use.
        :param sigmoid_output: Whether to use a sigmoid activation on the output.
        :param conv_number: The number of convolutional layers to use.
        """
        super().__init__()
        self.seq = nn.Sequential()
        if conv_number > 0:
            self.seq.append(nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=3, stride=1,
                                      padding=1,
                                      bias=True))
            self.seq.append(nn.BatchNorm2d(32))
            self.seq.append(nn.ReLU())
            for i in range(conv_number - 1):
                self.seq.append(nn.Conv2d(in_channels=32 * (2 ** i), out_channels=32 * (2 ** (i + 1)), kernel_size=3,
                                          stride=1,
                                          padding=1,
                                          bias=True))
                self.seq.append(nn.BatchNorm2d(32 * (2 ** (i + 1))))
                self.seq.append(nn.ReLU())
                self.seq.append(nn.MaxPool2d(kernel_size=2))
            input_size = 32 * (2 ** (conv_number - 1)) * ((input_shape[1] // (2 ** (conv_number - 1))) *
                                                          (input_shape[2] // (2 ** (conv_number - 1))))
        else:
            input_size = np.prod(input_shape)
        self.seq.append(nn.Flatten())
        self.seq.append(nn.Linear(input_size, hidden[0]))
        self.seq.append(nn.BatchNorm1d(hidden[0]))
        for i in range(len(hidden) - 1):
            if activation == 'relu':
                self.seq.append(nn.ReLU())
            else:
                self.seq.append(nn.Tanh())
            self.seq.append(nn.Linear(hidden[i], hidden[i + 1]))
            self.seq.append(nn.BatchNorm1d(hidden[i + 1]))
        if sigmoid_output:
            self.seq.append(nn.Sigmoid())

    def forward(self, x):
        """
        Forward pass.
        :param x: The input.
        :return: The output.
        """
        return self.seq(x)

    def _init_weights(self, module):
        module.weight.data.uniform_(-100, 100)
        module.bias.data.zero_()
