# model.py
# MazeRecurrentModel

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from rnn.base_model import MazeBaseModel


# implement len>1 to remember te path
class MazeRecurrentModel(MazeBaseModel):
    def __init__(self, mode_type="RNN", input_size=7, hidden_size=128, num_layers=2, output_size=4):
        """
        Initializes the MazeRecurrentModel.

        Parameters:
            mode_type (str): Type of recurrent layer to use. One of "RNN", "GRU", or "LSTM".
            input_size (int): Number of input features.
            hidden_size (int): Number of hidden units.
            num_layers (int): Number of recurrent layers.
            output_size (int): Number of output features.
        """
        super(MazeRecurrentModel, self).__init__()
        self.mode_type = mode_type.upper()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        if self.mode_type == "RNN":
            self.recurrent = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                                    batch_first=True)
        elif self.mode_type == "GRU":
            self.recurrent = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                                    batch_first=True)
        elif self.mode_type == "LSTM":
            self.recurrent = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                                     batch_first=True)
        else:
            raise ValueError("Invalid mode_type. Expected one of 'RNN', 'GRU', or 'LSTM'.")

        self.fc_dir = nn.Linear(hidden_size, 4)
        self.fc_exit = nn.Linear(hidden_size, 1)

        self.model_name = self.mode_type  # Set the model name based on the mode_type

        self._initialize_weights()

        self.fig, self.ax = plt.subplots()
        self.img = None

    def _initialize_weights(self):
        for name, param in self.recurrent.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x, return_activations=False, return_logits=False):
        """
        Forward pass for the model.
        x: Tensor of shape [batch_size, seq_length, input_size]
        return_activations (bool): If True, also return the recurrent output (hidden states)
        Returns:
            logits (Tensor): [batch_size, seq_length, output_size]
            activations (Tensor): [batch_size, seq_length, hidden_size] (optional)
        """
        batch_size = x.size(0)
        device = x.device
        self.last_input = x  # for collision penalty

        if self.mode_type == "LSTM":
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
            out, _ = self.recurrent(x, (h0, c0))  # out shape: [batch, seq_len, hidden_size]
        else:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
            out, _ = self.recurrent(x, h0)

        logits_dir = self.fc_dir(out)  # shape: [B, T, 4]
        logit_exit = self.fc_exit(out).squeeze(-1)  # shape: [B, T]
        if return_activations:
            return (logits_dir, logit_exit), out
        return logits_dir, logit_exit
