# model.py
# MazeRecurrentModel

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from base_model import MazeBaseModel


class MazeRecurrentModel(MazeBaseModel):
    def __init__(self, mode_type="RNN", input_size=7, hidden_size=128, num_layers=2, output_size=5):
        """
        Initializes the MazeRecurrentModel.

        Parameters:
            mode_type (str): Type of recurrent layer to use. One of "RNN", "GRU", or "LSTM".
            input_size (int): Number of input features.
            hidden_size (int): Number of hidden units.
            num_layers (int): Number of recurrent layers.
            output_size (int): Number of output features. 4 direction + at_exit preddiction
            predict_exit (bool): Whether to predict exit signals.
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

        self.fc = nn.Linear(hidden_size, output_size)

        # Additional output for exit prediction
        self.exit_predictor = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Single output for binary classification
        )

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

        # Initialize the fully connected layer
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

        for module in self.exit_predictor:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        """
        Forward pass for the model.

        x: Tensor of shape [batch_size, seq_length, input_size].
        Returns:
            Tensor of shape [batch_size, output_size].
        """
        batch_size = x.size(0)
        if self.mode_type == "LSTM":
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
            out, _ = self.recurrent(x, (h0, c0))
        else:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
            out, _ = self.recurrent(x, h0)

        last_hidden = out[:, -1, :]  # Use the last time-step's output
        action_logits = self.fc(last_hidden)
        exit_logits = self.exit_predictor(last_hidden)
        return action_logits, exit_logits.squeeze(-1)  # Return both outputs
