# neuro_net.py
# PyTorch model (input: 7 → output: 4)


import torch
import torch.nn as nn


class NeuroNet(nn.Module):
    """
    A small feedforward neural network for maze decision making.
    Input: 7 features [walls (N,E,S,W), dx, dy, normalized steps]
    Output: 4 logits (N, E, S, W) to be softmaxed
    """

    def __init__(self):
        super(NeuroNet, self).__init__()
        self.fc1 = nn.Linear(7, 10)  # input layer to hidden
        self.fc2 = nn.Linear(10, 4)  # hidden to output layer

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)  # raw logits
        return x

    @staticmethod
    def from_genome(genome: list[float]) -> "NeuroNet":
        """
        Instantiates a NeuroNet and loads weights from a flat genome list.
        """
        model = NeuroNet()
        with torch.no_grad():
            # Flatten all parameter shapes
            params = list(model.parameters())
            shapes = [p.shape for p in params]
            sizes = [p.numel() for p in params]
            total_size = sum(sizes)

            assert len(genome) == total_size, f"Genome length {len(genome)} != total required {total_size}"

            # Assign values
            offset = 0
            for p, shape, size in zip(params, shapes, sizes):
                values = genome[offset:offset + size]
                p.copy_(torch.tensor(values).view(shape))
                offset += size
        return model

    @staticmethod
    def genome_size():
        """Returns the total number of weights/biases needed."""
        dummy = NeuroNet()
        return sum(p.numel() for p in dummy.parameters())
