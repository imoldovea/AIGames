# neat_network.py

import numpy as np


class NEATNetwork:
    """
    NEAT network supporting feedforward AND recurrent topologies.
    For recurrent/cyclic networks, node activations are updated for multiple steps.
    """

    def __init__(self, genome, activation='tanh'):
        self.genome = genome
        self.activation = np.tanh if activation == 'tanh' else lambda x: x
        # Gather enabled connections
        self.connections = [c for c in self.genome.connections.values() if c.enabled]
        self.node_ids = list(self.genome.nodes.keys())
        # Useful to store node types for input/output lookup
        self.input_ids = [nid for nid, node in self.genome.nodes.items() if node.type == 'input']
        self.output_ids = [nid for nid, node in self.genome.nodes.items() if node.type == 'output']
        # Initialize node values for state tracking
        self.values = {nid: 0.0 for nid in self.node_ids}

    def activate(self, inputs, steps=5, initial_state=None):
        """
        Activates the network on given inputs, for 'steps' iterations.
        If there are cycles, activations propagate through the network over time.
        initial_state: dict of previous activations (optional, for memory).
        Returns: output activations as list.
        """
        if len(inputs) != len(self.input_ids):
            raise ValueError("Input vector size does not match number of input nodes.")

        # Initialize all node activations to zero (or given initial state)
        values = {nid: 0.0 for nid in self.node_ids}
        if initial_state is not None:
            values.update(initial_state)
        for nid, val in zip(self.input_ids, inputs):
            values[nid] = val

        # Iteratively update activations
        for _ in range(steps):
            new_values = values.copy()
            for nid in self.node_ids:
                if nid in self.input_ids:
                    continue  # Input nodes stay fixed during this input frame
                s = 0.0
                for conn in self.connections:
                    if conn.tgt == nid and conn.enabled:
                        s += values[conn.src] * conn.weight
                new_values[nid] = self.activation(s)
            values = new_values

        # Store current values as instance attribute for state tracking
        self.values = values

        # Output activations in order of output node ids
        outputs = [values[nid] for nid in self.output_ids]
        return outputs

    def get_state(self):
        """Get current state (for RNN-style memory)."""
        return self.values.copy()
