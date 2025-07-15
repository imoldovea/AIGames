# neat_config.py

class InnovationTracker:
    """
    Tracks and assigns unique innovation numbers for new connections.
    Ensures consistent crossover alignment across the population.
    """

    def __init__(self):
        self.counter = 0
        self.connection_history = {}  # (src, tgt): innovation_number

    def get_innovation_number(self, src, tgt):
        key = (src, tgt)
        if key not in self.connection_history:
            self.connection_history[key] = self.counter
            self.counter += 1
        return self.connection_history[key]


# Global singleton tracker for the run
innovation_tracker = InnovationTracker()

# NEAT hyperparameters (easy to modify/tune here)
config = {
    "mutation_rate_weight": 0.8,
    "mutation_rate_add_node": 0.03,
    "mutation_rate_add_connection": 0.05,
    "weight_mutation_std": 0.5,
    "compatibility_threshold": 3.0,
    "crossover_prob": 0.75,
    "num_inputs": 7,  # [N, E, S, W, dx, dy, step_norm]
    "num_outputs": 4,  # [N, E, S, W]
    "max_nodes": 128,
    "activation": "tanh",  # or "relu"
    "population_size": 150
}
