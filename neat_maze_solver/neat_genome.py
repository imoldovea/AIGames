import random
from enum import Enum
from typing import Dict, Tuple


# Define node types for neural network genes: input, hidden, output
class NodeType(str, Enum):
    INPUT = "input"
    HIDDEN = "hidden"
    OUTPUT = "output"


# NodeGene represents a single neuron/gene in the network
class NodeGene:
    def __init__(self, id: int, type: NodeType):
        self.id = id
        self.type = type

    def __repr__(self):
        return f"NodeGene(id={self.id}, type={self.type})"


# ConnectionGene represents a connection/gene between two neurons in the network
class ConnectionGene:
    def __init__(self, src: int, tgt: int, weight: float, enabled: bool, innovation: int):
        self.src = src  # Source node id
        self.tgt = tgt  # Target node id
        self.weight = weight  # Connection weight
        self.enabled = enabled  # Whether the connection is enabled
        self.innovation = innovation  # Unique innovation number

    def __repr__(self):
        return (
            f"ConnGene({self.src}->{self.tgt}, w={self.weight:.3f}, "
            f"{'on' if self.enabled else 'off'}, innov={self.innovation})"
        )


# Genome class encapsulates the entire genetic encoding for a neural network
class Genome:
    def __init__(self):
        self.nodes: Dict[int, NodeGene] = {}  # All nodes by id
        self.connections: Dict[Tuple[int, int], ConnectionGene] = {}  # Connections keyed by (src, tgt) tuple

    def add_node(self, id: int, type: NodeType):
        """Add a single node to the genome."""
        self.nodes[id] = NodeGene(id, type)

    def add_connection(self, src: int, tgt: int, weight: float, innovation: int):
        """Add a connection between nodes with a weight and innovation id."""
        self.connections[(src, tgt)] = ConnectionGene(src, tgt, weight, True, innovation)

    def mutate_weights(self, sigma=0.5, prob=0.8):
        """Randomly alter connection weights with a certain probability and noise."""
        for conn in self.connections.values():
            if random.random() < prob:
                conn.weight += random.gauss(0, sigma)

    def mutate_toggle_connection(self):
        """Randomly enable or disable (toggle) a single connection."""
        if not self.connections:
            return
        conn = random.choice(list(self.connections.values()))
        conn.enabled = not conn.enabled

    def mutate_add_connection(self, innovation_counter, max_attempts=10):
        potential = list(self.nodes.keys())
        for _ in range(max_attempts):
            src = random.choice(potential)
            tgt = random.choice(potential)
            if src == tgt or (src, tgt) in self.connections:
                continue
            if self.nodes[src].type == NodeType.OUTPUT:
                continue
            if self.nodes[tgt].type == NodeType.INPUT:
                continue
            innovation = innovation_counter(src, tgt)  # Corrected
            weight = random.uniform(-1, 1)
            self.add_connection(src, tgt, weight, innovation)
            break

    def mutate_add_node(self, innovation_counter):
        enabled_conns = [c for c in self.connections.values() if c.enabled]
        if not enabled_conns:
            return
        conn = random.choice(enabled_conns)
        conn.enabled = False

        new_node_id = max(self.nodes.keys()) + 1
        self.add_node(new_node_id, NodeType.HIDDEN)

        innov1 = innovation_counter(conn.src, new_node_id)  # Corrected
        innov2 = innovation_counter(new_node_id, conn.tgt)  # Corrected
        self.add_connection(conn.src, new_node_id, 1.0, innov1)
        self.add_connection(new_node_id, conn.tgt, conn.weight, innov2)

    def copy(self):
        """Create a deep copy (clone) of the current genome."""
        clone = Genome()
        for id, node in self.nodes.items():
            clone.nodes[id] = NodeGene(id, node.type)
        for key, conn in self.connections.items():
            clone.connections[key] = ConnectionGene(
                conn.src, conn.tgt, conn.weight, conn.enabled, conn.innovation
            )
        return clone

    def __repr__(self):
        return f"Genome(nodes={len(self.nodes)}, conns={len(self.connections)})"
