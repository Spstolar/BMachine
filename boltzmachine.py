import numpy as np


class BoltzmannMachine(object):
    def __init__(self, input_size, hidden_size, output_size):
        self.total_nodes = input_size + hidden_size + output_size

        self.state = np.random.randint(0, 2, self.total_nodes, dtype=int)  # Begin with a random 0-1 draw.
        self.state = (self.state - .5)*2  # Convert to -1, +1 state.

        self.weights = self.create_random_weights()

    def print_current_state(self):
        print self.state

    def state_energy(self):
        agreement_matrix = np.outer(self.state, self.state)  # The (i,j) entry is 1 if i,j agree, else -1
        energy_contributions = agreement_matrix * self.weights
        energy = np.sum(energy_contributions)  # Leaving off bias for now.
        return energy

    def create_random_weights(self):
        weights = np.random.normal(0, 1, size=(self.total_nodes, self.total_nodes))  # Random weights from N(0,1).
        weights = (weights + weights.T) / 2  # To make it symmetric.

        # The next few lines create a symmetric array of weights to zero out, and remove self-connections.
        disconnect = np.random.randint(0, 2, (self.total_nodes, self.total_nodes))  # Choose weights to zero out.
        disconnect_lower_triangle = np.tril(disconnect)
        disconnect = disconnect_lower_triangle + disconnect_lower_triangle.T
        weights = disconnect * weights  # Zero out the weights chosen by disconnect. Point-wise mult.
        return weights

BM = BoltzmannMachine(2, 4, 2)
BM.print_current_state()
print BM.state_energy()

