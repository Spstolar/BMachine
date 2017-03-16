import numpy as np


class BoltzmannMachine(object):
    def __init__(self, input_size, hidden_size, output_size):
        self.total_nodes = input_size + hidden_size + output_size
        self.state = np.random.randint(0, 2, self.total_nodes, dtype=bool)  # Begin with a random Boolean State.
        self.weights = np.random.normal(0, 1, size=(self.total_nodes,self.total_nodes))  # Random weights from N(0,1).
        self.disconnect = np.random.randint(0, 2, (self.total_nodes,self.total_nodes))  # Choose weights to zero out.
        self.weights = self.disconnect * self.weights  # Zero out the weights chosen by disconnect. Point-wise mult.

    def print_current_state(self):
        print self.state

BM = BoltzmannMachine(2, 4, 2)
BM.print_current_state()

