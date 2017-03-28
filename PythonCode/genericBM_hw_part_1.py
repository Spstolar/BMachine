import numpy as np


def sigmoid(input):
    return 1.0 / (1 + np.exp(-input))

class BoltzmannMachine(object):
    def __init__(self, input_size, hidden_size, output_size):
        self.total_nodes = input_size + hidden_size + output_size
        self.state = np.random.randint(0, 2, self.total_nodes, dtype=int)  # Begin with a random 0-1 draw.
        self.state = (self.state - .5) * 2  # Convert to -1, +1 state.
        self.weights = self.create_random_weights()
        self.history = self.state

    def print_current_state(self):
        print self.state

    def state_energy(self):
        agreement_matrix = np.outer(self.state, self.state)  # The (i,j) entry is 1 if i,j agree, else -1
        energy_contributions = agreement_matrix * self.weights  # Element-wise product.
        energy = np.sum(energy_contributions) / 2  # Leaving off bias for now.
        return energy

    def state_prob(self):
        """
        The (non-normalized) probability of this state. Does the whole calculation rather than just over the
        affected subsets.
        :return: conditional probability of this
        """
        return sigmoid(self.state_energy())

    def conditional_prob(self, node):
        lin_sum_neighbors = np.dot(self.weights[node,:], self.state)
        return sigmoid(lin_sum_neighbors)

    def update(self, node):
        plus_prob = self.conditional_prob(node)    # P( x_j = 1 |  all other node states)
        coin_flip = np.random.binomial(1, plus_prob, 1)
        result = 2*(coin_flip - .5)  # Convert biased coin flip to -1 or 1.
        # print result
        self.state[node] = result

    def run_machine(self, sweep_num):
        visit_list = np.arange(self.total_nodes)  # The array [0 1 ... n-1].
        for sweep in range(sweep_num):
            np.random.shuffle(visit_list)  # Shuffle the array [0 1 ... n-1].
            for node_num in range(self.total_nodes):
                node_to_update = visit_list[node_num]
                self.update(node_to_update)
            self.history = np.vstack((self.history, self.state))

    def create_random_weights(self):
        weights = np.random.uniform(-1, 1, size=(self.total_nodes, self.total_nodes))  # Random weights ~ U([-1,1])
        weights = np.triu(weights, k=1)  # discard lower diagonal terms (and the diagonal to avoid self-connections)
        weights = weights + weights.T  # make the weights symmetric
        return weights

    def empirical_mean(self):
        return np.mean(self.history, axis=0)


BM = BoltzmannMachine(0, 30, 0)

BM.print_current_state()

BM.run_machine(3)

print BM.history

print BM.empirical_mean()
