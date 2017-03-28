import numpy as np

class BoltzmannMachine(object):
    def __init__(self, input_size, hidden_size, output_size):
        self.total_nodes = input_size + hidden_size + output_size
        self.state = np.random.randint(0, 2, self.total_nodes, dtype=int)  # Begin with a random 0-1 draw.
        self.state = (self.state - .5) * 2  # Convert to -1, +1 state.
        self.weights = self.create_random_weights()

    def print_current_state(self):
        print self.state

    def state_energy(self):
        agreement_matrix = np.outer(self.state, self.state)  # The (i,j) entry is 1 if i,j agree, else -1
        energy_contributions = agreement_matrix * self.weights
        energy = np.sum(energy_contributions) / 2  # Leaving off bias for now.
        return energy

    def state_prob(self):
        """
        The (non-normalized) probability of this state. Does the whole calculation rather than just over the
        affected subsets.
        :return: P( state ) * Z
        """
        return np.exp(-self.state_energy())

    def update(self, node):
        self.state[node] = 1
        plus_prob = self.state_prob()
        self.state[node] = -1
        minus_prob = self.state_prob()
        plus_prob = plus_prob / (plus_prob + minus_prob)
        # print plus_prob
        coin_flip = np.random.binomial(1, plus_prob, 1)
        result = 2*(coin_flip - .5)  # Convert biased coin flip to -1 or 1.
        # print result
        self.state[node] = result

    def run_machine(self):
        visit_list = np.arange(self.total_nodes)  # The array [0 1 ... n-1].
        np.random.shuffle(visit_list)  # Shuffle the array [0 1 ... n-1].
        for run in range(100):  # why hard-coded 100? What does it correspond to?
            node_to_update = visit_list[run % self.total_nodes]  # I don't understand this
            self.update(node_to_update)

    def create_random_weights(self):
        weights = np.random.uniform(-1, 1, size=(self.total_nodes, self.total_nodes))  # Random weights ~ U([-1,1])
        weights = np.triu(weights, k=1)  # discard lower diagonal terms (and the diagonal to avoid self-connections)
        weights = weights + weights.T  # make the weights symmetric
        return weights

BM = BoltzmannMachine(0, 30, 0)

BM.print_current_state()

for i in range(10):  # Do the update process 10 times, printing the end state each time.
    BM.run_machine()
    BM.print_current_state()