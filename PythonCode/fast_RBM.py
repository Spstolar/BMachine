import numpy as np
import time

def sigmoid(input):
    return 1.0 / (1 + np.exp(-input))

def rand_bern(length):
    # Return a random vector of -1s and 1s.
    rand_vec = np.random.randint(0, 2, length, dtype=int)  # Begin with a random 0-1 draw.
    return (rand_vec - .5) * 2  # Convert to -1, +1 state.

class BoltzmannMachine(object):
    def __init__(self, input_size, hidden_size, output_size):
        self.total_nodes = input_size + hidden_size + output_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden_ind = input_size
        self.out_ind = input_size + output_size
        
        self.state = rand_bern(self.total_nodes)
        self.weights = self.create_random_weights()
        self.correct_weights()
        
        self.history = self.state
        self.sweeps = 1000
        self.stabilization = np.zeros((self.sweeps, self.total_nodes))
        self.threshold = .01
        self.energy_history = np.zeros(200)

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
        return np.exp(-self.state_energy())

    def conditional_prob(self, node):
        lin_sum_neighbors = np.dot(self.weights[node,:], self.state)
        return sigmoid(lin_sum_neighbors)

    def update(self, node):
        plus_prob = self.conditional_prob(node)    # P( x_j = 1 |  all other node states)
        coin_flip = np.random.binomial(1, plus_prob, 1)
        result = 2*(coin_flip - .5)  # Convert biased coin flip to -1 or 1.
        # print result
        self.state[node] = result
    
    def mle_update(self,node):
        if self.conditional_prob(node) > .5 :
            self.state[node] = 1    
        else:
            self.state[node] = -1

    def run_machine(self, sweep_num, stabilized=0):
        visit_list = np.arange(self.total_nodes)  # The array [0 1 ... n-1].
        for sweep in range(sweep_num):
            np.random.shuffle(visit_list)  # Shuffle the array [0 1 ... n-1].
            for node_num in range(self.total_nodes):
                node_to_update = visit_list[node_num]
                self.update(node_to_update)
            if stabilized == 0:
                if self.stabilization_check(sweep) == 1:
                    break
            if stabilized == 1:
                self.history = np.vstack((self.history, self.state))
                self.energy_history[sweep] = self.state_energy()

    def stabilization_check(self, sweep):
        prev_mean = self.empirical_mean()
        self.history = np.vstack((self.history, self.state))
        current_mean = self.empirical_mean()
        difference = np.abs(current_mean - prev_mean)
        self.stabilization[sweep, :] = np.less(difference, self.threshold)
        if (np.sum(self.stabilization[sweep, :]) > 27) & (sweep > 100):
            print sweep
            print self.stabilization[sweep, :]
            return 1
        else:
            return 0

    def create_random_weights(self):
        weights = np.random.uniform(-1, 1, size=(self.total_nodes, self.total_nodes))  # Random weights ~ U([-1,1])
        weights = np.triu(weights, k=1)  # discard lower diagonal terms (and the diagonal to avoid self-connections)
        weights = weights + weights.T  # make the weights symmetric
        return weights
    
    def correct_weights(self):
        self.weights[:self.hidden_ind, :self.hidden_ind] = 0
        self.weights[-self.output_size:,-self.output_size:] = 0
        self.weights[self.hidden_ind:,:self.hidden_ind] = 0
        self.weights[:self.hidden_ind, self.hidden_ind:] = 0
        

    def empirical_mean(self):
        return np.mean(self.history, axis=0)
        
    def clamped_run(self, in_state, out_state, sweep_num=1):
        # Runs the machine while forcing the input nodes and output nodes to stay the same. 
        # Updates the hidden states.
        self.state[:self.hidden_ind] = in_state
        self.state[self.out_ind:] = out_state
        visit_list = np.arange(self.hidden_ind, self.out_ind)  
        for sweep in range(sweep_num):
            np.random.shuffle(visit_list)  
            for node_num in range(self.hidden_size):
                node_to_update = visit_list[node_num]
                self.update(node_to_update)
        
    def unclamped_run(self, in_state, sweep_num=1):
        # Runs the machine while forcing the input nodes to stay the same. 
        # Updates the hidden states and output states.
        self.state[:self.hidden_ind] = in_state
        visit_list = np.arange(self.hidden_ind,self.total_nodes)  
        for sweep in range(sweep_num):
            np.random.shuffle(visit_list)  
            for node_num in range(self.hidden_size + self.output_size):
                node_to_update = visit_list[node_num]
                self.update(node_to_update)
		
    def clamped_run_mle(self, in_state, out_state):
        # Update the machine using the maximum likelihood states. 
        self.state[:self.hidden_ind] = in_state
        self.state[self.out_ind:] = out_state
        for node_num in range(self.hidden_size):
            self.mle_update(node_num + self.input_size)
            
    def unclamped_run_mle(self, in_state):
        self.state[:self.hidden_ind] = in_state
        for node_num in range(self.hidden_size + self.output_size):
            self.mle_update(node_num + self.input_size)
                    
    def training(self, example_set, batch_size, num_batches, iterations):
        for it in range(iterations):
            # shuffle data
            for b in range(num_batches):
                batch_process(batch)
                            
    def batch_process(self, batch, batch_size):
        batch_coactivity_clampled = np.zeros((self.total_nodes,self.total_nodes))
        batch_coactivity_unclampled = np.zeros((self.total_nodes,self.total_nodes))
        for ex in range(batch_size):
            # First clamp down the input nodes and output nodes and compute coactivity.
            clamped_run_mle(batch[ex, :self.hidden_ind], batch[ex, self.out_ind:])
            batch_coactivity_clamped = batch_coactivity_clamped + self.coactivity()
            # Next clamp down just the input nodes and compute coactivity.
            unclamped_run(batch[ex, :self.hidden_ind])
            batch_coactivity_unclamped = batch_coactivity_unclamped + self.coactivity()
        dW = (batch_coactivity_clamped - batch_coactivity_unclamped) / batch_size
                
    def coactivity(self):
        return np.outer(self.state, self.state) 
        
    
        

start_time = time.time()

BM = BoltzmannMachine(0, 30, 0)

# BM.print_current_state()

BM.run_machine(BM.sweeps)

# print BM.history

# print BM.empirical_mean()

print BM.stabilization

end_time = time.time()

print end_time - start_time

print BM.stabilization[-1,:]

BM.run_machine(200,1)
print BM.energy_history
