import numpy as np
import time

'''
Parameters to modulate:
    Learning rate = how much to change the weights by for each batch
    Batch size = how many examples to use for weight change
    Coactivity sweeps = how many sweeps to compute coactivity for given weights
    Readout sweeps = how many sweeps to run before getting a readout
    Size of training set = how many training examples you'll use
    iterations = how many times to run through the training set
'''

def sigmoid(input_comb):
    return 1.0 / (1 + np.exp(-input_comb))


def rand_bern(length):
    # Return a random vector of -1s and 1s.
    rand_vec = np.random.randint(0, 2, length, dtype=int)  # Begin with a random 0-1 draw.
    return (rand_vec - .5) * 2  # Convert to -1, +1 state.


def rand_bern_with_thresh(length, fix1, fix2):
    # Return a random vector of -1s and 1s.
    rand_vec = np.random.randint(0, 2, length, dtype=int)  # Begin with a random 0-1 draw.
    rand_vec[fix1] = 1
    rand_vec[fix2] = 1
    return (rand_vec - .5) * 2  # Convert to -1, +1 state.


def convert_binary_to_pm1(matrix):
    """
    Convert a 0/1 matrix to a -1/1 matrix.
    :param matrix: A binary matrix.
    :return: Converted matrix.
    """
    converted_matrix = (matrix - 0.5) * 2
    return converted_matrix


class BoltzmannMachine(object):
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size + 1
        self.hidden_size = hidden_size + 1
        self.output_size = output_size
        self.total_nodes = self.input_size + self.hidden_size + self.output_size
        self.hidden_ind = self.input_size  # coordinate index where the hidden layer STARTS
        self.out_ind = self.input_size + self.hidden_size  # coordinate index where the output layer STARTS
        self.input_thresh = self.hidden_ind - 1
        self.hidden_thresh = self.out_ind - 1
        self.hidden_nodes = np.arange(self.hidden_ind, self.hidden_thresh)
        self.out_nodes = np.arange(self.out_ind, self.total_nodes)
        self.clamped_visit_list = self.hidden_nodes
        self.unclamped_visit_list = np.hstack((self.hidden_nodes, self.out_nodes))
        
        self.state = rand_bern_with_thresh(self.total_nodes, self.input_thresh, self.hidden_thresh)

        self.weights = self.create_random_weights()
        self.correct_weights()

        self.in_to_hidden = np.ix_(np.arange(self.hidden_ind), self.hidden_nodes)
        self.hidden_to_hidden = np.ix_(self.hidden_nodes, self.hidden_nodes)
        self.hidden_to_out = np.ix_(np.arange(self.hidden_ind,self.out_ind), self.out_nodes)

        self.batch_size = 500
        self.inc = 400
        self.learning_rate = .005
        self.rate = self.learning_rate

        self.history = self.state
        self.sweeps = 100
        self.stabilization = np.zeros((self.sweeps, self.total_nodes))
        self.threshold = .01
        self.energy_history = np.zeros(200)

    def print_current_state(self):
        print self.state

    def state_energy(self):
        """
        Computes the current energy of the system.
        :return: Current system energy.
        """
        agreement_matrix = np.outer(self.state, self.state)  # The (i,j) entry is 1 if i,j agree, else -1
        energy_contributions = agreement_matrix * self.weights  # Element-wise product.
        energy = 0.5 * np.sum(energy_contributions)  # Leaving off bias.
        return energy

    def state_prob(self):
        """
        The (non-normalized) probability of this configuration. Does the whole calculation rather than just over some
        affected subsets.
        :return: conditional probability of this
        """
        return np.exp(-self.state_energy())

    def conditional_prob(self, node):
        lin_sum_neighbors = np.dot(self.weights[node, :], self.state)
        return sigmoid(lin_sum_neighbors)

    def update(self, node):
        """
        Probabilistically update a single node fixing all others.
        :param node: The number of the node to update.
        :return: Change the state of the node according to the probabilities of the two alternatives.
        """
        plus_prob = self.conditional_prob(node)    # P( x_j = 1 |  all other node states)
        coin_flip = np.random.binomial(1, plus_prob)
        result = 2*(coin_flip - .5)  # Convert biased coin flip to -1 or 1.
        self.state[node] = result

    def simultaneous_update(self, mle=0):
        """
        Update ALL nodes at once. This currently will update everything, including the input/output nodes.
        :param mle: Whether to pick the most likely state of each node (1), or to use the probabilistic update (0).
        :return: Updates the state vector.
        """
        new_state = self.state
        if mle == 1:
            for node in range(0, self.total_nodes):
                new_state[node] = self.mle_update(node, 0)
            self.state = new_state
        elif mle == 0:
            for node in range(0, self.total_nodes):
                plus_prob = self.conditional_prob(node)  # P( x_j = 1 |  all other node states)
                coin_flip = np.random.binomial(1, plus_prob)
                result = 2 * (coin_flip - .5)  # Convert biased coin flip to -1 or 1.
                new_state[node] = result
            self.state = new_state
    
    def mle_update(self, node, alter=1):
        """
        Update a node to it's highest probability state.
        :param node: Which node to update.
        :param alter: Actually change the node (1), or just return the most likely state (0).
        :return: The most likely state for that given node.
        """
        if alter == 1:
            if self.conditional_prob(node) > .5:
                self.state[node] = 1
            else:
                self.state[node] = -1
        elif alter == 0:
            if self.conditional_prob(node) > .5:
                return 1
            else:
                return -1

    def run_machine(self, sweep_num, stabilized=0):
        """
        For sweep_num passes or until it stabilizes, update each of the nodes, except the inputs and thresholds.
        :param sweep_num: A maximum number of times to update each node.
        :param stabilized: Update the machine until it stabilizes (0), or just run the machine the given amount (1).
        :return:
        """
        for sweep in range(sweep_num):
            np.random.shuffle(self.unclamped_visit_list)
            for node in self.unclamped_visit_list:
                self.update(node)
            if stabilized == 0:
                if self.stabilization_check(sweep) == 1:
                    break
            # if stabilized == 1:
            #     self.history = np.vstack((self.history, self.state))
            #     self.energy_history[sweep] = self.state_energy()

    def stabilization_check(self, sweep):
        """
        Check to see if the machine has stabilized by this sweep.
        :param sweep: Which sweep of updates this is checking after.
        :return: Whether or not stabilization of the mean activation has occurred for 90% or more of the nodes.
        """
        prev_mean = self.empirical_mean()
        self.history = np.vstack((self.history, self.state))
        current_mean = self.empirical_mean()
        difference = np.abs(current_mean - prev_mean)
        self.stabilization[sweep, :] = np.less(difference, self.threshold)
        minimum_stabilized = np.floor(self.total_nodes * .9)
        if (np.sum(self.stabilization[sweep, :]) > minimum_stabilized) & (sweep > 100):
            # print sweep
            # print self.stabilization[sweep, :]
            return 1
        else:
            return 0

    def create_random_weights(self):
        weights = np.random.uniform(-1, 1, size=(self.total_nodes, self.total_nodes))  # Random weights ~ U([-1,1])
        weights = np.triu(weights, k=1)  # discard lower diagonal terms (and the diagonal to avoid self-connections)
        weights = weights + weights.T  # make the weights symmetric
        return weights
    
    def correct_weights(self):
        self.weights[:self.hidden_ind, :self.hidden_ind] = 0  # forbids connections INP <-> INP
        self.weights[-self.output_size:, -self.output_size:] = 0  # forbids connections OUT <-> OUT
        self.weights[-self.output_size:, :self.input_size] = 0  # forbids connections IN -> OUT
        self.weights[:self.input_size, -self.output_size:] = 0  # forbids connections IN <- OUT
        self.weights[self.hidden_thresh, :self.out_ind] = 0  # forbid hidden_thresh -> hidden & in
        self.weights[:self.out_ind, self.hidden_thresh] = 0  # forbid hidden -> hidden_thresh
        np.fill_diagonal(self.weights, 0)
        # self.weights[self.hidden_ind:, :self.hidden_ind] = 0
        # self.weights[:self.hidden_ind, self.hidden_ind:] = 0

    def check_weights(self):
        w = 0
        w += np.sum(self.weights[:self.hidden_ind, :self.hidden_ind])
        w += np.sum(self.weights[-self.output_size:, -self.output_size:])
        w += np.sum(self.weights[-self.output_size:, :self.input_size])
        w += np.sum(self.weights[:self.input_size, -self.output_size:])
        w += np.sum(self.weights[self.hidden_thresh, self.hidden_ind:self.out_ind])
        w += np.sum(self.weights[self.hidden_ind:self.out_ind, self.hidden_thresh])
        w += np.sum(self.weights.diagonal())
        print 'Sum of the non-connecting weights: ' + str(w)

    def empirical_mean(self, history=0):
        """
        Return the average state of each node for a given history.
        :param history: A record of the state of the machine over a given number of sweeps, defaults to the object
        history.
        :return: The mean activation of each node.
        """
        if history == 0:
            history = self.history
        return np.mean(history, axis=0)
        
    def clamped_run(self, in_state, out_state, sweep_num=1):
        """
        Runs the machine while forcing the input nodes and output nodes to stay the same.
        :param in_state: What the input node states are.
        :param out_state: What the output node states are.
        :param sweep_num: How many times to update the non-clamped nodes.
        :return: Updates the hidden states.
        """
        self.state[:self.input_thresh] = in_state
        self.state[self.out_ind:] = out_state
        for sweep in range(sweep_num):
            np.random.shuffle(self.clamped_visit_list)
            for node in self.clamped_visit_list:
                self.update(node)
        
    def unclamped_run(self, in_state, sweep_num=1):
        """
        Runs the machine while forcing the input nodes to stay the same.
        :param in_state: What the input node states are.
        :param sweep_num: How many times to update the non-clamped nodes.
        :return: Updates the hidden and output states.
        """
        self.state[:self.input_thresh] = in_state
        for sweep in range(sweep_num):
            np.random.shuffle(self.unclamped_visit_list)
            for node in self.unclamped_visit_list:
                self.update(node)

    def clamped_run_mle(self, in_state, out_state):
        # Update the machine using the maximum likelihood states. 
        self.state[:self.input_thresh] = in_state
        self.state[self.out_ind:] = out_state
        for node in self.clamped_visit_list:
            self.mle_update(node)
            
    def unclamped_run_mle(self, in_state):
        self.state[:self.input_thresh] = in_state
        for node in self.clamped_visit_list:
            self.mle_update(node)
                    
    def training(self, example_set, iterations):
        """
        Go through the entire example set for the given number of iterations. For each iteration, you divide the set
        into batches and then pass each batch to the batch_process method to update the weights. After each iteration
        you update the rate at which the weights are changed.
        :param example_set: A set of unlabelled examples.
        :param iterations: How many times to go through the set for the learning.
        :return: The machine's weights are updated.
        """
        # Compute how to go through the batches.
        batch_size = self.batch_size
        inc = self.inc  # This is to allow overlap between batches, let it be about 80% of the batch size.
        set_size = example_set.shape[0]  # How many examples.
        batches_per_iteration = int(set_size / inc) + 1  # How many batches will be needed.
        record_mse = 0
        ramse = np.zeros(iterations*batches_per_iteration)
        testing = 1

        for it in range(iterations):
            np.random.permutation(example_set)
            print "Iteration: " + str(it)
            last_batch_ind = 0  # initialize variable for row index of last batch
            for batch_num, b in enumerate(range(0, set_size - inc, inc)):
                batch = example_set[b:b + batch_size, :]
                # To do learning rate decay change by batch number:
                num_batches_seen = batch_num + batches_per_iteration * it
                if testing == 1:
                    print 'Working on batch ' + str(num_batches_seen)
                    print 'Threshold units are at ' + str(self.state[self.input_thresh]) + ' and ' + str(self.state[self.hidden_thresh])
                    print 'Current weights: ' + str(self.weights)
                    print 'Current sum of weights: ' + str(np.sum(self.weights))
                    self.check_weights()

                self.rate = self.learning_rate / (num_batches_seen + 1)
                self.batch_process(batch)
                if record_mse == 1:
                    ramse[num_batches_seen] = self.average_rmse(batch)  # Compute the batch root mean square error.
                last_batch_ind = b

            # Manually calculate last batch. It includes some of the first and some of the last examples.
            last_batch_ind += inc
            wrap_around_ind = batch_size - (set_size - last_batch_ind)
            batch = np.vstack((example_set[last_batch_ind:, :], example_set[:wrap_around_ind, :]))
            self.batch_process(batch)
            if record_mse == 1:
                ramse[batches_per_iteration * (it + 1)-1] = self.average_rmse(batch)

            # self.rate = self.learning_rate / (1.0 + it)  # If we want to decrease the rate more slowly.
        if record_mse == 1:
            print ramse
            np.save('root_avg_mse.npy', ramse)

    def batch_process(self, batch):
        """
        Take in a batch of examples without class labels to be fed into autoencoder training.
        Go through each example in the batch, compute coactivity for clamped and unclamped runs adding to a running
        total that you then average to change the weights.
        :param batch: a matrix whose rows are examples.
        :return: Changes the weights of the machine.
        """
        batch_size = self.batch_size

        # 3 submatrices for storing coactivity across the batch.
        b_clamped_ih = np.zeros((self.input_size,self.hidden_size - 1))  # minus 1 since threshold doesn't connect
        b_clamped_hh = np.zeros((self.hidden_size - 1, self.hidden_size - 1))  # minus 1 since threshold doesn't connect
        b_clamped_ho = np.zeros((self.hidden_size, self.output_size))

        b_unclamped_ih = np.zeros((self.input_size,self.hidden_size - 1))
        b_unclamped_hh = np.zeros((self.hidden_size - 1, self.hidden_size - 1))
        b_unclamped_ho = np.zeros((self.hidden_size, self.output_size))

        dw = np.zeros((self.total_nodes, self.total_nodes))

        for ex in range(batch_size):
            # First clamp down the input nodes and output nodes and compute coactivity.
            self.state = rand_bern_with_thresh(self.total_nodes, self.input_thresh, self.hidden_thresh)
            self.clamped_run(batch[ex, :], batch[ex, :])
            ex_c_ih, ex_c_hh, ex_c_ho = self.coactivity(clamped=1, sweeps=1)  # Store the coactivities for this example.
            b_clamped_ih += ex_c_ih
            b_clamped_hh += ex_c_hh
            b_clamped_ho += ex_c_ho

            # Next clamp down just the input nodes and compute coactivity.
            self.state = rand_bern_with_thresh(self.total_nodes, self.input_thresh, self.hidden_thresh)
            self.unclamped_run(batch[ex, :])
            ex_u_ih, ex_u_hh, ex_u_ho = self.coactivity(clamped=0, sweeps=1)
            b_unclamped_ih += ex_u_ih
            b_unclamped_hh += ex_u_hh
            b_unclamped_ho += ex_u_ho

        b_clamped_ih, b_clamped_hh, b_clamped_ho = b_clamped_ih, b_clamped_hh, b_clamped_ho
        b_unclamped_ih, b_unclamped_hh, b_unclamped_ho = b_unclamped_ih, b_unclamped_hh, b_unclamped_ho

        dw[self.in_to_hidden] = b_clamped_ih - b_unclamped_ih
        dw[self.hidden_to_hidden] = b_clamped_hh - b_unclamped_hh
        dw[self.hidden_to_out] = b_clamped_ho - b_unclamped_ho
        dw /= batch_size
        dw = (dw + dw.T)/2  # Since weights are symmetric

        self.weights += self.rate * dw  # Not sure if this should be minus. TODO: find the correct rule.
        self.correct_weights()  # Lazy correction.
                
    def coactivity(self, clamped=1, sweeps=1):
        """
        Computed the coactivity of each node pair averaged over a given number of sweeps.
        :param clamped: If 0, then do unclamped updating. Otherwise, clamp output nodes.
        :param sweeps: how many times to do a full update and compute the coactivities.
        :return: A matrix of coactivity.
        """
        # coactivity_matrix = np.zeros((self.total_nodes, self.total_nodes))
        c_in_to_hidden = np.zeros((self.input_size,self.hidden_size - 1))  # minus 1 since threshold doesn't connect
        c_hidden = np.zeros((self.hidden_size - 1, self.hidden_size - 1))  # minus 1 since threshold doesn't connect
        c_hidden_to_out = np.zeros((self.hidden_size, self.output_size))

        for s in range(0, sweeps):
            c_in_to_hidden += np.outer(self.state[:self.hidden_ind], self.state[self.hidden_nodes])
            c_hidden += np.outer(self.state[self.hidden_nodes], self.state[self.hidden_nodes])
            c_hidden_to_out += np.outer(self.state[self.hidden_ind:self.out_ind], self.state[self.out_nodes])

            if clamped == 0 & sweeps > 1:
                self.unclamped_run(self.state[:self.hidden_ind])
            elif clamped == 1 & sweeps > 1:
                self.clamped_run(self.state[:self.hidden_ind], self.state[self.out_ind:])

        # coactivity_matrix[self.in_to_hidden] = c_in_to_hidden
        # coactivity_matrix[self.hidden_to_hidden] = c_hidden
        # coactivity_matrix[self.hidden_to_out] = c_hidden_to_out
        # coactivity_matrix = (coactivity_matrix + coactivity_matrix.T) / 2.0

        return (c_in_to_hidden/ sweeps), (c_hidden / sweeps), (c_hidden_to_out / sweeps)
        # TODO: Check how to compute coactivity during training.
    
    def read_output(self, input_state, print_out=1):
        # Need to fix an input and then run the machine till it has stabilized.
        # Once stabilized, we can return both the maximizer state as well as the
        # averages for 100 or so states.
        sweep_num = 10
        self.unclamped_run(input_state, sweep_num)
        output = np.zeros(self.output_size, dtype=float)
        post_stab_sweeps = 10
        for i in range(post_stab_sweeps):
            self.unclamped_run(input_state)
            output += self.state[self.out_ind:]
        average_output = output / float(post_stab_sweeps)
        output_state = np.sign(average_output)
        if print_out == 1:
            # print (output_state == input_state)
            print average_output
            print np.sum(np.equal(output_state, input_state))
        return output_state  # TODO: Decide on the exact rule for reading off the state.

    def read_hidden(self, example_set):
        num_ex = example_set.shape[0]
        num_features = example_set.shape[1]
        hidden_states = np.zeros((num_ex, self.hidden_size - 1))
        for j in range(num_ex):
            if (j % 100) == 0:
                print 'Example ' + str(j)
            sweep_num = 10
            input_state = example_set[j,:]
            self.unclamped_run(input_state, sweep_num)
            hidden = np.zeros(self.hidden_size - 1, dtype=float)
            post_stab_sweeps = 10
            for i in range(post_stab_sweeps):
                self.unclamped_run(input_state)
                hidden += self.state[self.hidden_ind:self.hidden_thresh]
            average_output = hidden / float(post_stab_sweeps)
            hidden_states[j,:] = np.sign(average_output)
        np.save('hidden_activations', hidden_states)



    def average_rmse(self, example_set):
        """
        Computes the difference between computed output and input averaged over the examples.
        :param example_set:
        :return: The square root of the average mean square error.
        """
        num_ex = example_set.shape[0]
        error = np.zeros(example_set.shape[1])
        for i in range(num_ex):
            input_state = example_set[i,:]
            error += np.abs(input_state - self.read_output(input_state, 0))
        total_error = np.sum(error)
        return np.sqrt(total_error / float(num_ex))

    # def modulate_params(self):


def simple_example():
    examples = np.load('toy_example_set.npy')
    np.random.permutation(examples)

    input_size = 3
    BM = BoltzmannMachine(input_size, 3, input_size)

    BM.weights = np.ones((11,11))
    print BM.weights
    BM.correct_weights()
    print BM.weights

    BM.run_machine(BM.sweeps)
    BM.training(examples, 3)

    ones_vec = np.ones(5)
    neg_ones_vec = -np.ones(5)

    vec_1 = np.hstack((ones_vec, neg_ones_vec))
    vec_2 = np.hstack((ones_vec, ones_vec))
    vec_3 = np.hstack((neg_ones_vec, ones_vec))
    vec_4 = np.hstack((neg_ones_vec, neg_ones_vec))

    print BM.read_output(vec_1)
    print BM.read_output(vec_2)
    print BM.read_output(vec_3)
    print BM.read_output(vec_4)

    print 'Random vectors: '
    score = 0
    for r in range(10):
        rand = rand_bern(10)
        output_state = BM.read_output(rand)
        score += np.sum(np.equal(output_state, rand))
        print 'In: ' + str(rand) + 'Out: ' + str(output_state)
    print str(score) + 'out of 100'


def main():
    start_time = time.time()

    examples = np.load('testSetSimple.npy')
    examples = convert_binary_to_pm1(examples)

    input_size = examples.shape[1]

    BM = BoltzmannMachine(input_size, 300, input_size)

    BM.training(examples,3)

    np.save('trained_weights.npy',BM.weights)

    end_time = time.time()

    print "Training time: "

    print end_time - start_time


def read():
    start_time = time.time()
    BM.weights = np.load('trained_weights.npy')
    print "Starting Hidden Layer Reading"
    BM.read_hidden(examples)

    end_time = time.time()

    print "Reading time: "
    print end_time - start_time

if __name__ == "__main__":
    main()
    # read()
