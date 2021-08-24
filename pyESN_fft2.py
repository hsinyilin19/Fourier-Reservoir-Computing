import numpy as np
import torch


def correct_dimensions(s, targetlength):
    """
    checks the dimensionality of some numeric argument s, broadcasts it
       to the specified length if possible.

    Args:
        s: None, scalar or 1D array
        targetlength: expected length of s

    Returns:
        None if s is None, else numpy vector of length targetlength
    """
    if s is not None:
        s = np.array(s)
        if s.ndim == 0:
            s = np.array([s] * targetlength)
        elif s.ndim == 1:
            if not len(s) == targetlength:
                raise ValueError("arg must have length " + str(targetlength))
        else:
            raise ValueError("Invalid argument")
    return s


class ESN:
    def __init__(self, n_inputs, n_outputs, n_reservoir=200,
                 spectral_radius=0.95, sparsity=0, noise=0.001, input_shift=None,
                 input_scaling=None, label_scaling=None, label_shift=None, random_state=None, leak = 1, lamb=0):
        """
        Args:
            n_inputs: nr of input dimensions
            n_outputs: nr of output dimensions
            n_reservoir: nr of reservoir neurons
            spectral_radius: spectral radius of the recurrent weight matrix
            sparsity: proportion of recurrent weights set to zero
            noise: noise added to each neuron (regularization)
            input_shift: scalar or vector of length n_inputs to add to each
                        input dimension before feeding it to the network.
            input_scaling: scalar or vector of length n_inputs to multiply
                        with each input dimension before feeding it to the netw.
            label_scaling: factor applied to the target signal
            label_shift: additive term applied to the target signal
            random_state: positive integer seed, np.rand.RandomState object,
                          or None to use numpy's builting RandomState.
        """
        # check for proper dimensionality of all arguments and write them down.
        self.n_inputs = n_inputs
        self.n_reservoir = n_reservoir
        self.n_outputs = n_outputs
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.noise = noise
        self.input_shift = correct_dimensions(input_shift, n_inputs)
        self.input_scaling = correct_dimensions(input_scaling, n_inputs)
        self.label_scaling = label_scaling
        self.label_shift = label_shift
        self.random_state = random_state
        self.s_last = np.zeros(self.n_reservoir)
        self.u_last = np.zeros(self.n_inputs)
        self.leak = leak
        self.lamb = lamb

        # the given random_state might be either an actual RandomState object,
        # a seed or None (in which case we use numpy's builtin RandomState)
        if isinstance(random_state, np.random.RandomState):
            self.random_state_ = random_state
        elif random_state:
            try:
                self.random_state_ = np.random.RandomState(random_state)
            except TypeError as e:
                raise Exception("Invalid seed: " + str(e))
        else:
            self.random_state_ = np.random.mtrand._rand

        self.weights()

    def weights(self):
        # initialize weights with a random matrix centered around zero:
        W = self.random_state_.rand(self.n_reservoir, self.n_reservoir) - 0.5

        # delete the fraction of connections given by (self.sparsity):
        W[self.random_state_.rand(*W.shape) < self.sparsity] = 0

        # compute the spectral radius of self.W_in, u) these weights:
        radius = np.max(np.abs(np.linalg.eigvals(W)))

        # rescale them to reach the requested spectral radius:
        self.W = W * (self.spectral_radius / radius)

        # random input weights:
        self.W_in = self.random_state_.rand(self.n_reservoir, self.n_inputs) * 2 - 1

    # def update(self, s, u):
    #     """
    #     update state vector: s(t) = W * s(t-1) + W_in * u(t-1)
    #     u = input signal
    #     s = (reservoir) state
    #     """
    #     p = np.dot(self.W, s) + np.dot(self.W_in, u)
    #
    #     return np.tanh(p) + self.noise * (self.random_state_.rand(self.n_reservoir) - 0.5)

    def update(self, s, u):
        """
        update state vector: s(t) = W * s(t-1) + W_in * u(t-1)
        u = input signal
        s = (reservoir) state
        """
        p = np.dot(self.W, s) + np.dot(self.W_in, u)

        return self.leak * np.tanh(p) + (1 - self.leak) * s + self.noise * (self.random_state_.rand(self.n_reservoir) - 0.5)
        # return self.leak * np.tanh(p) + self.noise * (self.random_state_.rand(self.n_reservoir) - 0.5)
        # return self.leak * np.tanh(p) + (1 - self.leak) * s

        # return self.leak * self.relu(p) + (1 - self.leak) * s + self.noise * (self.random_state_.rand(self.n_reservoir) - 0.5)

    def relu(self, x):
        y = np.max(0, x)
        return y


    def scaling(self, x, scale, shift):
        """scaling and shift for inputs and labels"""

        if scale is not None:
            x = np.dot(x, np.diag(scale))
        if shift is not None:
            x = x + shift
        return x

    def reverse_scaling(self, x, scale, shift):
        """inverse of the scaling"""
        if shift is not None:
            x = x - shift
        if scale is not None:
            x = x / scale
        return x

    def train(self, u, y):
        """
        train the readout weights (W_out)

        Args:
            u: inputs (training_samples * outputs)
            y: label (training_samples * outputs)

        Returns:
            the prediction, loss and W_out (trained weights)
        """
        # transform shape (x,) into shape (x,1)
        if u.ndim < 2:
            u = np.reshape(u, (len(u), -1))
        if y.ndim < 2:
            y = np.reshape(y, (len(y), -1))

        # scale the input and label signal
        u_scaled = self.scaling(u, self.input_scaling, self.input_shift)
        y_scaled = self.scaling(y, self.label_scaling, self.label_shift)

        # encoding input signal {u(t)} -> {s(t)}
        s = np.zeros((u.shape[0], self.n_reservoir))
        for t in range(1, u.shape[0]):
            s[t, :] = self.update(s[t - 1], u_scaled[t, :])
            #s[t, :] = self.update(s[t - 1], u_scaled[t - 1, :])

        print("ESN training...")

        # learn the weights by solving for final layer weights W_out analytically
        # (this is actually a linear regression, a no-hidden layer neural network with the identity activation function)
        
        # self.W_out = np.dot(np.linalg.pinv(s[:, :]), y_scaled[:, :]).T

        self.W_out = np.linalg.solve((s[:, :].T.dot(s[:, :]) + self.lamb * np.identity(s[:, :].shape[1])), (s[:, :].T.dot(y_scaled[:, :])))

        self.W_out = self.W_out.T


        # record the last (time step) state
        self.s_last = s[-1, :]
        self.u_last = u[-1, :]
        self.y_last = y_scaled[-1, :]

        # apply the learned weights (W_out) to the collected states:
        y_pred = self.reverse_scaling(np.dot(s, self.W_out.T), self.input_scaling, self.input_shift)

        loss = np.sqrt(np.mean(np.sum((y_pred - y)**2, axis=1), axis=0))

        return y_pred, loss, self.W_out, self.s_last, self.y_last

    def predict(self, s_last, u_last, n_samples):
        """
        Apply the learned weights to new input.

        Args:
            u inputs array of dimensions (N_test_samples x n_inputs)
            continuation: if True, start the network from the last training state

        Returns:
            Array of output activations
        """

        s = np.zeros((n_samples, self.n_reservoir))
        y = np.zeros((n_samples, self.n_outputs))

        s[0, :] = s_last
        y[0, :] = u_last

        s[1, :] = self.update(s_last, u_last)
        y[1, :] = np.dot(self.W_out, s[0, :])

        for t in range(n_samples-1):
            s[t + 1, :] = self.update(s[t, :], y[t, :])
            y[t + 1, :] = np.dot(self.W_out, s[t + 1, :])

        return self.reverse_scaling(y, self.label_scaling, self.label_shift)
