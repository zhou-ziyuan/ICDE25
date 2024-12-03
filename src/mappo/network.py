import torch.nn as nn
import math
import functools
import torch as ch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from .torch_utils import *

'''
Neural network models for estimating value and policy functions
Contains:
- Initialization utilities
- Value Network(s)
- Policy Network(s)
- Retrieval Function
'''

########################
### INITIALIZATION UTILITY FUNCTIONS:
# initialize_weights
########################

HIDDEN_SIZES = (64, 64)
ACTIVATION = nn.Hardtanh
STD = 2 ** 0.5


def initialize_weights(mod, initialization_type, scale=STD):
    '''
    Weight initializer for the models.
    Inputs: A model, Returns: none, initializes the parameters
    '''
    for p in mod.parameters():
        if initialization_type == "normal":
            p.data.normal_(0.01)
        elif initialization_type == "xavier":
            if len(p.data.shape) >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                p.data.zero_()
        elif initialization_type == "orthogonal":
            if len(p.data.shape) >= 2:
                orthogonal_init(p.data, gain=scale)
            else:
                p.data.zero_()
        else:
            raise ValueError("Need a valid initialization key")


########################
### INITIALIZATION UTILITY FUNCTIONS:
# Generic Value network, Value network MLP
########################

class ValueDenseNet(nn.Module):
    '''
    An example value network, with support for arbitrarily many
    fully connected hidden layers (by default 2 * 128-neuron layers),
    maps a state of size (state_dim) -> a scalar value.
    '''

    def __init__(self, state_dim, init=None, hidden_sizes=(64, 64), activation=None):
        '''
        Initializes the value network.
        Inputs:
        - state_dim, the input dimension of the network (i.e dimension of state)
        - hidden_sizes, an iterable of integers, each of which represents the size
        of a hidden layer in the neural network.
        Returns: Initialized Value network
        '''
        super().__init__()
        if isinstance(activation, str):
            self.activation = activation_with_name(activation)()
        else:
            # Default to tanh.
            self.activation = ACTIVATION()
        self.affine_layers = nn.ModuleList()

        prev = state_dim
        for h in hidden_sizes:
            l = nn.Linear(prev, h)
            if init is not None:
                initialize_weights(l, init)
            self.affine_layers.append(l)
            prev = h

        self.final = nn.Linear(prev, 1)
        if init is not None:
            initialize_weights(self.final, init, scale=1.0)

    def initialize(self, init="orthogonal"):
        for l in self.affine_layers:
            initialize_weights(l, init)
        initialize_weights(self.final, init, scale=1.0)

    def forward(self, x):
        '''
        Performs inference using the value network.
        Inputs:
        - x, the state passed in from the agent
        Returns:
        - The scalar (float) value of that state, as estimated by the net
        '''
        for affine in self.affine_layers:
            x = self.activation(affine(x))
        value = self.final(x)
        return value

    def get_value(self, x):
        return self(x)

    def reset(self):
        return

    # MLP does not maintain history.
    def pause_history(self):
        return

    def continue_history(self):
        return


def pack_history(features, not_dones):
    # Features has dimension (N, state_dim), where N contains a few episodes
    # not_dones splits these episodes (0 in not_dones is end of an episode)
    nnz = ch.nonzero(1.0 - not_dones, as_tuple=False).view(-1).cpu().numpy()
    # nnz has the position where not_dones = 0 (end of episode)
    all_pieces = []
    lengths = []
    start = 0
    for i in nnz:
        end = i + 1
        all_pieces.append(features[start:end, :])
        lengths.append(end - start)
        start = end
    # The last episode is missing, unless the previous episode end at the last element.
    if end != features.size(0):
        all_pieces.append(features[end:, :])
        lengths.append(features.size(0) - end)
    # print(lengths)
    padded = pad_sequence(all_pieces, batch_first=True)
    packed = pack_padded_sequence(padded, lengths, batch_first=True, enforce_sorted=False)
    return packed


def unpack_history(padded_pieces, lengths):
    # padded pieces in shape (batch, time, hidden)
    # lengths in shape (batch,)
    all_pieces = []
    for i, l in enumerate(lengths.cpu().numpy()):
        # For each batch element in padded_pieces, get the first l elements.
        all_pieces.append(padded_pieces[i, 0:l, :])
    # return shape (N, hidden)
    return ch.cat(all_pieces, dim=0)


########################
### POLICY NETWORKS
# Discrete and Continuous Policy Examples
########################
class CtsPolicy(nn.Module):
    '''
    A continuous policy using a fully connected neural network.
    The parameterizing tensor is a mean and standard deviation vector,
    which parameterize a gaussian distribution.
    '''

    def __init__(self, state_dim, action_dim, init="xavier", hidden_sizes=HIDDEN_SIZES,
                 time_in_state=False, share_weights=False, activation=None, use_merged_bias=False):
        super().__init__()
        if isinstance(activation, str):
            self.activation = activation_with_name(activation)()
        else:
            # Default to tanh.
            self.activation = ACTIVATION()
        print('Using activation function', self.activation)
        self.action_dim = action_dim
        self.discrete = False
        self.time_in_state = time_in_state
        self.use_merged_bias = use_merged_bias

        self.affine_layers = nn.ModuleList()
        prev_size = state_dim
        for i in hidden_sizes:
            if use_merged_bias:
                # Use an extra dimension for weight perturbation, simulating bias.
                lin = nn.Linear(prev_size + 1, i, bias=False)
            else:
                lin = nn.Linear(prev_size, i, bias=True)
            initialize_weights(lin, init)
            self.affine_layers.append(lin)
            prev_size = i

        if use_merged_bias:
            self.final_mean = nn.Linear(prev_size + 1, action_dim, bias=False)
        else:
            self.final_mean = nn.Linear(prev_size, action_dim, bias=True)
        initialize_weights(self.final_mean, init, scale=0.01)

        # For the case where we want to share parameters
        # between the policy and value networks
        self.share_weights = share_weights
        if share_weights:
            assert not use_merged_bias
            if time_in_state:
                self.final_value = nn.Linear(prev_size + 1, 1)
            else:
                self.final_value = nn.Linear(prev_size, 1)

            initialize_weights(self.final_value, init, scale=1.0)

        stdev_init = ch.zeros(action_dim)
        self.log_stdev = ch.nn.Parameter(stdev_init)

    def forward(self, x):
        # If the time is in the state, discard it
        if self.time_in_state:
            x = x[:, :-1]
        for affine in self.affine_layers:
            if self.use_merged_bias:
                # Generate an extra "one" for each element, which acts as a bias.
                bias_padding = ch.ones(x.size(0), 1)
                x = ch.cat((x, bias_padding), dim=1)
            else:
                pass
            x = self.activation(affine(x))

        if self.use_merged_bias:
            bias_padding = ch.ones(x.size(0), 1)
            x = ch.cat((x, bias_padding), dim=1)
        means = self.final_mean(x)
        std = ch.exp(self.log_stdev)

        return means, std

    def get_value(self, x):
        assert self.share_weights, "Must be sharing weights to use get_value"

        # If the time is in the state, discard it
        t = None
        if self.time_in_state:
            t = x[..., -1:]
            x = x[..., :-1]

        for affine in self.affine_layers:
            x = self.activation(affine(x))

        if self.time_in_state:
            return self.final_value(ch.cat((x, t), -1))
        else:
            return self.final_value(x)

    def sample(self, p):
        '''
        Given prob dist (mean, var), return: actions sampled from p_i, and their
        probabilities. p is tuple (means, var). means shape
        (batch_size, action_space), var (action_space,), here are batch_size many
        prboability distributions you're sampling from

        Returns tuple (actions, probs):
        - actions: shape (batch_size, action_dim)
        - probs: shape (batch_size, action_dim)
        '''
        means, std = p
        return (means + ch.randn_like(means) * std).detach()

    def get_loglikelihood(self, p, actions):
        try:
            mean, std = p
            nll = 0.5 * ((actions - mean) / std).pow(2).sum(-1) \
                  + 0.5 * np.log(2.0 * np.pi) * actions.shape[-1] \
                  + self.log_stdev.sum(-1)
            return -nll
        except Exception as e:
            raise ValueError("Numerical error")

    def calc_kl(self, p, q):
        '''
        Get the expected KL distance between two sets of gaussians over states -
        gaussians p and q where p and q are each tuples (mean, var)
        - In other words calculates E KL(p||q): E[sum p(x) log(p(x)/q(x))]
        - From https://stats.stackexchange.com/a/60699
        '''
        p_mean, p_std = p
        q_mean, q_std = q
        p_var, q_var = p_std.pow(2), q_std.pow(2)
        assert shape_equal([-1, self.action_dim], p_mean, q_mean)
        assert shape_equal([self.action_dim], p_var, q_var)

        d = q_mean.shape[1]
        diff = q_mean - p_mean

        log_quot_frac = ch.log(q_var).sum() - ch.log(p_var).sum()
        tr = (p_var / q_var).sum()
        quadratic = ((diff / q_var) * diff).sum(dim=1)

        kl_sum = 0.5 * (log_quot_frac - d + tr + quadratic)
        assert kl_sum.shape == (p_mean.shape[0],)
        return kl_sum

    def entropies(self, p):
        '''
        Get entropies over the probability distributions given by p
        p_i = (mean, var), p mean is shape (batch_size, action_space),
        p var is shape (action_space,)
        '''
        _, std = p
        detp = determinant(std)
        d = std.shape[0]
        entropies = ch.log(detp) + .5 * (d * (1. + math.log(2 * math.pi)))
        return entropies

    def reset(self):
        return

    def pause_history(self):
        return

    def continue_history(self):
        return

POLICY_NETS = {
    "CtsPolicy": CtsPolicy,
}

VALUE_NETS = {
    "ValueNet": ValueDenseNet,
}


def partialclass(cls, *args, **kwds):
    class NewCls(cls):
        __init__ = functools.partialmethod(cls.__init__, *args, **kwds)

    return NewCls


ACTIVATIONS = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "leaky": nn.LeakyReLU,
    "leaky0.05": partialclass(nn.LeakyReLU, negative_slope=0.05),
    "leaky0.1": partialclass(nn.LeakyReLU, negative_slope=0.1),
    "hardtanh": nn.Hardtanh,
}


def activation_with_name(name):
    return ACTIVATIONS[name]


def policy_net_with_name(name):
    return POLICY_NETS[name]


def value_net_with_name(name):
    return VALUE_NETS[name]
