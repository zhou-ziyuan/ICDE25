# import torch.nn as nn
# import torch.nn.functional as F


# class RNNAgent(nn.Module):
#     def __init__(self, input_shape, args):
#         super(RNNAgent, self).__init__()
#         self.args = args

#         self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
#         self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
#         self.fc2 = nn.Linear(args.rnn_hidden_dim, args.obs_shape)

#     def init_hidden(self):
#         # make hidden states on same device as model
#         return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

#     def forward(self, inputs, hidden_state):
#         x = F.relu(self.fc1(inputs))
#         h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
#         h = self.rnn(x, h_in)
#         q = self.fc2(h)
#         return q, h
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        # self.fc2 = nn.Linear(args.rnn_hidden_dim, args.obs_shape)
        self.fc_mu = nn.Linear(args.rnn_hidden_dim, args.obs_shape)
        self.fc_std = nn.Linear(args.rnn_hidden_dim, args.obs_shape)
        self.action_bound = args.epsilon_ball

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state, actions=None):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        # x = self.fc2(h)
        mu = self.fc_mu(h)
        std = F.softplus(self.fc_std(h))
        dist = Normal(mu, std)
        # print(dist)
        # aaa
        normal_sample = dist.rsample()  # rsample()是重参数化采样
        log_prob = dist.log_prob(normal_sample)
        # print(log_prob)
        # aaa
        actions = torch.tanh(normal_sample)
        
        # actions = F.tanh(x)
        # 计算tanh_normal分布的对数概率密度
        log_prob = log_prob - torch.log(1 - torch.tanh(actions).pow(2) + 1e-7)
        actions = actions * self.action_bound
        return {"actions": actions, "hidden_state": h, "log_prob": log_prob}