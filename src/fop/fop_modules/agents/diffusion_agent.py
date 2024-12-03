import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from .helpers import (cosine_beta_schedule,
                            linear_beta_schedule,
                            vp_beta_schedule,
                            extract,
                            SinusoidalPosEmb, init_weights,
                            Losses)

class Model_RNN(nn.Module):
    def __init__(self, input_shape, args):
        super(Model_RNN, self).__init__()

        
        self.args = args

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(args.time_dim),
            nn.Linear(args.time_dim, args.hidden_size),
            nn.ReLU() ,
            nn.Linear(args.hidden_size, args.time_dim),
        )
        # print(input_shape)
        # print(args.obs_shape)
        # print(args.time_dim)
        # print("---------------------------------------")

        input_dim = input_shape + args.obs_shape + args.time_dim
        # print(input_shape) # 输入 obs+id 34
        # print(args.obs_shape) # 输出 action = obs_shape 30
        # print(args.time_dim) # 32
        # print('******************************')
        self.mish = nn.ReLU() 
        self.fc1 = nn.Linear(input_dim, args.rnn_hidden_dim)
        self.rnn1 = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.obs_shape)
        # self.fc2 = nn.Linear(int(args.rnn_hidden_dim*2), args.rnn_hidden_dim)
        # self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        # self.fc3 = nn.Linear(args.rnn_hidden_dim, args.obs_shape)
        # # self.fc_mu = nn.Linear(args.rnn_hidden_dim, args.obs_shape)
        # # self.fc_std = nn.Linear(args.rnn_hidden_dim, args.obs_shape)
        self.action_bound = args.epsilon_ball

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()
    # x, time, state
    def forward(self, x_n, time, inputs, hidden_state, actions=None):
        
        t = self.time_mlp(time)
        # print(x_n.size())
        # print(t.size())
        # print(inputs.size())
        out = torch.cat([x_n, t, inputs], dim=-1)
        # print(x_n.size()) # 30
        # print(t.size()) # 32
        # print(inputs.size()) # 42
        # print(out.size())
        # print("--------------------------")

        x = self.mish(self.fc1(out))
        # x = self.mish(self.fc2(x))

        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        # print(h_in.size())
        h = self.rnn1(x, h_in)
        actions = self.fc2(h)

        # x = self.fc2(h)
        # mu = self.fc_mu(x)
        # std = F.softplus(self.fc_std(x))
        # dist = Normal(mu, std)
        # normal_sample = dist.rsample()  # rsample()是重参数化采样
        # log_prob = dist.log_prob(normal_sample)
        # actions = torch.tanh(x)
        # actions = F.tanh(x)
        # 计算tanh_normal分布的对数概率密度
        # log_prob = log_prob - torch.log(1 - torch.tanh(actions).pow(2) + 1e-7)
        # actions = actions * self.action_bound
        return {"actions": actions, "hidden_state": h}




# class Model_MLP(nn.Module):
#     # def __init__(self, state_dim, action_dim, hidden_size=256, time_dim=32):
#     def __init__(self, input_shape, args):
#         super(Model_MLP, self).__init__()

#         self.time_mlp = nn.Sequential(
#             SinusoidalPosEmb(args.time_dim),
#             nn.Linear(args.time_dim, args.hidden_size),
#             nn.Mish(),
#             nn.Linear(args.hidden_size, args.time_dim),
#         )

#         input_dim = input_shape + args.action_dim + args.time_dim
#         self.layer = nn.Sequential(nn.Linear(input_dim, args.hidden_size),
#                                        nn.Mish(),
#                                        nn.Linear(args.hidden_size, args.hidden_size),
#                                        nn.Mish(),
#                                        nn.Linear(args.hidden_size, args.hidden_size),
#                                        nn.Mish(),
#                                        nn.Linear(args.hidden_size, args.action_dim))
#         self.apply(init_weights)
        

#     def forward(self, x, time, state):

#         t = self.time_mlp(time)
#         out = torch.cat([x, t, state], dim=-1)
#         out = self.layer(out)
#         log_prob = log_prob(out)

#         return {"actions": out, "log_prob": log_prob}

class Model_MLP(nn.Module):
    def __init__(self, input_shape, args):
        super(Model_MLP, self).__init__()

        
        self.args = args

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(args.time_dim),
            nn.Linear(args.time_dim, args.hidden_size),
            nn.ReLU() ,
            nn.Linear(args.hidden_size, args.time_dim),
        )
        # print(input_shape)
        # print(args.obs_shape)
        # print(args.time_dim)
        # print("---------------------------------------")

        input_dim = input_shape + args.obs_shape + args.time_dim
        # print(input_dim)
        self.layer = nn.Sequential(nn.Linear(input_dim, args.hidden_size),
                                       nn.ReLU(),
                                       nn.Linear(args.hidden_size, args.hidden_size),
                                       nn.ReLU(),
                                       nn.Linear(args.hidden_size, args.hidden_size),
                                       nn.ReLU(),
                                       nn.Linear(args.hidden_size, args.hidden_size),
                                       nn.ReLU())
        self.fc1 = nn.Linear(input_dim, args.rnn_hidden_dim)
        self.fc_mu = nn.Linear(args.hidden_size, args.obs_shape)
        self.fc_std = nn.Linear(args.hidden_size, args.obs_shape)
        
        # self.apply(init_weights)
        self.action_bound = args.epsilon_ball

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()
    # x, time, state
    def forward(self, x_n, time, inputs, hidden_state, actions=None):
  
        t = self.time_mlp(time)
        
        # print(x_n.size())
        # print(t.size())
        # print(inputs.size())

        out = self.layer(torch.cat([x_n, t, inputs], dim=-1))
        # out = out)
        mu = self.fc_mu(out)
        std = F.softplus(self.fc_std(out))
        dist = Normal(mu, std) 
        normal_sample = dist.rsample()  # rsample()是重参数化采样
        log_prob = dist.log_prob(normal_sample)
        actions = torch.tanh(normal_sample)
        
        # actions = F.tanh(x)
        # 计算tanh_normal分布的对数概率密度
        log_prob = log_prob - torch.log(1 - torch.tanh(actions).pow(2) + 1e-7)
        # actions = actions * self.action_bound
        return {"actions": actions, "hidden_state": None, "log_prob": log_prob}
        # actions = self.layer(out)
        # return {"actions": actions, "hidden_state": None}