import torch
import os
import torch.nn.functional as F
from mappo.network import CtsPolicy as PPOActor
from mappo.network import ValueDenseNet as PPOCritic
from components.episode_buffer import EpisodeBatch
from torch.distributions import Categorical
import numpy as np


class MAPPO:
    def __init__(self, args):
        self.n_actions = args.obs_shape #动作的维度，即输出的扰动
        self.n_agents = args.n_agents #智能体的数目
        self.state_shape = args.obs_shape #状态的维度，即输入的状态
        self.hidden_dim = args.mappo_hidden_dim
        self.gamma = args.mappo_gamma
        self.lamda = args.mappo_lambda
        self.epsilon = args.mappo_epsilon
        self.args = args
        self.epochs = args.mappo_epochs

        self.actor_net = PPOActor(self.state_shape,self.n_actions)
        self.critic_net = PPOCritic(self.state_shape)
        if self.args.use_cuda:
            self.actor_net.cuda()
            self.critic_net.cuda()
            # self.target_critic.cuda()

        # self.parameters = list(self.actor_net.parameters()) + list(self.critic_net.parameters())
        # self.optimizer = torch.optim.RMSprop(self.parameters, lr=args.lr)

        self.actor_optimizer = torch.optim.RMSprop(self.actor_net.parameters(), lr=1e-6)
        self.critic_optimizer = torch.optim.RMSprop(self.critic_net.parameters(), lr=1e-6)


    def _get_critic_input_shape(self):
        input_shape=self.state_shape*self.n_agents
        # input_shape += self.n_actions * self.n_agents * 2  # 54
        return input_shape

    def train(self, batch:EpisodeBatch, t_env:int,episode_num:int ):
        # Get the relevant quantities
        states = batch["obs"][:, :-1]
        next_states = batch["obs"][:, 1:]
        rewards = batch["reward"][:, :-1]
        perturbations = batch["perturbations"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        td_target = torch.unsqueeze(rewards,2) + self.gamma * self.critic_net(next_states) * torch.unsqueeze(mask,2)
        td_delta = td_target - self.critic_net(states)
        advantage = self.compute_advantage(self.gamma, self.lamda, td_delta).transpose(0, 1)
        mu, std = self.actor_net(states)
        action_dists = torch.distributions.Normal(mu.detach(), std.detach())
        # 动作是正态分布
        old_log_probs = action_dists.log_prob(perturbations)

        for _ in range(self.epochs):
            mu, std = self.actor_net(states)
            action_dists = torch.distributions.Normal(mu, std)
            log_probs = action_dists.log_prob(perturbations)
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            critic_loss = torch.mean(
                F.mse_loss(self.critic_net(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_net.parameters(), 10)
            self.actor_optimizer.step()
            self.critic_optimizer.step()


    def compute_advantage(self,gamma, lmbda, td_delta):
        td_delta = td_delta.detach().numpy()
        advantage_list = []
        advantage = np.zeros((td_delta.shape[0],td_delta.shape[2],td_delta.shape[3]), dtype=np.float32)
        for delta in np.flip(td_delta,1).swapaxes(0,1): # delta是所有batch的第一个时间步的delta
            advantage = gamma * lmbda * advantage + delta
            advantage_list.append(advantage)
        advantage_list.reverse()
        return torch.tensor(advantage_list, dtype=torch.float)

    def save_models(self, path):
        torch.save(self.actor_net.state_dict(), "{}/actor_net.pth".format(path))
        torch.save(self.critic_net.state_dict(), "{}/critic_net.pth".format(path))
        torch.save(self.actor_optimizer.state_dict(), "{}/actor_optimizer.th".format(path))
        torch.save(self.critic_optimizer.state_dict(), "{}/critic_optimizer.th".format(path))

    def load_models(self, path):
        self.actor_net.load_state_dict(torch.load("{}/actor_net.pth".format(path), map_location=lambda storage, loc: storage))
        self.critic_net.load_state_dict(torch.load("{}/critic_net.pth".format(path), map_location=lambda storage, loc: storage))
        self.actor_optimizer.load_state_dict(torch.load("{}/actor_optimizer.th".format(path), map_location=lambda storage, loc: storage))
        self.critic_optimizer.load_state_dict(torch.load("{}/critic_optimizer.th".format(path), map_location=lambda storage, loc: storage))