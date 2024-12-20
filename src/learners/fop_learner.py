import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.fop import FOPMixer
import torch.nn.functional as F
import torch.nn as nn

import torch as th
from torch.optim import RMSprop
import numpy as np
from torch.distributions import Categorical
from modules.critics.fop import FOPCritic
from utils.rl_utils import build_td_lambda_targets

class FOP_Learner:
    def __init__(self, mac, vic_mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.vic_mac = vic_mac
        self.logger = logger
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.last_target_update_episode = 0
        self.critic_training_steps = 0

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.critic1 = FOPCritic(scheme, args)
        self.critic2 = FOPCritic(scheme, args)

        self.mixer1 = FOPMixer(args)
        self.mixer2 = FOPMixer(args)
        
        self.target_mixer1 = copy.deepcopy(self.mixer1)
        self.target_mixer2 = copy.deepcopy(self.mixer2)
        
        self.target_critic1 = copy.deepcopy(self.critic1)
        self.target_critic2 = copy.deepcopy(self.critic2)
        
        self.agent_params = list(mac.parameters())
        self.critic_params1 = list(self.critic1.parameters()) + list(self.mixer1.parameters())
        self.critic_params2 = list(self.critic2.parameters()) + list(self.mixer2.parameters())
 
        self.p_optimiser = RMSprop(params=self.agent_params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.c_optimiser1 = RMSprop(params=self.critic_params1, lr=args.c_lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.c_optimiser2 = RMSprop(params=self.critic_params2, lr=args.c_lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # 使用alpha的log值,可以使训练结果比较稳定
        self.log_alpha = th.tensor(np.log(args.kappa_max), dtype=th.float)
        self.log_alpha.requires_grad = True  # 可以对alpha求梯度
        self.log_alpha_optimizer = th.optim.Adam([self.log_alpha],
                                                    lr=1e-2)
        
        self.target_entropy = -self.n_actions*self.n_agents

    def train_actor(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        bs = batch.batch_size
        max_t = batch.max_seq_length
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        mask = mask.repeat(1, 1, self.n_agents).view(-1)
        avail_actions = batch["avail_actions"]

        mac = self.mac
        # alpha = max(0.05, 0.5 - t_env / 200000) # linear decay
        alpha = self.log_alpha.exp()

        mac_out = []
        # mac.init_hidden(batch.batch_size)
        self.actor_hidden_state = mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            # agent_outs = mac.forward(batch, t=t)
            agent_outs, hidden_state_= mac.forward(batch, t=t, hidden_states = self.actor_hidden_state)
            self.actor_hidden_state = hidden_state_
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Mask out unavailable actions, renormalise (as in action selection)
        mac_out[avail_actions == 0] = 1e-10
        mac_out = mac_out/mac_out.sum(dim=-1, keepdim=True)
        mac_out[avail_actions == 0] = 1e-10

        pi = mac_out[:,:-1].clone()
        pi = pi.reshape(-1, self.n_actions)
        log_pi = th.log(pi)
        # print(log_pi)

        inputs = self.critic1._build_inputs(batch, bs, max_t)
        q_vals1 = self.critic1.forward(inputs)
        q_vals2 = self.critic2.forward(inputs)
        q_vals = th.min(q_vals1, q_vals2)

        pi = mac_out[:,:-1].reshape(-1, self.n_actions)
        entropies = - (pi * log_pi).sum(dim=-1)

        # policy target for discrete actions (from Soft Actor-Critic for Discrete Action Settings)
        pol_target = (pi * (alpha * log_pi - q_vals[:,:-1].reshape(-1, self.n_actions))).sum(dim=-1)

        policy_loss = (pol_target * mask).sum() / mask.sum()

        # Optimise
        self.p_optimiser.zero_grad()
        policy_loss.backward()
        agent_grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
        self.p_optimiser.step()

        alpha_loss = th.mean(
            (entropies - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("adv_policy_loss", policy_loss.item(), t_env)
            self.logger.log_stat("adv_agent_grad_norm", agent_grad_norm, t_env)
            self.logger.log_stat("pi_max", (pi.max(dim=1)[0] * mask).sum().item() / mask.sum().item(), t_env)
            self.logger.log_stat("alpha", alpha, t_env)
            self.logger.log_stat("ent", entropies.mean().item(), t_env)

    def train_actor_loss(self, batch: EpisodeBatch, vic_batch: EpisodeBatch, t_env: int, episode_num: int):
        bs = batch.batch_size
        max_t = batch.max_seq_length
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        mask = mask.repeat(1, 1, self.n_agents).view(-1)
        avail_actions = batch["avail_actions"]

        mac = self.mac
        vic_mac = self.vic_mac
        # alpha = max(0.05, 0.5 - t_env / 200000) # linear decay
        alpha = max(self.args.kappa_min, self.args.kappa_max - t_env / 200000) # linear decay

        mac_out = []
        margins = []
        # mac.init_hidden(batch.batch_size)
        self.actor_hidden_state = mac.init_hidden(batch.batch_size)
        self.ori_hidden_state = vic_mac.init_hidden(vic_batch.batch_size)
        # for t in range(batch.max_seq_length):
        #     # agent_outs = mac.forward(batch, t=t)
        #     agent_outs, hidden_state_= mac.forward(batch, t=t, hidden_states = self.actor_hidden_state)
        #     # clean_agent_out = mac_agent.forward()
        #     mac_out.append(agent_outs)
        #     self.actor_hidden_state = hidden_state_
        # mac_out = th.stack(mac_out, dim=1)  # Concat over time
        criterion = nn.CrossEntropyLoss()
        for t in range(batch.max_seq_length):
            agent_outs, hidden_state_ = self.mac.forward(batch, t=t, hidden_states=self.actor_hidden_state)
            ori_agent_outs, ori_hidden_state_ = self.vic_mac.forward(vic_batch, t=t, hidden_states=self.ori_hidden_state)
            # -------
            # if self.args.attack_method == 'adv_reg':
            # ori_max_action_outs = th.argmax(agent_outs, dim=2).clone().detach()
            ori_max_action_outs = th.argmax(ori_agent_outs, dim=2).clone().detach()
            # max_action_outs = th.argmax(agent_outs, dim=2).clone().detach()
            # adv_margin = logits_margin(agent_outs, ori_max_action_outs)
            # print(adv_tar_logits, adv_actions[0])
            # print(ori_agent_outs.size(),ori_agent_outs)
            # print(ori_max_action_outs.size(),ori_max_action_outs)
            adv_margin = criterion(agent_outs.view(-1, agent_outs.size()[2]), ori_max_action_outs.view(-1))
            # adv_margin = criterion(ori_agent_outs.view(-1, ori_agent_outs.size()[2]), ori_max_action_outs.view(-1))
            # print(loss_action)
            margins.append(adv_margin)
            # -------
            self.actor_hidden_state = hidden_state_
            self.ori_hidden_state = ori_hidden_state_
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        margins = th.stack(margins, dim=0)
        margins = margins.mean()
        reg_loss = th.clamp(margins, min=-self.args.hinge_c)


        # Mask out unavailable actions, renormalise (as in action selection)
        mac_out[avail_actions == 0] = 1e-10
        mac_out = mac_out/mac_out.sum(dim=-1, keepdim=True)
        mac_out[avail_actions == 0] = 1e-10

        pi = mac_out[:,:-1].clone()
        pi = pi.reshape(-1, self.n_actions)
        log_pi = th.log(pi)

        inputs = self.critic1._build_inputs(batch, bs, max_t)
        q_vals1 = self.critic1.forward(inputs)
        q_vals2 = self.critic2.forward(inputs)
        q_vals = th.min(q_vals1, q_vals2)

        pi = mac_out[:,:-1].reshape(-1, self.n_actions)
        entropies = - (pi * log_pi).sum(dim=-1)

        # policy target for discrete actions (from Soft Actor-Critic for Discrete Action Settings)
        pol_target = (pi * (alpha * log_pi - q_vals[:,:-1].reshape(-1, self.n_actions))).sum(dim=-1)

        policy_loss = (pol_target * mask + reg_loss * self.args.kappa).sum() / mask.sum()

        # Optimise
        self.p_optimiser.zero_grad()
        policy_loss.backward()
        agent_grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
        self.p_optimiser.step()

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("adv_policy_loss", policy_loss.item(), t_env)
            self.logger.log_stat("adv_agent_grad_norm", agent_grad_norm, t_env)
            self.logger.log_stat("pi_max", (pi.max(dim=1)[0] * mask).sum().item() / mask.sum().item(), t_env)
            self.logger.log_stat("alpha", alpha, t_env)
            self.logger.log_stat("ent", entropies.mean().item(), t_env)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, show_demo=False, save_data=None):
        self.train_actor(batch, t_env, episode_num)
        self.train_critic(batch, t_env)
        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

    def train_action_loss(self, batch: EpisodeBatch, vic_batch:EpisodeBatch ,t_env: int, episode_num: int, show_demo=False, save_data=None):
        self.train_actor_loss(batch, vic_batch, t_env, episode_num)
        self.train_critic(batch, t_env)
        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

    def train_critic(self, batch, t_env):
        bs = batch.batch_size
        max_t = batch.max_seq_length
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        actions_onehot = batch["actions_onehot"][:, :-1]
        states = batch["state"]

        mac = self.mac
        mixer1 = self.mixer1
        mixer2 = self.mixer2
        # alpha = max(0.05, 0.5 - t_env / 200000) # linear decay
        # alpha = self.log_alpha.exp()
        alpha = max(self.args.kappa_min, self.args.kappa_max - t_env / 200000) # linear decay

        mac_out = []
        # mac.init_hidden(batch.batch_size)
        self.critic_hidden_state = mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            # agent_outs = mac.forward(batch, t=t)
            agent_outs, hidden_state_= mac.forward(batch, t=t, hidden_states = self.critic_hidden_state)
            # print(agent_outs)
            self.critic_hidden_state = hidden_state_
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Mask out unavailable actions
        mac_out[avail_actions == 0] = 0.0
        mac_out = mac_out/mac_out.sum(dim=-1, keepdim=True)
        mac_out[avail_actions == 0] = 1e-10

        t_mac_out = mac_out.clone().detach() 
        pi = t_mac_out

        # sample actions for next timesteps
        # print(pi)
        next_actions = Categorical(pi).sample().long().unsqueeze(3)
        next_actions_onehot = th.zeros(next_actions.squeeze(3).shape + (self.n_actions,))
        if self.args.use_cuda:
            next_actions_onehot = next_actions_onehot.cuda()
        next_actions_onehot = next_actions_onehot.scatter_(3, next_actions, 1)

        pi_taken = th.gather(pi, dim=3, index=next_actions).squeeze(3)[:,1:]
        pi_taken[mask.expand_as(pi_taken) == 0] = 1.0
        log_pi_taken = th.log(pi_taken)

        target_inputs = self.target_critic1._build_inputs(batch, bs, max_t)
        target_q_vals1 = self.target_critic1.forward(target_inputs).detach()
        target_q_vals2 = self.target_critic2.forward(target_inputs).detach()

        # directly caculate the values by definition
        next_vs1 = th.logsumexp(target_q_vals1 / alpha, dim=-1) * alpha
        next_vs2 = th.logsumexp(target_q_vals2 / alpha, dim=-1) * alpha
        
        next_chosen_qvals1 = th.gather(target_q_vals1, dim=3, index=next_actions).squeeze(3)
        next_chosen_qvals2 = th.gather(target_q_vals2, dim=3, index=next_actions).squeeze(3)

        target_qvals1 = self.target_mixer1(next_chosen_qvals1, states, actions=next_actions_onehot, vs=next_vs1)
        target_qvals2 = self.target_mixer2(next_chosen_qvals2, states, actions=next_actions_onehot, vs=next_vs2)

        target_qvals = th.min(target_qvals1, target_qvals2)

        # Calculate td-lambda targets
        target_v = build_td_lambda_targets(rewards, terminated, mask, target_qvals, self.n_agents, self.args.gamma, self.args.td_lambda)
        targets = target_v - alpha * log_pi_taken.mean(dim=-1, keepdim=True)

        inputs = self.critic1._build_inputs(batch, bs, max_t)
        q_vals1 = self.critic1.forward(inputs)
        q_vals2 = self.critic2.forward(inputs)

        # directly caculate the values by definition
        vs1 = th.logsumexp(q_vals1 / alpha, dim=-1) * alpha
        vs2 = th.logsumexp(q_vals2 / alpha, dim=-1) * alpha

        q_taken1 = th.gather(q_vals1[:,:-1], dim=3, index=actions).squeeze(3)
        q_taken2 = th.gather(q_vals2[:,:-1], dim=3, index=actions).squeeze(3)

        q_taken1 = mixer1(q_taken1, states[:, :-1], actions=actions_onehot, vs=vs1[:, :-1])
        q_taken2 = mixer2(q_taken2, states[:, :-1], actions=actions_onehot, vs=vs2[:, :-1])

        td_error1 = q_taken1 - targets.detach()
        td_error2 = q_taken2 - targets.detach()

        mask = mask.expand_as(td_error1)

        # 0-out the targets that came from padded data
        masked_td_error1 = td_error1 * mask
        loss1 = (masked_td_error1 ** 2).sum() / mask.sum() 
        masked_td_error2 = td_error2 * mask
        loss2 = (masked_td_error2 ** 2).sum() / mask.sum() 
        
        # Optimise
        self.c_optimiser1.zero_grad()
        loss1.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params1, self.args.grad_norm_clip)
        self.c_optimiser1.step()
        
        self.c_optimiser2.zero_grad()
        loss2.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params2, self.args.grad_norm_clip)
        self.c_optimiser2.step()
 

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("adv_critic_loss", loss1.item(), t_env)
            self.logger.log_stat("adv_critic_grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("adv_critic_td_error_abs", (masked_td_error1.abs().sum().item() / mask_elems), t_env)
            self.logger.log_stat("adv_critic_q_taken_mean",
                                 (q_taken1 * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("adv_critic_target_mean", (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
                                 t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        self.target_mixer1.load_state_dict(self.mixer1.state_dict())
        self.target_mixer2.load_state_dict(self.mixer2.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.critic1.cuda()
        self.mixer1.cuda()
        self.target_critic1.cuda()
        self.target_mixer1.cuda()
        self.critic2.cuda()
        self.mixer2.cuda()
        self.target_critic2.cuda()
        self.target_mixer2.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.critic1.state_dict(), "{}/critic1.th".format(path))
        th.save(self.mixer1.state_dict(), "{}/mixer1.th".format(path))
        th.save(self.critic2.state_dict(), "{}/critic2.th".format(path))
        th.save(self.mixer2.state_dict(), "{}/mixer2.th".format(path))
        th.save(self.p_optimiser.state_dict(), "{}/agent_opt.th".format(path))
        th.save(self.c_optimiser1.state_dict(), "{}/critic_opt1.th".format(path))
        th.save(self.c_optimiser2.state_dict(), "{}/critic_opt2.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.critic1.load_state_dict(th.load("{}/critic1.th".format(path), map_location=lambda storage, loc: storage))
        self.critic2.load_state_dict(th.load("{}/critic2.th".format(path), map_location=lambda storage, loc: storage))
        # Not quite right but I don't want to save target networks
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.mixer1.load_state_dict(th.load("{}/mixer1.th".format(path), map_location=lambda storage, loc: storage))
        self.mixer2.load_state_dict(th.load("{}/mixer2.th".format(path), map_location=lambda storage, loc: storage))

        self.p_optimiser.load_state_dict(th.load("{}/agent_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.c_optimiser1.load_state_dict(th.load("{}/critic_opt1.th".format(path), map_location=lambda storage, loc: storage))
        self.c_optimiser2.load_state_dict(th.load("{}/critic_opt2.th".format(path), map_location=lambda storage, loc: storage))
        
    def build_inputs(self, batch, bs, max_t, actions_onehot):
        inputs = []
        inputs.append(batch["obs"][:])
        actions = actions_onehot[:].reshape(bs, max_t, self.n_agents, -1)
        inputs.append(actions)
        inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1))
        inputs = th.cat([x.reshape(bs, max_t, self.n_agents, -1) for x in inputs], dim=-1)
        return inputs

# import copy
# from components.episode_buffer import EpisodeBatch
# from modules.mixers.fop import FOPMixer
# import torch.nn.functional as F
# import torch as th
# from torch.optim import RMSprop
# import numpy as np
# from torch.distributions import Categorical
# import torch.nn as nn

# from modules.critics.fop import FOPCritic
# from utils.rl_utils import build_td_lambda_targets

# class FOP_Learner:
#     def __init__(self, mac, vic_mac, scheme, logger, args):
#         self.args = args
#         self.mac = mac
#         self.vic_mac = vic_mac
#         self.logger = logger
#         self.n_agents = args.n_agents
#         self.n_actions = args.n_actions
#         self.last_target_update_episode = 0
#         self.critic_training_steps = 0

#         self.log_stats_t = -self.args.learner_log_interval - 1

#         self.critic1 = FOPCritic(scheme, args)
#         self.critic2 = FOPCritic(scheme, args)

#         self.mixer1 = FOPMixer(args)
#         self.mixer2 = FOPMixer(args)
        
#         self.target_mixer1 = copy.deepcopy(self.mixer1)
#         self.target_mixer2 = copy.deepcopy(self.mixer2)
        
#         self.target_critic1 = copy.deepcopy(self.critic1)
#         self.target_critic2 = copy.deepcopy(self.critic2)
        
#         self.agent_params = list(mac.parameters())
#         self.critic_params1 = list(self.critic1.parameters()) + list(self.mixer1.parameters())
#         self.critic_params2 = list(self.critic2.parameters()) + list(self.mixer2.parameters())
 
#         self.p_optimiser = RMSprop(params=self.agent_params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
#         self.c_optimiser1 = RMSprop(params=self.critic_params1, lr=args.c_lr, alpha=args.optim_alpha, eps=args.optim_eps)
#         self.c_optimiser2 = RMSprop(params=self.critic_params2, lr=args.c_lr, alpha=args.optim_alpha, eps=args.optim_eps)

#     def train_actor(self, batch: EpisodeBatch, t_env: int, episode_num: int):
#         bs = batch.batch_size
#         max_t = batch.max_seq_length
#         terminated = batch["terminated"][:, :-1].float()
#         mask = batch["filled"][:, :-1].float()
#         mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
#         mask = mask.repeat(1, 1, self.n_agents).view(-1)
#         avail_actions = batch["avail_actions"]

#         mac = self.mac
#         alpha = max(0.05, 0.5 - t_env / 200000) # linear decay

#         mac_out = []
#         # mac.init_hidden(batch.batch_size)
#         self.actor_hidden_state = mac.init_hidden(batch.batch_size)
#         for t in range(batch.max_seq_length):
#             # agent_outs = mac.forward(batch, t=t)
#             agent_outs, hidden_state_= mac.forward(batch, t=t, hidden_states = self.actor_hidden_state)
#             # clean_agent_out = mac_agent.forward()
#             mac_out.append(agent_outs)
#             self.actor_hidden_state = hidden_state_
#         mac_out = th.stack(mac_out, dim=1)  # Concat over time

#         # Mask out unavailable actions, renormalise (as in action selection)
#         mac_out[avail_actions == 0] = 1e-10
#         mac_out = mac_out/mac_out.sum(dim=-1, keepdim=True)
#         mac_out[avail_actions == 0] = 1e-10

#         pi = mac_out[:,:-1].clone()
#         pi = pi.reshape(-1, self.n_actions)
#         log_pi = th.log(pi)

#         inputs = self.critic1._build_inputs(batch, bs, max_t)
#         q_vals1 = self.critic1.forward(inputs)
#         q_vals2 = self.critic2.forward(inputs)
#         q_vals = th.min(q_vals1, q_vals2)

#         pi = mac_out[:,:-1].reshape(-1, self.n_actions)
#         entropies = - (pi * log_pi).sum(dim=-1)

#         # policy target for discrete actions (from Soft Actor-Critic for Discrete Action Settings)
#         pol_target = (pi * (alpha * log_pi - q_vals[:,:-1].reshape(-1, self.n_actions))).sum(dim=-1)

#         policy_loss = (pol_target * mask).sum() / mask.sum()

#         # Optimise
#         self.p_optimiser.zero_grad()
#         policy_loss.backward()
#         agent_grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
#         self.p_optimiser.step()

#         if t_env - self.log_stats_t >= self.args.learner_log_interval:
#             self.logger.log_stat("adv_policy_loss", policy_loss.item(), t_env)
#             self.logger.log_stat("adv_agent_grad_norm", agent_grad_norm, t_env)
#             self.logger.log_stat("pi_max", (pi.max(dim=1)[0] * mask).sum().item() / mask.sum().item(), t_env)
#             self.logger.log_stat("alpha", alpha, t_env)
#             self.logger.log_stat("ent", entropies.mean().item(), t_env)

#     def train_actor_loss(self, batch: EpisodeBatch, vic_batch: EpisodeBatch, t_env: int, episode_num: int):
#         bs = batch.batch_size
#         max_t = batch.max_seq_length
#         terminated = batch["terminated"][:, :-1].float()
#         mask = batch["filled"][:, :-1].float()
#         mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
#         mask = mask.repeat(1, 1, self.n_agents).view(-1)
#         avail_actions = batch["avail_actions"]

#         mac = self.mac
#         vic_mac = self.vic_mac
#         # alpha = max(0.05, 0.5 - t_env / 200000) # linear decay
#         alpha = max(self.args.kappa_min, self.args.kappa_max - t_env / 200000) # linear decay

#         mac_out = []
#         margins = []
#         # mac.init_hidden(batch.batch_size)
#         self.actor_hidden_state = mac.init_hidden(batch.batch_size)
#         self.ori_hidden_state = vic_mac.init_hidden(vic_batch.batch_size)
#         # for t in range(batch.max_seq_length):
#         #     # agent_outs = mac.forward(batch, t=t)
#         #     agent_outs, hidden_state_= mac.forward(batch, t=t, hidden_states = self.actor_hidden_state)
#         #     # clean_agent_out = mac_agent.forward()
#         #     mac_out.append(agent_outs)
#         #     self.actor_hidden_state = hidden_state_
#         # mac_out = th.stack(mac_out, dim=1)  # Concat over time
#         criterion = nn.CrossEntropyLoss()
#         for t in range(batch.max_seq_length):
#             agent_outs, hidden_state_ = self.mac.forward(batch, t=t, hidden_states=self.actor_hidden_state)
#             ori_agent_outs, ori_hidden_state_ = self.vic_mac.forward(vic_batch, t=t, hidden_states=self.ori_hidden_state)
#             # -------
#             # if self.args.attack_method == 'adv_reg':
#             ori_max_action_outs = th.argmax(agent_outs, dim=2).clone().detach()
#             # max_action_outs = th.argmax(agent_outs, dim=2).clone().detach()
#             # adv_margin = logits_margin(agent_outs, ori_max_action_outs)
#             # print(adv_tar_logits, adv_actions[0])
#             # print(ori_agent_outs.size(),ori_agent_outs)
#             # print(ori_max_action_outs.size(),ori_max_action_outs)
            
#             adv_margin = criterion(ori_agent_outs.view(-1, ori_agent_outs.size()[2]), ori_max_action_outs.view(-1))
#             # print(loss_action)
#             margins.append(adv_margin)
#             # -------
#             self.actor_hidden_state = hidden_state_
#             self.ori_hidden_state = ori_hidden_state_
#             mac_out.append(agent_outs)
#         mac_out = th.stack(mac_out, dim=1)  # Concat over time
#         margins = th.stack(margins, dim=0)
#         margins = margins.mean()
#         reg_loss = th.clamp(margins, min=-self.args.hinge_c)


#         # Mask out unavailable actions, renormalise (as in action selection)
#         mac_out[avail_actions == 0] = 1e-10
#         mac_out = mac_out/mac_out.sum(dim=-1, keepdim=True)
#         mac_out[avail_actions == 0] = 1e-10

#         pi = mac_out[:,:-1].clone()
#         pi = pi.reshape(-1, self.n_actions)
#         log_pi = th.log(pi)

#         inputs = self.critic1._build_inputs(batch, bs, max_t)
#         q_vals1 = self.critic1.forward(inputs)
#         q_vals2 = self.critic2.forward(inputs)
#         q_vals = th.min(q_vals1, q_vals2)

#         pi = mac_out[:,:-1].reshape(-1, self.n_actions)
#         entropies = - (pi * log_pi).sum(dim=-1)

#         # policy target for discrete actions (from Soft Actor-Critic for Discrete Action Settings)
#         pol_target = (pi * (alpha * log_pi - q_vals[:,:-1].reshape(-1, self.n_actions))).sum(dim=-1)

#         policy_loss = (pol_target * mask + reg_loss * self.args.kappa).sum() / mask.sum()

#         # Optimise
#         self.p_optimiser.zero_grad()
#         policy_loss.backward()
#         agent_grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
#         self.p_optimiser.step()

#         if t_env - self.log_stats_t >= self.args.learner_log_interval:
#             self.logger.log_stat("adv_policy_loss", policy_loss.item(), t_env)
#             self.logger.log_stat("adv_agent_grad_norm", agent_grad_norm, t_env)
#             self.logger.log_stat("pi_max", (pi.max(dim=1)[0] * mask).sum().item() / mask.sum().item(), t_env)
#             self.logger.log_stat("alpha", alpha, t_env)
#             self.logger.log_stat("ent", entropies.mean().item(), t_env)

#     def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, show_demo=False, save_data=None):
#         self.train_actor(batch, t_env, episode_num)
#         self.train_critic(batch, t_env)
#         if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
#             self._update_targets()
#             self.last_target_update_episode = episode_num

#     def train_action_loss(self, batch: EpisodeBatch, vic_batch:EpisodeBatch ,t_env: int, episode_num: int, show_demo=False, save_data=None):
#         self.train_actor_loss(batch, vic_batch, t_env, episode_num)
#         self.train_critic(batch, t_env)
#         if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
#             self._update_targets()
#             self.last_target_update_episode = episode_num

#     def train_critic(self, batch, t_env):
#         bs = batch.batch_size
#         max_t = batch.max_seq_length
#         rewards = batch["reward"][:, :-1]
#         actions = batch["actions"][:, :-1]
#         terminated = batch["terminated"][:, :-1].float()
#         mask = batch["filled"][:, :-1].float()
#         mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
#         avail_actions = batch["avail_actions"]
#         actions_onehot = batch["actions_onehot"][:, :-1]
#         states = batch["state"]

#         mac = self.mac
#         mixer1 = self.mixer1
#         mixer2 = self.mixer2
#         # alpha = max(0.05, 0.5 - t_env / 200000) # linear decay
#         alpha = max(self.args.kappa_min, self.args.kappa_max - t_env / 200000)

#         mac_out = []
#         # mac.init_hidden(batch.batch_size)
#         self.critic_hidden_state = mac.init_hidden(batch.batch_size)
#         for t in range(batch.max_seq_length):
#             # agent_outs = mac.forward(batch, t=t)
#             agent_outs, hidden_state_= mac.forward(batch, t=t, hidden_states = self.critic_hidden_state)
#             # print(agent_outs)
#             self.critic_hidden_state = hidden_state_
#             mac_out.append(agent_outs)
#         # for t in range(batch.max_seq_length):
#         #     agent_outs = mac.forward(batch, t=t)
#         #     mac_out.append(agent_outs)
#         mac_out = th.stack(mac_out, dim=1)  # Concat over time

#         # Mask out unavailable actions
#         mac_out[avail_actions == 0] = 0.0
#         mac_out = mac_out/mac_out.sum(dim=-1, keepdim=True)
#         mac_out[avail_actions == 0] = 1e-10

#         t_mac_out = mac_out.clone().detach() 
#         pi = t_mac_out

#         # sample actions for next timesteps
#         next_actions = Categorical(pi).sample().long().unsqueeze(3)
#         next_actions_onehot = th.zeros(next_actions.squeeze(3).shape + (self.n_actions,))
#         if self.args.use_cuda:
#             next_actions_onehot = next_actions_onehot.cuda()
#         next_actions_onehot = next_actions_onehot.scatter_(3, next_actions, 1)

#         pi_taken = th.gather(pi, dim=3, index=next_actions).squeeze(3)[:,1:]
#         pi_taken[mask.expand_as(pi_taken) == 0] = 1.0
#         log_pi_taken = th.log(pi_taken)

#         target_inputs = self.target_critic1._build_inputs(batch, bs, max_t)
#         target_q_vals1 = self.target_critic1.forward(target_inputs).detach()
#         target_q_vals2 = self.target_critic2.forward(target_inputs).detach()

#         # directly caculate the values by definition
#         next_vs1 = th.logsumexp(target_q_vals1 / alpha, dim=-1) * alpha
#         next_vs2 = th.logsumexp(target_q_vals2 / alpha, dim=-1) * alpha
        
#         next_chosen_qvals1 = th.gather(target_q_vals1, dim=3, index=next_actions).squeeze(3)
#         next_chosen_qvals2 = th.gather(target_q_vals2, dim=3, index=next_actions).squeeze(3)

#         target_qvals1, _ = self.target_mixer1(next_chosen_qvals1, states, actions=next_actions_onehot, vs=next_vs1)
#         target_qvals2, _ = self.target_mixer2(next_chosen_qvals2, states, actions=next_actions_onehot, vs=next_vs2)

#         target_qvals = th.min(target_qvals1, target_qvals2)

#         # Calculate td-lambda targets
#         target_v = build_td_lambda_targets(rewards, terminated, mask, target_qvals, self.n_agents, self.args.gamma, self.args.td_lambda)
#         #target_v = rewards + self.args.gamma * (1 - terminated) * target_qvals
#         targets = target_v - alpha * log_pi_taken.mean(dim=-1, keepdim=True)

#         inputs = self.critic1._build_inputs(batch, bs, max_t)
#         q_vals1 = self.critic1.forward(inputs)
#         q_vals2 = self.critic2.forward(inputs)

#         # directly caculate the values by definition
#         vs1 = th.logsumexp(q_vals1 / alpha, dim=-1) * alpha
#         vs2 = th.logsumexp(q_vals2 / alpha, dim=-1) * alpha

#         q_taken1 = th.gather(q_vals1[:,:-1], dim=3, index=actions).squeeze(3)
#         q_taken2 = th.gather(q_vals2[:,:-1], dim=3, index=actions).squeeze(3)

#         q_taken1, q_attend_regs1 = mixer1(q_taken1, states[:, :-1], actions=actions_onehot, vs=vs1[:, :-1])
#         q_taken2, q_attend_regs2 = mixer2(q_taken2, states[:, :-1], actions=actions_onehot, vs=vs2[:, :-1])

#         td_error1 = q_taken1 - targets.detach()
#         td_error2 = q_taken2 - targets.detach()

#         mask = mask.expand_as(td_error1)

#         # 0-out the targets that came from padded data
#         masked_td_error1 = td_error1 * mask
#         loss1 = (masked_td_error1 ** 2).sum() / mask.sum() + q_attend_regs1
#         masked_td_error2 = td_error2 * mask
#         loss2 = (masked_td_error2 ** 2).sum() / mask.sum() + q_attend_regs2
        
#         # Optimise
#         self.c_optimiser1.zero_grad()
#         loss1.backward()
#         grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params1, self.args.grad_norm_clip)
#         self.c_optimiser1.step()
        
#         self.c_optimiser2.zero_grad()
#         loss2.backward()
#         grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params2, self.args.grad_norm_clip)
#         self.c_optimiser2.step()
 

#         if t_env - self.log_stats_t >= self.args.learner_log_interval:
#             # self.logger.log_stat("loss", loss1.item(), t_env)
#             # try:
#             # 	self.logger.log_stat("grad_norm", grad_norm.item(), t_env)
#             # except:
#             #     self.logger.log_stat("grad_norm", grad_norm, t_env)
#             # mask_elems = mask.sum().item()
#             # self.logger.log_stat("td_error_abs", (masked_td_error1.abs().sum().item() / mask_elems), t_env)
#             # self.logger.log_stat("q_taken_mean",
#             #                      (q_taken1 * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
#             # self.logger.log_stat("target_mean", (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
#             #                      t_env)
#             # self.log_stats_t = t_env
#             self.logger.log_stat("adv_critic_loss", loss1.item(), t_env)
#             self.logger.log_stat("adv_critic_grad_norm", grad_norm, t_env)
#             mask_elems = mask.sum().item()
#             self.logger.log_stat("adv_critic_td_error_abs", (masked_td_error1.abs().sum().item() / mask_elems), t_env)
#             self.logger.log_stat("adv_critic_q_taken_mean",
#                                  (q_taken1 * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
#             self.logger.log_stat("adv_critic_target_mean", (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
#                                  t_env)
#             self.log_stats_t = t_env

#     def _update_targets(self):
#         self.target_critic1.load_state_dict(self.critic1.state_dict())
#         self.target_critic2.load_state_dict(self.critic2.state_dict())
#         self.target_mixer1.load_state_dict(self.mixer1.state_dict())
#         self.target_mixer2.load_state_dict(self.mixer2.state_dict())
#         self.logger.console_logger.info("Updated target network")

#     def cuda(self):
#         self.mac.cuda()
#         self.critic1.cuda()
#         self.mixer1.cuda()
#         self.target_critic1.cuda()
#         self.target_mixer1.cuda()
#         self.critic2.cuda()
#         self.mixer2.cuda()
#         self.target_critic2.cuda()
#         self.target_mixer2.cuda()

#     def save_models(self, path):
#         self.mac.save_models(path)
#         th.save(self.critic1.state_dict(), "{}/critic1.th".format(path))
#         th.save(self.mixer1.state_dict(), "{}/mixer1.th".format(path))
#         th.save(self.critic2.state_dict(), "{}/critic2.th".format(path))
#         th.save(self.mixer2.state_dict(), "{}/mixer2.th".format(path))
#         th.save(self.p_optimiser.state_dict(), "{}/agent_opt.th".format(path))
#         th.save(self.c_optimiser1.state_dict(), "{}/critic_opt1.th".format(path))
#         th.save(self.c_optimiser2.state_dict(), "{}/critic_opt2.th".format(path))

#     def load_models(self, path):
#         self.mac.load_models(path)
#         self.critic1.load_state_dict(th.load("{}/critic1.th".format(path), map_location=lambda storage, loc: storage))
#         self.critic2.load_state_dict(th.load("{}/critic2.th".format(path), map_location=lambda storage, loc: storage))
#         # Not quite right but I don't want to save target networks
#         self.target_critic1.load_state_dict(self.critic1.state_dict())
#         self.target_critic2.load_state_dict(self.critic2.state_dict())

#         self.mixer1.load_state_dict(th.load("{}/mixer1.th".format(path), map_location=lambda storage, loc: storage))
#         self.mixer2.load_state_dict(th.load("{}/mixer2.th".format(path), map_location=lambda storage, loc: storage))

#         self.p_optimiser.load_state_dict(th.load("{}/agent_opt.th".format(path), map_location=lambda storage, loc: storage))
#         self.c_optimiser1.load_state_dict(th.load("{}/critic_opt1.th".format(path), map_location=lambda storage, loc: storage))
#         self.c_optimiser2.load_state_dict(th.load("{}/critic_opt2.th".format(path), map_location=lambda storage, loc: storage))
        
#     def build_inputs(self, batch, bs, max_t, actions_onehot):
#         inputs = []
#         inputs.append(batch["obs"][:])
#         actions = actions_onehot[:].reshape(bs, max_t, self.n_agents, -1)
#         inputs.append(actions)
#         inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1))
#         inputs = th.cat([x.reshape(bs, max_t, self.n_agents, -1) for x in inputs], dim=-1)
#         return inputs
    
# def logits_margin(logits, y):
#     comp_logits = logits-th.zeros_like(logits).scatter(2, th.unsqueeze(y, 2), 1e10)
#     sec_logits, _ = th.max(comp_logits, dim=2)
#     # margin=logits.mean(dim=1)
#     margin = sec_logits - th.gather(logits, 2, th.unsqueeze(y, 2)).squeeze(2)
#     margin = margin.mean(1).sum()
#     return margin