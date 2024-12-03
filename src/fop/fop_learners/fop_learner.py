import copy
from components.episode_buffer import EpisodeBatch
# from controllers import basic_controller
# from fop_modules.agents.rnn_agent import RNNAgent
from fop.fop_modules.critics.critic_mer import FOPCritic
from fop.fop_modules.mixers.mix_mer import FOPMixer
import torch.nn.functional as F
import torch as th
from torch.optim import RMSprop
import numpy as np
from torch.distributions import Categorical
# from modules.critics.fop import FOPCritic
from utils.rl_utils import build_td_lambda_targets

class FOP_Learner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.n_agents = args.n_agents
        self.n_actions = args.obs_shape
        self.last_target_update_episode = 0
        self.critic_training_steps = 0

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.critic1 = FOPCritic(scheme, args).to(device= "cuda" if args.use_cuda else "cpu")
        self.critic2 = FOPCritic(scheme, args).to(device= "cuda" if args.use_cuda else "cpu")

        self.mixer1 = FOPMixer(scheme,args).to(device= "cuda" if args.use_cuda else "cpu")
        self.mixer2 = FOPMixer(scheme,args).to(device= "cuda" if args.use_cuda else "cpu")
        
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
        # alpha = max(0.05, 0.5 - t_env / 500000) # linear decay
        # alpha = self.calculate_k_each_round(self.args.kappa_min, self.args.kappa_max, self.args.loop_c, self.args.start_step, self.args.t_max, t_env)

        # alpha = calculate_a_each_round(
        #             k1 = self.args.kappa_min,   # k1
        #             k2 = self.args.kappa_max,   # k2
        #             n = self.args.loop_c,      # n
        #             m1 =self.args.start_step,  # m1
        #             m = self.args.t_max,      # m
        #             qc= t_env
        #             )

        mac_out = []
        log_pi_out = []
        self.actor_hidden_state = mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            chosen_actions_x = mac.forward(batch, t=t, select_actions=True, hidden_states = self.actor_hidden_state)
            # print(chosen_actions_x)
            # aaa
            agent_outs = chosen_actions_x["actions"].view(batch.batch_size,
                                                    self.n_agents,
                                                    self.n_actions)
            self.actor_hidden_state = chosen_actions_x["hidden_state"]
            log_agent_outs = chosen_actions_x["log_prob"].view(batch.batch_size,
                                                    self.n_agents,
                                                    self.n_actions)
            mac_out.append(agent_outs)
            log_pi_out.append(log_agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        log_pi_out = th.stack(log_pi_out, dim=1)

        # # Mask out unavailable actions, renormalise (as in action selection)
        # mac_out[avail_actions == 0] = 1e-10
        # mac_out = mac_out/mac_out.sum(dim=-1, keepdim=True)
        # mac_out[avail_actions == 0] = 1e-10

        pi = mac_out[:,:-1].clone()
        # pi = pi.reshape(-1, self.n_actions)
        log_pi = log_pi_out[:,:-1].clone()
        # log_pi = log_pi.reshape(-1, self.n_actions)

        # log_pi = th.log(pi+1e-8)

        self.critic1.init_hidden(batch.batch_size)
        self.critic2.init_hidden(batch.batch_size)

        inputs = self.critic1._build_inputs(batch, bs, max_t)
        q_vals1, self.critic1.hidden_states = self.critic1.forward(inputs[:, 0:max_t-1], pi.detach(),
                                                                self.critic1.hidden_states)
        q_vals2, self.critic2.hidden_states = self.critic2.forward(inputs[:, 0:max_t-1], pi.detach(),
                                                                self.critic2.hidden_states)


        q_vals = th.min(q_vals1, q_vals2)

        # pi = mac_out[:,:-1].reshape(-1, self.n_actions)
        
        entropies = - (pi * log_pi).sum(dim=-1)
        # print("------------------------",pi.size())
        # print("----------------------------------",log_pi.size())
        # print("----------------------------------------------",q_vals[:,:-1].size())
        # # policy target for discrete actions (from Soft Actor-Critic for Discrete Action Settings)
        # policy_loss = th.mean(alpha * log_pi - q_vals[:,:-1].expand(-1, -1, -1, self.n_actions))

        # policy target for discrete actions (from Soft Actor-Critic for Discrete Action Settings)
        # pol_target = (pi * (alpha * log_pi - q_vals[:,:-1].reshape(-1, self.n_actions))).sum(dim=-1)
        # pol_target = ((self.log_alpha.exp() * log_pi - q_vals[:,:-1].expand(-1, -1, -1, self.n_actions))).sum(dim=-1)
        # pol_target = (pi * (alpha * log_pi - q_vals[:,:-1])).sum(dim=-1)
        # print(log_pi.sum(dim=-1).view(-1).size())
        # print(q_vals.size())
        pol_target = (self.log_alpha.exp() * log_pi.sum(dim=-1).view(-1) - q_vals).sum(dim=-1)

        pol_target = pol_target.view(-1)
        policy_loss = (pol_target * mask).sum() / mask.sum()
        

        # Optimise
        self.p_optimiser.zero_grad()
        policy_loss.backward()
        agent_grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
        self.p_optimiser.step()
        # print(policy_loss)

        # 更新alpha值
        entropy = - log_pi
        alpha_loss = th.mean(
            (entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("adv_policy_loss", policy_loss.item(), t_env)
            self.logger.log_stat("adv_agent_grad_norm", agent_grad_norm, t_env)
            # self.logger.log_stat("pi_max", (pi.max(dim=1)[0] * mask).sum().item() / mask.sum().item(), t_env)
            self.logger.log_stat("alpha", self.log_alpha.exp(), t_env)
            self.logger.log_stat("entropies", entropies.mean().item(), t_env)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, show_demo=False, save_data=None):
        
        self.train_actor(batch, t_env, episode_num)
        self.train_critic(batch, t_env)
        
        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

    def train_critic(self, batch: EpisodeBatch, t_env: int):
        bs = batch.batch_size
        max_t = batch.max_seq_length
        rewards = batch["reward"][:, :-1]
        actions = batch["perturbations"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        actions_onehot = batch["actions_onehot"][:, :-1]
        states = batch["state"]

        mac = self.mac
        mixer1 = self.mixer1
        mixer2 = self.mixer2
        # alpha = max(0.05, 0.5 - t_env / 500000) # linear decay
        # alpha = self.calculate_k_each_round(self.args.kappa_min, self.args.kappa_max, self.args.loop_c, self.args.start_step, self.args.t_max, t_env)
        # alpha = calculate_a_each_round(
        #     k1 = self.args.kappa_min,   # k1
        #     k2 = self.args.kappa_max,   # k2
        #     n = self.args.loop_c,      # n
        #     m1 =self.args.start_step,  # m1
        #     m = self.args.t_max,      # m
        #     qc= t_env
        #     )

        mac_out = []
        log_mac_out = []
        self.critic_hidden_state = mac.init_hidden(batch.batch_size)
        # print(self.critic_hidden_state)
        # aaa
        # mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            # print(t)
            chosen_actions_x = mac.forward(batch, t=t, select_actions=True, hidden_states = self.critic_hidden_state)
            # print(chosen_actions_x)
           
            agent_outs = chosen_actions_x["actions"].view(batch.batch_size,
                                                    self.n_agents,
                                                    self.n_actions)
            log_outs = chosen_actions_x["log_prob"].view(batch.batch_size,
                                                    self.n_agents,
                                                    self.n_actions)
            self.critic_hidden_state = chosen_actions_x["hidden_state"]
            mac_out.append(agent_outs)
            log_mac_out.append(log_outs)

        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        log_mac_out = th.stack(log_mac_out, dim = 1)

        pi = mac_out.clone().detach() 
        log_pi_taken = log_mac_out.clone().detach() 

        # sample actions for next timesteps
        # next_actions = Categorical(pi).sample().long().unsqueeze(3)
        next_actions = pi  #th.zeros(next_actions.squeeze(3).shape + (self.n_actions,))
        if self.args.use_cuda:
            next_actions_ = next_actions.cuda()


        target_inputs = self.target_critic1._build_inputs(batch, bs, max_t)
        self.target_critic1.init_hidden(batch.batch_size)
        self.target_critic2.init_hidden(batch.batch_size)
        target_inputs = self.target_critic1._build_inputs(batch, bs, max_t)

        target_q_vals1, self.target_critic1.hidden_states = self.target_critic1.forward(target_inputs, next_actions.detach(),
                                                                self.target_critic1.hidden_states)#.detach()
        target_q_vals2, self.target_critic2.hidden_states = self.target_critic2.forward(target_inputs, next_actions.detach(),
                                                                self.target_critic2.hidden_states)#.detach()

        # directly caculate the values by definition
        next_vs1 = th.logsumexp(target_q_vals1 / self.log_alpha.exp(), dim=-1) * self.log_alpha.exp()
        next_vs2 = th.logsumexp(target_q_vals2 / self.log_alpha.exp(), dim=-1) * self.log_alpha.exp()
        
        target_qvals1 = self.target_mixer1(target_q_vals1, states, actions=next_actions, vs=next_vs1)
        target_qvals2 = self.target_mixer2(target_q_vals2, states, actions=next_actions, vs=next_vs2)

        target_qvals = th.min(target_qvals1, target_qvals2).reshape(bs, -1, 1)

        # print(terminated.size())
        # Calculate td-lambda targets
        target_v = build_td_lambda_targets(rewards, terminated, mask, target_qvals, self.n_agents, self.args.gamma, self.args.td_lambda)
        # print(log_pi_taken.size())
        # print(target_v.size())# .expand(-1,-1,self.n_agents).unsqueeze(3)
        # print(log_pi_taken[:,1:max_t].sum(dim=-1).sum(dim=-1, keepdim = True).size())
        targets = target_v - (self.log_alpha.exp() * log_pi_taken[:,1:max_t].sum(dim=-1).sum(dim=-1, keepdim = True))
        # target_q_vals1 = self.target_critic1.forward(target_inputs).detach()
        # target_q_vals2 = self.target_critic2.forward(target_inputs).detach()

        # # directly caculate the values by definition
        # next_vs1 = th.logsumexp(target_q_vals1 / self.log_alpha.exp(), dim=-1) * self.log_alpha.exp()
        # next_vs2 = th.logsumexp(target_q_vals2 / self.log_alpha.exp(), dim=-1) * self.log_alpha.exp()
        
        # target_qvals1 = self.target_mixer1(target_q_vals1, states, actions=next_actions, vs=next_vs1)
        # target_qvals2 = self.target_mixer2(target_q_vals2, states, actions=next_actions, vs=next_vs2)

        # target_qvals = th.min(target_qvals1, target_qvals2)

        # # Calculate td-lambda targets
        # target_v = build_td_lambda_targets(rewards, terminated, mask, target_qvals, self.n_agents, self.args.gamma, self.args.td_lambda)
        # # print(target_v.expand(-1,-1,self.n_agents).unsqueeze(3).size())
        # # print((alpha * log_pi_taken.mean(dim=-1, keepdim=True))[:,:-1].size())
        # targets = target_v.expand(-1,-1,self.n_agents).unsqueeze(3) - (self.log_alpha.exp() * log_pi_taken.mean(dim=-1, keepdim=True))[:,:-1]

        # inputs = self.critic1._build_inputs(batch, bs, max_t)
        self.critic1.init_hidden(batch.batch_size)
        self.critic2.init_hidden(batch.batch_size)

        inputs = self.critic1._build_inputs(batch, bs, max_t)
        q_vals1, self.critic1.hidden_states = self.critic1.forward(inputs[:, 0:max_t-1], actions.detach(),
                                                                self.critic1.hidden_states)
        q_vals2, self.critic2.hidden_states = self.critic2.forward(inputs[:, 0:max_t-1], actions.detach(),
                                                                self.critic2.hidden_states)
        # q_vals1 = self.critic1.forward(inputs)
        # q_vals2 = self.critic2.forward(inputs)

        # directly caculate the values by definition
        vs1 = th.logsumexp(q_vals1 / self.log_alpha.exp(), dim=-1) * self.log_alpha.exp()
        vs2 = th.logsumexp(q_vals2 / self.log_alpha.exp(), dim=-1) * self.log_alpha.exp()

        # print(actions.size())
        # print(next_actions.size())

        # q_taken1 = mixer1(q_vals1[:, :-1], states[:, :-1], actions=actions, vs=vs1[:, :-1])
        # q_taken2 = mixer2(q_vals2[:, :-1], states[:, :-1], actions=actions, vs=vs2[:, :-1])
        # print(q_taken1.size())
        q_taken1 = mixer1(q_vals1, states[:, 0:max_t-1], actions=actions.detach(), vs=vs1)
        q_taken2 = mixer2(q_vals2, states[:, 0:max_t-1], actions=actions.detach(), vs=vs2)
        # print(targets.size())
        td_error1 = q_taken1.reshape(bs, -1, 1) - targets.detach()
        td_error2 = q_taken2.reshape(bs, -1, 1) - targets.detach()
        
        # 0-out the targets that came from padded data
        masked_td_error1 = td_error1 * mask
        loss1 = (masked_td_error1 ** 2).sum() / mask.sum() 
        masked_td_error2 = td_error2 * mask
        loss2 = (masked_td_error2 ** 2).sum() / mask.sum() 

        self.c_optimiser1.zero_grad()
        loss1.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params1, self.args.grad_norm_clip)
        self.c_optimiser1.step()
        
        self.c_optimiser2.zero_grad()
        loss2.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params2, self.args.grad_norm_clip)
        self.c_optimiser2.step()

        # aaaaa

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("adv_loss", loss1.item(), t_env)
            self.logger.log_stat("adv_grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("adv_td_error_abs", (masked_td_error1.abs().sum().item() / mask_elems), t_env)
            # print(q_taken1.size())
            # print(mask.size())
            # print(mask_elems.size())
            self.logger.log_stat("adv_q_taken_mean",
                                 (q_taken1.reshape(bs, -1, 1) * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("adv_target_mean", (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
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
        self.critic_optimiser1.load_state_dict(th.load("{}/critic_opt1.th".format(path), map_location=lambda storage, loc: storage))
        self.critic_optimiser2.load_state_dict(th.load("{}/critic_opt2.th".format(path), map_location=lambda storage, loc: storage))
        
    def build_inputs(self, batch, bs, max_t, actions_onehot):
        inputs = []
        inputs.append(batch["obs"][:])
        actions = actions_onehot[:].reshape(bs, max_t, self.n_agents, -1)
        inputs.append(actions)
        inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1))
        inputs = th.cat([x.reshape(bs, max_t, self.n_agents, -1) for x in inputs], dim=-1)
        return inputs
    
def calculate_a_each_round(k1, k2, n, m1, m, qc):
        # print(k1, k2, n, m, m1, q)
        if qc < 1 or qc > m:
            raise ValueError("当前回合数 q 必须在 1 和 总回合数 m 之间")
        if m1 < 1 or m1 > m:
            raise ValueError("起始轮数 m1 必须在 1 和 总回合数 m 之间")

        # 计算每轮的长度
        round_length = (m - m1 + 1) // n

        if qc < m1:
            return k2

        round_index = (qc - m1) // round_length
        round_start = m1 + round_index * round_length
        round_end = round_start + round_length - 1

        # 计算当前回合数在当前轮中的位置
        position_in_round = (qc - round_start) / (round_end - round_start)

        # 计算当前回合数 k 的值，线性减小
        current_k = k2 - position_in_round * (k2 - k1)

        return current_k

