import copy
from components.episode_buffer import EpisodeBatch, DiffusionMemory
# from controllers import basic_controller
# from fop_modules.agents.rnn_agent import RNNAgent
import torch
from fop.fop_modules.critics.critic_mer import FOPCritic
from fop.fop_modules.mixers.mix_mer import FOPMixer
import torch.nn as nn
import torch.nn.functional as F
import torch as th
from torch.optim import RMSprop
import numpy as np
from torch.distributions import Categorical
# from modules.critics.fop import FOPCritic
from utils.rl_utils import build_td_lambda_targets
# from fop.fop_modules.agents.diffusion_agent import Diffusion
class FOP_Diffusion_Learner:
    def __init__(self, mac, scheme, logger, args, diffusion_buffer):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.n_agents = args.n_agents
        self.n_actions = args.obs_shape
        self.last_target_update_episode = 0
        self.critic_training_steps = 0
        self.action_lr = args.action_lr
        self.action_gradient_steps = args.action_gradient_steps

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
        
        self.target_entropy = -self.n_actions * self.n_agents
        self.diffusion_buffer = diffusion_buffer

    def train_actor_crrtic(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        bs = batch.batch_size
        max_t = batch.max_seq_length
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        # print(mask[:, 1:])
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        avail_actions = batch["avail_actions"]
        rewards = batch["reward"][:, :-1]
        actions = batch["perturbations"][:, :-1]
        states = batch["state"]
        # terminated = batch["terminated"][:, :-1].float()
        # print(actions.size())
        # print(states.size())

        mac = self.mac

        mixer1 = self.mixer1
        mixer2 = self.mixer2

        mac_out = []
        log_mac_out = []
        hidden_state = []
        # actor_inputs = []
        self.critic_hidden_state = mac.init_hidden(batch.batch_size)

        for t in range(batch.max_seq_length):

            state = mac._build_inputs(batch, t)
            # actor_inputs.append(state)
            out, chosen_actions_x= mac.forward(state = state, hid_s = self.critic_hidden_state, eval=False)
            # print(chosen_actions_x)
            agent_outs = out.view(batch.batch_size, self.n_agents, self.n_actions)
            log_outs = chosen_actions_x["log_prob"].view(batch.batch_size,
                                                    self.n_agents,
                                                    self.n_actions)
            # log_outs = F.softmax(agent_outs)
            # print(mustd[0])
            # print(mustd[1])
            # aaa
            # dist = torch.distributions.Normal(mustd[0], mustd[1])#.rsample()

            # log_outs = dist.log_prob(out).view(batch.batch_size, self.n_agents, self.n_actions)
            # # log_outs = th.log(agent_outs)
            # print(log_outs)
            # aaa
            self.critic_hidden_state = chosen_actions_x["hidden_state"]
            mac_out.append(agent_outs)
            log_mac_out.append(log_outs)
            hidden_state.append(self.critic_hidden_state)
        # aaa
        mac_out_critic = th.stack(mac_out, dim=1)  # Concat over time
        log_mac_out_critic = th.stack(log_mac_out, dim = 1)
        # print(log_mac_out_critic)

        pi = mac_out_critic.clone().detach() 
        log_pi_taken = log_mac_out_critic.clone().detach() 

        # sample actions for next timesteps
        # next_actions = Categorical(pi).sample().long().unsqueeze(3)
        next_actions = pi  #th.zeros(next_actions.squeeze(3).shape + (self.n_actions,))

        if self.args.use_cuda:
            next_actions_ = next_actions.cuda()
        self.critic1.init_hidden(batch.batch_size)
        self.critic2.init_hidden(batch.batch_size)

        inputs = self.critic1._build_inputs(batch, bs, max_t)
        q_vals1, self.critic1.hidden_states = self.critic1.forward(inputs[:, 0:max_t-1], actions.detach(),
                                                                self.critic1.hidden_states)
        q_vals2, self.critic2.hidden_states = self.critic2.forward(inputs[:, 0:max_t-1], actions.detach(),
                                                                self.critic2.hidden_states)

        # directly caculate the values by definition
        vs1 = th.logsumexp(q_vals1 / self.log_alpha.exp(), dim=-1) * self.log_alpha.exp()
        vs2 = th.logsumexp(q_vals2 / self.log_alpha.exp(), dim=-1) * self.log_alpha.exp()
        # print(q_vals1)
        # print(vs1)
        # print(states[:, 0:max_t-1])
        # print(actions)
        q_taken1 = mixer1(q_vals1, states[:, 0:max_t-1], actions=actions.detach(), vs=vs1)
        q_taken2 = mixer2(q_vals2, states[:, 0:max_t-1], actions=actions.detach(), vs=vs2)
        # print(q_taken1)
        # sss
        self.target_critic1.init_hidden(batch.batch_size)
        self.target_critic2.init_hidden(batch.batch_size)
        target_inputs = self.target_critic1._build_inputs(batch, bs, max_t)

        target_q_vals1, self.target_critic1.hidden_states = self.target_critic1.forward(inputs, next_actions.detach(),
                                                                self.target_critic1.hidden_states)#.detach()
        target_q_vals2, self.target_critic2.hidden_states = self.target_critic2.forward(inputs, next_actions.detach(),
                                                                self.target_critic2.hidden_states)#.detach()

        # directly caculate the values by 
        # print(target_q_vals2)
        next_vs1 = th.logsumexp(target_q_vals1 / self.log_alpha.exp(), dim=-1) * self.log_alpha.exp()
        next_vs2 = th.logsumexp(target_q_vals2 / self.log_alpha.exp(), dim=-1) * self.log_alpha.exp()
        # print(next_vs1)
        target_qvals1 = self.target_mixer1(target_q_vals1, states, actions=next_actions, vs=next_vs1)
        target_qvals2 = self.target_mixer2(target_q_vals2, states, actions=next_actions, vs=next_vs2)
        # print(target_qvals1)
        # aaa
        target_qvals = th.min(target_qvals1, target_qvals2).reshape(bs, -1, 1)
        # print(target_qvals)
        # print(terminated.size())
        # Calculate td-lambda targets
        target_v = build_td_lambda_targets(rewards, terminated, mask, target_qvals, self.n_agents, self.args.gamma, self.args.td_lambda)
        # print(log_pi_taken.size())

        # print(target_v.size())# .expand(-1,-1,self.n_agents).unsqueeze(3)
        # print(mac_out_critic)
        # print(log_pi_taken[:,1:max_t])
        # aaa
        targets = target_v - (self.log_alpha.exp() * log_pi_taken[:,1:max_t].sum(dim=-1).sum(dim=-1, keepdim = True))
        # print(targets)
        # print(targets.size()) # 32 60 30
        # print(q_taken1)
        targets = target_v - (self.log_alpha.exp() * log_pi_taken[:,1:max_t].sum(dim=-1).sum(dim=-1, keepdim = True))
        # print(targets)
        # aaa
        # print(mask.size())
        td_error1 = q_taken1.reshape(bs, -1, 1) - targets.detach()
        td_error2 = q_taken2.reshape(bs, -1, 1) - targets.detach()
        # print(td_error1)
        # 0-out the targets that came from padded data
        masked_td_error1 = td_error1 * mask
        loss1 = (masked_td_error1 ** 2).sum() / mask.sum() 
        masked_td_error2 = td_error2 * mask
        loss2 = (masked_td_error2 ** 2).sum() / mask.sum() 
        # print(loss1)
        self.c_optimiser1.zero_grad()
        loss1.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params1, self.args.grad_norm_clip)
        self.c_optimiser1.step()
        
        self.c_optimiser2.zero_grad()
        loss2.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params2, self.args.grad_norm_clip)
        self.c_optimiser2.step()
        # qqq

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("adv_loss", loss1.item(), t_env)
            self.logger.log_stat("adv_grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("adv_td_error_abs", (masked_td_error1.abs().sum().item() / mask_elems), t_env)
            self.logger.log_stat("adv_q_taken_mean",
                                 (q_taken1.reshape(bs, -1, 1) * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("adv_target_mean", (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
                                 t_env)

        """ Policy Training """
        mask = mask.repeat(1, 1, self.n_agents).view(-1)
        mac_out_actor = th.stack(mac_out, dim=1)  # Concat over time
        log_pi_out_actor = th.stack(log_mac_out, dim=1)

        pi_actor = mac_out_actor[:,:-1].clone()
        log_pi_actor = log_pi_out_actor[:,:-1].clone()

        entropies = - (pi_actor * log_pi_actor).sum(dim=-1)
        best_actions = actions.clone().detach()
        actions_optim = th.optim.Adam([best_actions], lr=self.action_lr, eps=1e-5)
        for i in range(self.action_gradient_steps):
            # Detach and re-create the best_actions tensor
            best_actions = best_actions.clone().detach().requires_grad_(True)

            best_q_vals1, self.critic1.hidden_states = self.critic1.forward(inputs[:, 0:max_t-1], best_actions.detach(),
                                                                self.critic1.hidden_states)
            best_q_vals2, self.critic2.hidden_states = self.critic2.forward(inputs[:, 0:max_t-1], best_actions.detach(),
                                                                self.critic2.hidden_states)
            
            best_q_taken1 = mixer1(best_q_vals1, states[:, :-1], actions=best_actions, vs=vs1)
            best_q_taken2 = mixer2(best_q_vals2, states[:, :-1], actions=best_actions, vs=vs2)
            qvals = th.min(best_q_taken1, best_q_taken2).reshape(bs, -1, 1)

            best_actions_loss = th.mean(self.log_alpha.exp() * log_pi_actor.sum(dim=-1).sum(dim=-1, keepdim = True) - qvals) #+  

            actions_optim.zero_grad()

            # Ensure retain_graph=True if multiple backward passes are needed
            best_actions_loss.backward(retain_graph=True) # th.ones_like(best_actions_loss), retain_graph=True

            actions_grad_norms = nn.utils.clip_grad_norm_([best_actions], max_norm=self.args.grad_norm_clip, norm_type=2)

            actions_optim.step()

            # Replace in-place clamp operation with out-of-place clamp
            best_actions = best_actions.clamp(-1., 1.).detach()
            best_actions.requires_grad_(False)
        # Detach after optimization step to avoid accumulating gradients
        best_actions = best_actions.detach()
        # mac._build_inputs(batch, t)
        # actor_inputs =  mac._build_inputs(batch, bs, max_t)[:, :max_t-1, :, :]
        actor_inputs = self.critic1._build_inputs(batch, bs, max_t)[:, :max_t-1, :, :]
        dim1 = th.prod(th.tensor(best_actions.shape[:-1])) 
        dim2 = th.prod(th.tensor(actor_inputs.shape[:-1])) 
        # print(hidden_state[0].reshape(bs, self.n_agents, -1).size(),hidden_state[1].size())
        if self.args.diffusionagent == 'rnn':
            hidden_state = th.cat([x.reshape(bs*self.n_agents, -1) for x in hidden_state[:-2]], dim=-1)

        policy_loss = mac.loss(best_actions.reshape(dim1, best_actions.shape[-1]), actor_inputs.reshape(dim2, actor_inputs.shape[-1]), hidden_state)

        # Optimise
        self.p_optimiser.zero_grad()
        policy_loss.backward()
        agent_grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
        self.p_optimiser.step()
        # print(policy_loss)

        # 更新alpha值
        entropy = - log_pi_actor
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

            
    
            self.log_stats_t = t_env

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, show_demo=False, save_data=None):
        
        self.train_actor_crrtic(batch, t_env, episode_num)
        # self.train_critic(batch, t_env)
        
        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

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
    
    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            if getattr(self.args, "discretize_actions", False):
                input_shape += scheme["actions_onehot"]["vshape"][0]
            else:
                input_shape += scheme["actions"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape
    
    def append_diffusion_memory(self, state, action):
        # action = (action - self.action_bias) / self.action_scale
        
        # self.memory.append(state, action, reward, next_state, mask)
        self.diffusion_buffer.append(state, action)
    
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

    