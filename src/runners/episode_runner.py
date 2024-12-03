import torch
from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
from adv.attack import attack_gd
from adv.attack_target import attack_target
from adv.attack_atla import get_state
from learners import REGISTRY as le_REGISTRY
import torch.nn.functional as F
import torch.nn as nn
import random
class EpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        self.adv_batch_size = self.args.adv_batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def setup_adv(self, scheme, groups, preprocess, mac, adv_mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac
        self.adv_mac = adv_mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.adv_batch = self.new_batch()
        self.adv_opp_batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def run(self, test_mode=False, learner = None, adv_test = False,adv_learner=None):
        self.reset()

        terminated = False
        episode_return = 0
        self.hidden_state = self.mac.init_hidden(batch_size=self.batch_size)
        if self.args.Number_attack > 0 and (self.args.attack_method == "adv_tar" or self.args.attack_method == "fop_adv_tar" or self.args.attack_method == "mer" or self.args.attack_method == "mer_diffusion"):
            self.adv_hidden_state = self.adv_mac.init_hidden(batch_size=self.batch_size)
            # print()
                
        env_info = self.env.get_env_info()
        obs_shape = env_info["obs_shape"]
        n_agents = env_info["n_agents"]

        while not terminated:

            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }

            # ad_obs 需要有两个batch normal_batch adv_batch
            self.batch.update(pre_transition_data, ts=self.t)    
            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions, hidden_state_true = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, hidden_states=self.hidden_state, test_mode=test_mode)

            ###############################################################

            if self.args.Number_attack > 0 and adv_test:
                if self.args.attack_method == "fgsm" or self.args.attack_method == "pgd" or self.args.attack_method == "adv_reg" or self.args.attack_method=="rand_noise" or self.args.attack_method == "gaussian":
                    adv_inputs = attack_gd(self.mac, self.batch, actions, learner.optimiser, self.args, self.t, self.t_env, self.hidden_state)
                    # print(adv_inputs[:,0:obs_shape] - pre_transition_data["obs"])
                    adv_transition_data = {
                        "state": pre_transition_data["state"],
                        "avail_actions": pre_transition_data["avail_actions"],
                        "obs": [adv_inputs[:,0:obs_shape]]
                    }
                    self.adv_batch.update(adv_transition_data, ts=self.t)
            
                    adv_actions, hidden_state_ = self.mac.select_actions(self.adv_batch, t_ep=self.t, t_env=self.t_env, hidden_states=self.hidden_state, test_mode=test_mode)
                    reward, terminated, env_info = self.env.step(adv_actions[0])
                    episode_return += reward
                    post_transition_data = {
                        "actions": adv_actions,
                        "reward": [(reward,)],
                        "terminated": [(terminated != env_info.get("episode_limit", False),)],
                    }
                    self.adv_batch.update(post_transition_data, ts=self.t)
                    self.batch.update(post_transition_data, ts=self.t)
                    self.hidden_state = hidden_state_
                elif self.args.attack_method == "adv_tar" or self.args.attack_method == "fop_adv_tar":
                    if self.args.attack_method == "fop_adv_tar":
                        tar_actions, adv_hidden_state_, adv_tar_logits = self.adv_mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, hidden_states=self.adv_hidden_state, test_mode=test_mode)
                    else:
                        tar_actions, adv_hidden_state_ = self.adv_mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, hidden_states=self.adv_hidden_state, test_mode=test_mode)
                    adv_inputs = attack_target(self.mac, self.batch, actions, tar_actions, learner.optimiser, self.args, self.t, self.t_env, self.hidden_state)
                    ################*********************##################*******************
                    adv_transition_data = {
                        "state": pre_transition_data["state"],
                        "avail_actions": pre_transition_data["avail_actions"],
                        "obs": [adv_inputs[:,0:obs_shape]]
                    }
                    
                    self.adv_batch.update(adv_transition_data, ts=self.t)
                    self.adv_opp_batch.update(pre_transition_data, ts=self.t) # 攻击者输入正常状态
                    
                    adv_actions, hidden_state_= self.mac.select_actions(self.adv_batch, t_ep=self.t, t_env=self.t_env, hidden_states=self.hidden_state, test_mode=test_mode)
                    reward, terminated, env_info = self.env.step(adv_actions[0])
                    # print(adv_tar_logits, adv_actions[0])
                    # criterion = nn.CrossEntropyLoss()
                    # loss_action = criterion(adv_tar_logits.squeeze(0), adv_actions[0])
                    # print(loss_action)
                    # print(adv_actions - tar_actions)
                    # print()
                    episode_return += reward
                    post_transition_data = {
                        "actions": adv_actions,
                        "reward": [(reward,)],
                        "terminated": [(terminated != env_info.get("episode_limit", False),)],
                    }
                    # print(reward)
                    # print(loss_action)
                    opp_post_transition_data = {
                        "actions": tar_actions,
                        "reward": [(-reward,)], # 可做修改
                        "terminated": [(terminated != env_info.get("episode_limit", False),)],
                    }
                    self.adv_batch.update(post_transition_data, ts=self.t)
                    self.adv_opp_batch.update(opp_post_transition_data, ts=self.t) 
                    self.batch.update(post_transition_data, ts=self.t)
                    self.hidden_state = hidden_state_
                    self.adv_hidden_state = adv_hidden_state_
            ################################################################
                elif self.args.attack_method == "atla":
                    # TODO 对智能体的观测状态进行攻击
                    # ori_inputs= get_state(self.mac, self.batch, actions, learner.optimiser, self.args, self.t, self.t_env, self.hidden_state)
                    # mu,std=adv_learner.policy_net()
                    # adv_inputs=ori_inputs+mu
                    X = torch.tensor(pre_transition_data["obs"])  # 扰动前的观测

                    mu,sigma=adv_learner.actor_net(X)
                    # perturbations=[torch.rand(size=(obs_shape,)).cpu().data.numpy() for i in range(self.args.n_agents)] #扰动值，测试用
                    perturbations=[(F.hardtanh(torch.distributions.Normal(torch.squeeze(mu,0)[0],torch.squeeze(sigma,0)[0]).sample())*self.args.epsilon_ball).cpu().data.numpy() for i in range(self.args.n_agents)] #扰动值
                    
                    
                    adv_inputs_obs=np.array(X).squeeze(0)+np.array(perturbations) #扰动后观测
                    attacked_agent_id =  random.sample(range(0, n_agents), self.args.Number_attack)
                    adv_inputs = pre_transition_data["obs"][0].copy()
                    for i in range (self.args.Number_attack):
                    # print(adv_inputs[0])
                        adv_inputs[attacked_agent_id[i]] = adv_inputs_obs[attacked_agent_id[i]].copy()

                    ################*********************##################*******************
                    adv_transition_data = {
                        "state": pre_transition_data["state"],
                        "avail_actions": pre_transition_data["avail_actions"],
                        "obs": [arr[:obs_shape] for arr in adv_inputs]
                    }
                    self.adv_batch.update(adv_transition_data, ts=self.t) # 被攻击者输入被扰动的状态，用于训练鲁棒性网络
                    #self.adv_opp_batch.update(pre_transition_data, ts=self.t)  # 攻击者输入正常状态，用于训练对抗网络

                    adv_actions, hidden_state_ = self.mac.select_actions(self.adv_batch, t_ep=self.t, t_env=self.t_env,
                                                                         hidden_states=self.hidden_state,
                                                                         test_mode=test_mode)
                    reward, terminated, env_info = self.env.step(adv_actions[0])

                    episode_return += reward
                    post_transition_data = {
                        "actions": adv_actions,
                        "reward": [(reward,)],
                        "terminated": [(terminated != env_info.get("episode_limit", False),)],
                    }

                    opp_post_transition_data = {
                        "obs": pre_transition_data["obs"], # 智能体本来应该观测到的状态
                        "perturbations":[perturbations], # 扰动的状态，等效于mappo输出的动作
                        "reward": [(-reward,)],  # 可做修改，产生的奖励，是多智能体奖励的取反
                        "terminated": [(terminated != env_info.get("episode_limit", False),)],
                    }
                    self.adv_batch.update(post_transition_data, ts=self.t)
                    self.adv_opp_batch.update(opp_post_transition_data, ts=self.t)
                    self.batch.update(post_transition_data, ts=self.t)
                    self.hidden_state = hidden_state_
                    
                    # self.adv_hidden_state = adv_hidden_state_
                ######################################################################
                elif self.args.attack_method == "mer":
                    # TODO 对智能体的观测状态进行攻击
                    # ori_inputs= get_state(self.mac, self.batch, actions, learner.optimiser, self.args, self.t, self.t_env, self.hidden_state)
                    # mu,std=adv_learner.policy_net()
                    # adv_inputs=ori_inputs+mu
                    X = torch.tensor(pre_transition_data["obs"])  # 扰动前的观测
                    # print(X.size())
                    # aaa
                    # mu,sigma=adv_learner.actor_net(X)
                    # perturbations = adv_learner.mac.forward(batch, t=t)
                    perturbations, adv_hidden_state_ = self.adv_mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, hidden_states=self.adv_hidden_state, test_mode=test_mode)
                    # perturbations=[torch.rand(size=(obs_shape,)).cpu().data.numpy() for i in range(self.args.n_agents)] #扰动值，测试用
                    # perturbations=[(F.hardtanh(torch.distributions.Normal(torch.squeeze(mu,0)[0],torch.squeeze(sigma,0)[0]).sample())*self.args.epsilon_ball).cpu().data.numpy() for i in range(self.args.n_agents)] #扰动值
                    
                    # print(perturbations)
                    # print(X)
                    adv_inputs_obs=np.array(X.cpu()).squeeze(0)+np.array(perturbations.cpu()) #扰动后观测

                    attacked_agent_id = random.sample(range(0, n_agents), self.args.Number_attack)
                    adv_inputs = pre_transition_data["obs"][0].copy()
                    # print(adv_inputs)
                    # print(adv_inputs_obs)
                    for i in range (self.args.Number_attack):
                    # print(adv_inputs[0])
                        adv_inputs[attacked_agent_id[i]] = adv_inputs_obs[0][attacked_agent_id[i]].copy()

                    ################*********************##################*******************
                    # print(adv_inputs)
                    adv_transition_data = {
                        "state": pre_transition_data["state"],
                        "avail_actions": pre_transition_data["avail_actions"],
                        "obs": [arr[:obs_shape] for arr in adv_inputs]
                    }
                    self.adv_batch.update(adv_transition_data, ts=self.t) # 被攻击者输入被扰动的状态，用于训练鲁棒性网络
                    self.adv_opp_batch.update(pre_transition_data, ts=self.t)  # 攻击者输入正常状态，用于训练对抗网络

                    adv_actions, hidden_state_ = self.mac.select_actions(self.adv_batch, t_ep=self.t, t_env=self.t_env,
                                                                         hidden_states=self.hidden_state,
                                                                         test_mode=test_mode)
                    reward, terminated, env_info = self.env.step(adv_actions[0])

                    episode_return += reward
                    post_transition_data = {
                        "actions": adv_actions,
                        "reward": [(reward,)],
                        "terminated": [(terminated != env_info.get("episode_limit", False),)],
                    }

                    opp_post_transition_data = {
                        "obs": pre_transition_data["obs"], # 智能体本来应该观测到的状态
                        "perturbations":perturbations, # 扰动的状态，等效于mappo输出的动作
                        "reward": [(-reward,)],  # 可做修改，产生的奖励，是多智能体奖励的取反
                        "terminated": [(terminated != env_info.get("episode_limit", False),)],
                    }
                    self.adv_batch.update(post_transition_data, ts=self.t)
                    self.adv_opp_batch.update(opp_post_transition_data, ts=self.t)
                    self.batch.update(post_transition_data, ts=self.t)
                    self.hidden_state = hidden_state_
                    self.adv_hidden_state = adv_hidden_state_
                    # adv_learner.append_diffusion_memory(np.array(pre_transition_data["obs"]).reshape(-1), np.array(perturbations).reshape(-1))
                    # self.diffusion_buffer.append(opp_post_transition_data["obs"], opp_post_transition_data["perturbations"]) # .view(-1, self.obs_shape)
                elif self.args.attack_method == "mer_diffusion":
                    # TODO 对智能体的观测状态进行攻击
                    # ori_inputs= get_state(self.mac, self.batch, actions, learner.optimiser, self.args, self.t, self.t_env, self.hidden_state)
                    # mu,std=adv_learner.policy_net()
                    # adv_inputs=ori_inputs+mu
                    X = torch.tensor(pre_transition_data["obs"])  # 扰动前的观测
                    # print(X.size())
                    # aaa
                    # mu,sigma=adv_learner.actor_net(X)
                    # perturbations = adv_learner.mac.forward(batch, t=t)
                    perturbations, adv_hidden_state_ = self.adv_mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, hidden_states=self.adv_hidden_state, test_mode=test_mode)
                    # perturbations=[torch.rand(size=(obs_shape,)).cpu().data.numpy() for i in range(self.args.n_agents)] #扰动值，测试用
                    # perturbations=[(F.hardtanh(torch.distributions.Normal(torch.squeeze(mu,0)[0],torch.squeeze(sigma,0)[0]).sample())*self.args.epsilon_ball).cpu().data.numpy() for i in range(self.args.n_agents)] #扰动值
                    
                    # print(perturbations)
                    # print(X)
                    adv_inputs_obs=np.array(X).squeeze(0)+np.array(perturbations) #扰动后观测

                    attacked_agent_id =  random.sample(range(0, n_agents), self.args.Number_attack)
                    adv_inputs = pre_transition_data["obs"][0].copy()
                    # print(adv_inputs)
                    # print(adv_inputs_obs)
                    for i in range (self.args.Number_attack):
                    # print(adv_inputs[0])
                        adv_inputs[attacked_agent_id[i]] = adv_inputs_obs[attacked_agent_id[i]].copy()

                    ################*********************##################*******************
                    # print(adv_inputs)
                    adv_transition_data = {
                        "state": pre_transition_data["state"],
                        "avail_actions": pre_transition_data["avail_actions"],
                        "obs": [arr[:obs_shape] for arr in adv_inputs]
                    }
                    self.adv_batch.update(adv_transition_data, ts=self.t) # 被攻击者输入被扰动的状态，用于训练鲁棒性网络
                    self.adv_opp_batch.update(pre_transition_data, ts=self.t)  # 攻击者输入正常状态，用于训练对抗网络

                    adv_actions, hidden_state_ = self.mac.select_actions(self.adv_batch, t_ep=self.t, t_env=self.t_env,
                                                                         hidden_states=self.hidden_state,
                                                                         test_mode=test_mode)
                    reward, terminated, env_info = self.env.step(adv_actions[0])

                    episode_return += reward
                    post_transition_data = {
                        "actions": adv_actions,
                        "reward": [(reward,)],
                        "terminated": [(terminated != env_info.get("episode_limit", False),)],
                    }

                    opp_post_transition_data = {
                        "obs": pre_transition_data["obs"], # 智能体本来应该观测到的状态
                        "perturbations":perturbations, 
                        "reward": [(-reward,)],  # 可做修改，产生的奖励，是多智能体奖励的取反
                        "terminated": [(terminated != env_info.get("episode_limit", False),)],
                    }
                    self.adv_batch.update(post_transition_data, ts=self.t)
                    self.adv_opp_batch.update(opp_post_transition_data, ts=self.t)
                    self.batch.update(post_transition_data, ts=self.t)
                    self.hidden_state = hidden_state_
                    self.adv_hidden_state = adv_hidden_state_
                    # adv_learner.append_diffusion_memory(np.array(pre_transition_data["obs"]).reshape(-1), np.array(perturbations).reshape(-1))
                    # self.diffusion_buffer.append(opp_post_transition_data["obs"], opp_post_transition_data["perturbations"]) # .view(-1, self.obs_shape)
            else:
                reward, terminated, env_info = self.env.step(actions[0])
                episode_return += reward

                post_transition_data = {
                    "actions": actions,
                    "reward": [(reward,)],
                    "terminated": [(terminated != env_info.get("episode_limit", False),)],
                }
                self.batch.update(post_transition_data, ts=self.t)
                self.hidden_state = hidden_state_true
            self.t += 1

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        self.batch.update(last_data, ts=self.t)
        actions,hid = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, hidden_states=self.hidden_state, test_mode=test_mode)
        # print(actions)
        self.batch.update({"actions": actions}, ts=self.t)
        if self.args.Number_attack > 0 and adv_test:
            if self.args.attack_method == "fgsm" or self.args.attack_method == "pgd" or self.args.attack_method == "adv_reg" or self.args.attack_method=="rand_noise" or self.args.attack_method == "gaussian":
                adv_inputs = attack_gd(self.mac, self.batch, actions, learner.optimiser, self.args, self.t, self.t_env, hidden_states=self.hidden_state)
                
                adv_last_data = {
                    "state": last_data["state"],
                    "avail_actions": last_data["avail_actions"],
                    "obs": [adv_inputs[:,0:obs_shape]]
                }
                self.adv_batch.update(adv_last_data, ts=self.t)
                adv_actions, hid = self.mac.select_actions(self.adv_batch, t_ep=self.t, t_env=self.t_env, hidden_states=self.hidden_state, test_mode=test_mode)
                self.adv_batch.update({"actions": adv_actions}, ts=self.t)

            elif self.args.attack_method == "adv_tar" or self.args.attack_method == "fop_adv_tar":
                if self.args.attack_method == "fop_adv_tar":
                    tar_actions, hid, _ = self.adv_mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, hidden_states=self.adv_hidden_state, test_mode=test_mode)
                else:
                    tar_actions, hid = self.adv_mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, hidden_states=self.adv_hidden_state, test_mode=test_mode)
                adv_inputs = attack_target(self.mac, self.batch, actions, tar_actions, learner.optimiser, self.args, self.t, self.t_env, hidden_state=self.hidden_state)
                adv_last_data = {
                    "state": last_data["state"],
                    "avail_actions": last_data["avail_actions"],
                    "obs": [adv_inputs[:,0:obs_shape]]
                }
                self.adv_batch.update(adv_last_data, ts=self.t)
                adv_actions, hid = self.mac.select_actions(self.adv_batch, t_ep=self.t, t_env=self.t_env, hidden_states=self.hidden_state, test_mode=test_mode)
                self.adv_batch.update({"actions": adv_actions}, ts=self.t)
                self.adv_opp_batch.update(last_data, ts=self.t)
                self.adv_opp_batch.update({"actions": tar_actions}, ts=self.t)
            elif self.args.attack_method == "atla":
                # TODO 对智能体的观测状态进行攻击
                # ori_inputs= get_state(self.mac, self.batch, actions, learner.optimiser, self.args, self.t, self.t_env, self.hidden_state)
                # mu,std=adv_learner.policy_net()
                # adv_inputs=ori_inputs+mu
                X = torch.tensor(last_data["obs"])  # 扰动前的观测
                mu, sigma = adv_learner.actor_net(X)
                # perturbations=[torch.rand(size=(obs_shape,)).cpu().data.numpy() for i in range(self.args.n_agents)] #扰动值，测试用
                perturbations = [(F.hardtanh(torch.distributions.Normal(torch.squeeze(mu, 0)[0],
                                                                        torch.squeeze(sigma, 0)[
                                                                            0]).sample()) * self.args.epsilon_ball).cpu().data.numpy()
                                 for i in range(self.args.n_agents)]  # 扰动值
                adv_inputs = np.array(X).squeeze(0) + np.array(perturbations)  # 扰动后观测
                adv_last_data = {
                    "state": last_data["state"],
                    "avail_actions": last_data["avail_actions"],
                    "obs": [adv_inputs[:,0:obs_shape]]
                }
                self.adv_batch.update(adv_last_data, ts=self.t)
                adv_actions, hid = self.mac.select_actions(self.adv_batch, t_ep=self.t, t_env=self.t_env, hidden_states=self.hidden_state, test_mode=test_mode)
                self.adv_batch.update({"actions": adv_actions}, ts=self.t)
                self.adv_opp_batch.update(last_data, ts=self.t)
                # self.adv_opp_batch.update({"perturbations": [perturbations]}, ts=self.t)

            elif self.args.attack_method == "mer":
                # TODO 对智能体的观测状态进行攻击

                # X = torch.tensor(last_data["obs"])  # 扰动前的观测
                # mu, sigma = adv_learner.actor_net(X)
                # # perturbations=[torch.rand(size=(obs_shape,)).cpu().data.numpy() for i in range(self.args.n_agents)] #扰动值，测试用
                # perturbations = [(F.hardtanh(torch.distributions.Normal(torch.squeeze(mu, 0)[0],
                #                                                         torch.squeeze(sigma, 0)[
                #                                                             0]).sample()) * self.args.epsilon_ball).cpu().data.numpy()
                #                  for i in range(self.args.n_agents)]  # 扰动值
                # adv_inputs = np.array(X).squeeze(0) + np.array(perturbations)  # 扰动后观测
                X = torch.tensor(pre_transition_data["obs"])  # 扰动前的观测
                    
                perturbations, adv_hidden_state_ = self.adv_mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, hidden_states=self.adv_hidden_state, test_mode=test_mode)
                    
                adv_inputs_obs=np.array(X).squeeze(0)+np.array(perturbations.cpu()) #扰动后观测
                attacked_agent_id =  random.sample(range(0, n_agents), self.args.Number_attack)
                adv_inputs = pre_transition_data["obs"][0].copy()
                    
                for i in range (self.args.Number_attack):

                    adv_inputs[attacked_agent_id[i]] = adv_inputs_obs[0][attacked_agent_id[i]].copy()
                adv_last_data = {
                    "state": last_data["state"],
                    "avail_actions": last_data["avail_actions"],
                    "obs": [arr[:obs_shape] for arr in adv_inputs]
                }
                self.adv_batch.update(adv_last_data, ts=self.t)
                adv_actions, hid = self.mac.select_actions(self.adv_batch, t_ep=self.t, t_env=self.t_env, hidden_states=self.hidden_state, test_mode=test_mode)
                self.adv_batch.update({"actions": adv_actions}, ts=self.t)
                self.adv_opp_batch.update(last_data, ts=self.t)
                # self.adv_opp_batch.update({"perturbations": [perturbations]}, ts=self.t)
            elif self.args.attack_method == "mer_diffusion":
                # TODO 对智能体的观测状态进行攻击

                # X = torch.tensor(last_data["obs"])  # 扰动前的观测
                # mu, sigma = adv_learner.actor_net(X)
                # # perturbations=[torch.rand(size=(obs_shape,)).cpu().data.numpy() for i in range(self.args.n_agents)] #扰动值，测试用
                # perturbations = [(F.hardtanh(torch.distributions.Normal(torch.squeeze(mu, 0)[0],
                #                                                         torch.squeeze(sigma, 0)[
                #                                                             0]).sample()) * self.args.epsilon_ball).cpu().data.numpy()
                #                  for i in range(self.args.n_agents)]  # 扰动值
                # adv_inputs = np.array(X).squeeze(0) + np.array(perturbations)  # 扰动后观测
                X = torch.tensor(pre_transition_data["obs"])  # 扰动前的观测
                    
                perturbations, adv_hidden_state_ = self.adv_mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, hidden_states=self.adv_hidden_state, test_mode=test_mode)
                # print(perturbations)
                adv_inputs_obs=np.array(X).squeeze(0)+np.array(perturbations) #扰动后观测
                attacked_agent_id =  random.sample(range(0, n_agents), self.args.Number_attack)
                adv_inputs = pre_transition_data["obs"][0].copy()
                    
                for i in range (self.args.Number_attack):

                    adv_inputs[attacked_agent_id[i]] = adv_inputs_obs[attacked_agent_id[i]].copy()
                adv_last_data = {
                    "state": last_data["state"],
                    "avail_actions": last_data["avail_actions"],
                    "obs": [arr[:obs_shape] for arr in adv_inputs]
                }
                self.adv_batch.update(adv_last_data, ts=self.t)
                adv_actions, hid = self.mac.select_actions(self.adv_batch, t_ep=self.t, t_env=self.t_env, hidden_states=self.hidden_state, test_mode=test_mode)
                self.adv_batch.update({"actions": adv_actions}, ts=self.t)
                self.adv_opp_batch.update(last_data, ts=self.t)
    
        # Select actions in the last stored state  

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""

        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)
        # print(cur_stats["battle_won"])
        if self.args.evaluate:
            print(episode_return,'-------------', cur_stats["battle_won"])
        if test_mode and (len(self.test_returns) == self.args.test_nepisode - 1):
            self._log(cur_returns, cur_stats, log_prefix)

        if self.args.Number_attack > 0 and adv_test:
            if self.args.attack_method == "fgsm" or self.args.attack_method == "pgd" or self.args.attack_method=="rand_noise" or self.args.attack_method == "gaussian":
                return self.adv_batch
            elif self.args.attack_method=="adv_reg":
                return self.batch,self.adv_batch
            elif self.args.attack_method == "adv_tar" or self.args.attack_method == "fop_adv_tar" or self.args.attack_method == "atla" or self.args.attack_method == "mer":
                return self.adv_batch, self.adv_opp_batch
            elif self.args.attack_method == "mer_diffusion":
                return self.adv_batch, self.adv_opp_batch
        else:
            return self.batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)

        returns.clear()
        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)

        stats.clear()
