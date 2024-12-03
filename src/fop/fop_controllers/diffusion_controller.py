from gym import spaces
import torch.distributions as tdist
import numpy as np
from .basic_controller import BasicMAC
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from components.action_selectors import REGISTRY as action_REGISTRY
from fop.fop_modules.agents.diffusion_agent import Model_RNN, Model_MLP
from fop.fop_modules.agents.helpers import (cosine_beta_schedule,
                            linear_beta_schedule,
                            vp_beta_schedule,
                            extract,
                            Losses)



class DiffusionMAC(nn.Module):
    # def __init__(self, state_dim, action_dim, noise_ratio,
    #              beta_schedule='vp', n_timesteps=1000,
    #              loss_type='l2', clip_denoised=True, predict_epsilon=True):
    def __init__(self, scheme, groups, args):    
        super(DiffusionMAC, self).__init__() # scheme, groups, args
        # noise_ratio = 

        # self.state_dim = input_shape
        # self.action_dim = args.action_dim
        # input_shape = args.
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        # self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type
        if args.action_selector is not None:
            self.action_selector = action_REGISTRY[args.action_selector](args)
        self.hidden_states = None
        # self.args = args
        input_shape = self._get_input_shape(scheme)
        self.action_dim = self.args.obs_shape
        if self.args.diffusionagent == 'rnn':
            self.model = Model_RNN(input_shape, self.args).to(device= "cuda" if args.use_cuda else "cpu")
        elif self.args.diffusionagent == 'mlp':
            self.model = Model_MLP(input_shape, self.args).to(device= "cuda" if args.use_cuda else "cpu")
        self.max_noise_ratio = args.noise_ratio
        self.noise_ratio = args.noise_ratio
        self.beta_schedule = args.beta_schedule
        self.n_timesteps = args.n_timesteps
        self.loss_type = args.loss_type
        self.clip_denoised = args.clip_denoised
        self.predict_epsilon = args.predict_epsilon

        if self.beta_schedule == 'linear':
            betas = linear_beta_schedule(self.n_timesteps)
        elif self.beta_schedule == 'cosine':
            betas = cosine_beta_schedule(self.n_timesteps)
        elif self.beta_schedule == 'vp':
            betas = vp_beta_schedule(self.n_timesteps)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(self.n_timesteps)
        self.clip_denoised = self.clip_denoised
        self.predict_epsilon = self.predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
                             torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
                             betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        self.loss_fn = Losses[self.loss_type]()

    # ------------------------------------------ sampling ------------------------------------------#
    # x_t, t, noise: 随机噪声，时间，评分函数输出
    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return (
                    extract(self.sqrt_recip_alphas_cumprod.to(device= "cuda" if self.args.use_cuda else "cpu"), t, x_t.shape) * x_t -
                    extract(self.sqrt_recipm1_alphas_cumprod.to(device= "cuda" if self.args.use_cuda else "cpu"), t, x_t.shape) * noise
            )
        else:
            return noise
    # # 预测值，随机噪声，时间
    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1.to(device= "cuda" if self.args.use_cuda else "cpu"), t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2.to(device= "cuda" if self.args.use_cuda else "cpu"), t, x_t.shape) * x_t
        )
        # print(self.posterior_variance)
        # aaa
        # print(t)
        posterior_variance = extract(self.posterior_variance.to(device= "cuda" if self.args.use_cuda else "cpu"), t, x_t.shape)
        # print(posterior_variance)
        # aaa
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped.to(device= "cuda" if self.args.use_cuda else "cpu"), t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    # x, t, s, hid_s: 随机噪声，时间，输入状态，隐藏层
    def p_mean_variance(self, x, t, s, hid_s):

        ret = self.model(x, t, s, hid_s)
        # print(x)
        # print(t)
        # print(ret["actions"])
        # # x, t, noise: 随机噪声，时间，评分函数输出
        x_recon = self.predict_start_from_noise(x, t=t, noise=ret["actions"])

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()
        # 预测值，随机噪声，时间
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, ret

    @torch.no_grad()
    def p_sample(self, x, t, s, hid_s):
        # x, t, s, hid_s: 随机噪声，时间，输入状态，隐藏层
        b, *_, device = *x.shape, x.device
        # # x, t, s, hid_s: 随机噪声，时间，输入状态，隐藏层
        model_mean, _, model_log_variance, ret = self.p_mean_variance(x=x, t=t, s=s, hid_s=hid_s)

        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        std = (0.5 * model_log_variance).exp()
        return model_mean + nonzero_mask * std * noise * self.noise_ratio, ret


    @torch.no_grad()
    # state, shape, hid_s：状态，动作维度，隐藏层
    def p_sample_loop(self, state, shape, hid_s):
        device= "cuda" if self.args.use_cuda else "cpu" #self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device)

        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            # x, timesteps, state, hid_s: 随机噪声，时间，输入状态，隐藏层
            x, ret = self.p_sample(x, timesteps, state, hid_s=hid_s)

        return x, ret#, mustd

    @torch.no_grad()
    # state, hid_s：输入状态，隐藏层
    def sample(self, state, hid_s, eval=False):
        self.noise_ratio = 0 if eval else self.max_noise_ratio
        
        batch_size = state.shape[0]
        shape = (batch_size, self.action_dim)
        # state, shape, hid_s：状态，动作维度，隐藏层
        action, ret  = self.p_sample_loop(state, shape, hid_s=hid_s)
        # print(mustd)
        # aaa
        return action.clamp_(-1., 1.), ret

    # ------------------------------------------ training ------------------------------------------#
    # x_start, t, noise: bestactions, t, random_noise
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
                extract(self.sqrt_alphas_cumprod.to(device= "cuda" if self.args.use_cuda else "cpu"), t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod.to(device= "cuda" if self.args.use_cuda else "cpu"), t, x_start.shape) * noise
        ) # 这个代码片段的作用是生成扩散过程中的一个样本。在扩散模型中，通常会通过在每一步对初始数据 bestaction 添加噪声来模拟数据的生成过程。

        return sample
    # # x_start, state, hid_s: bestactions, agent_input, hidden_state
    def p_losses(self, x_start, state, t, hid_s, weights=1.0):
        noise = torch.randn_like(x_start)
        # # x_start, t, noise: bestactions, t, random_noise
        # Forward Process: Add noise to best action
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        # (x, t, s, hid_s) 
        # print(x_noisy.size())
        # print(t.size())
        # print(state.size())
        # 对x_noisy去噪的方向，希望能够和加入的noise尽可能相似
        x_recon = self.model(x_noisy, t, state, hid_s)

        assert noise.shape == x_recon["actions"].shape

        if self.predict_epsilon:
            loss = self.loss_fn(x_recon["actions"], noise, weights)
        else:
            loss = self.loss_fn(x_recon["actions"], x_start, weights)

        return loss

    # x, state, hid_s: bestactions, agent_input, hidden_state
    def loss(self, x, state, hid_s, weights=1.0):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        # x, state, hid_s: bestactions, agent_input, hidden_state
        return self.p_losses(x, state, t, hid_s, weights)

    def forward(self, state, hid_s, eval=False):
        # def sample(self, state, hid_s, eval=False):
        # state: 输入状态，hid_s: 隐藏层 
        return self.sample(state, hid_s, eval)
    
    # (self.batch, t_ep=self.t, t_env=self.t_env, hidden_states=self.adv_hidden_state, test_mode=test_mode)
    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False, past_actions=None, critic=None,
                        target_mac=False, explore_agent_ids=None, hidden_states = None):
        # print(ep_batch["obs"])
        agent_inputs = self._build_inputs(ep_batch, t_ep)
        # print(agent_inputs)
        # aaaaaaa
        state = agent_inputs #torch.FloatTensor(agent_inputs.reshape(1, -1)).to(ep_batch.device)
        # print(state.size())
        # print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        # print(state.size())
        action, ret = self.forward(state, hidden_states, eval)
        # print(action)
        action = (action*self.args.epsilon_ball).cpu().data.numpy().clip(-self.args.epsilon_ball, self.args.epsilon_ball)
        # action = action * self.action_scale + self.action_bias
        # print(action)
        return action, ret["hidden_state"]
    
    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action_adv:
            if getattr(self.args, "discretize_actions", False):
                input_shape += scheme["actions_onehot"]["vshape"][0]
            else:
                # print(scheme["actions"]["vshape"][0])
                input_shape += scheme["obs"]["vshape"]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape
    
    def _build_inputs(self, batch, t, target_mac=False, last_target_action=None):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av


        if self.args.obs_last_action_adv:
            if t == 0:
                inputs.append(torch.zeros_like(batch["perturbations"][:, t]))
            else:
                inputs.append(batch["perturbations"][:, t - 1])
        if self.args.obs_agent_id:
            inputs.append(torch.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = torch.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)

        return inputs


    def init_hidden(self, batch_size):
        hidden_states = self.model.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav
        return hidden_states
    
    def save_models(self, path):
        torch.save(self.model.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(torch.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))