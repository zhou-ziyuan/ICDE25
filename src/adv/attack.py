from pickle import TRUE
from random import random
import torch
from torch import autograd
import torch.nn as nn
from torch.nn.modules import loss
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
TARGET_MULT = 10000.0
use_cuda = torch.cuda.is_available()
# device = torch.device("cuda" if use_cuda else "cpu")
device = torch.device("cpu")

def pgd(model, batch, actions, opt, attack_config, t, t_env, hidden_states,verbose=False,  env_id=""):
    loss_func = nn.CrossEntropyLoss()
    epsilon=attack_config.epsilon_ball
    agent_inputs = model._build_inputs(batch, t)
    niters = attack_config.attack_niters
    # print(agent_inputs)
    step_size = 0.005#epsilon / niters
    avail_actions = batch["avail_actions"][:, t]
    
    X_adv = Variable(agent_inputs.data, requires_grad=True)
    
    noise_0 = 2 * epsilon * torch.rand(agent_inputs.size()) - epsilon
    X_adv = agent_inputs.data + noise_0     
    noise_0 = torch.clamp(X_adv.data- agent_inputs.data, -epsilon, epsilon)
    X_adv = agent_inputs.data + noise_0

    X_adv = Variable(X_adv, requires_grad=True)

    for i in range(niters):
        logits, hid = model.soft(X_adv, avail_actions, batch, hidden_states, t_ep=t, t_env=t_env, test_mode=True)
        loss = loss_func(logits[0], actions[0])
        opt.zero_grad()
        loss.backward(retain_graph=True)
        eta_0 = step_size * X_adv.grad.data.sign()
        X_adv.data = Variable(X_adv.data + eta_0, requires_grad=True)
        eta_0 = torch.clamp(X_adv.data- agent_inputs.data, -epsilon, epsilon)
        X_adv.data = agent_inputs.data + eta_0
    return X_adv.cpu().data.numpy()


def fgsm(model, batch, actions, opt, attack_config, t, t_env, hidden_states,verbose=False,  env_id=""):
    loss_func = nn.CrossEntropyLoss()
    epsilon=attack_config.epsilon_ball
    agent_inputs = model._build_inputs(batch, t)
    # print(agent_inputs)
    avail_actions = batch["avail_actions"][:, t]

    X_adv = Variable(agent_inputs.data, requires_grad=True)

    logits, hid = model.soft(X_adv, avail_actions, batch, hidden_states, t_ep=t, t_env=t_env, test_mode=True)
    # y  = np.argmax(y.cpu().data.numpy(), -1)
    # print(logits[0])
    # print(actions[0])
    loss = loss_func(logits[0], actions[0])
    opt.zero_grad()
    loss.backward(retain_graph=True)
    # print( X_adv.grad)
    eta_0 = epsilon * X_adv.grad.data.sign()
    X_adv.data = Variable(X_adv.data + eta_0, requires_grad=True)

    eta_0 = torch.clamp(X_adv.data- agent_inputs.data, -epsilon, epsilon)
    X_adv.data = agent_inputs.data + eta_0
    # print(X_adv - X)
    return X_adv.cpu().data.numpy()

def rand_noise(model, batch, actions, opt, attack_config, t, t_env, hidden_states,verbose=False,  env_id=""):
    #epsilon=params.get('epsilon', 0.05)
    epsilon=attack_config.epsilon_ball
    agent_inputs = model._build_inputs(batch, t)

    X_adv = Variable(agent_inputs.data, requires_grad=True)
    eta_0 = 2 * epsilon * torch.rand(agent_inputs.size()) - epsilon

    X_adv.data = Variable(X_adv.data + eta_0, requires_grad=True)

    eta_0 = torch.clamp(X_adv.data- agent_inputs.data.to('cpu'), -epsilon, epsilon)
    X_adv.data = agent_inputs.data.to('cpu') + eta_0
    # print(X_adv - X)
    return X_adv.cpu().data.numpy()

def gaussian_noise(model, batch, actions, opt, attack_config, t, t_env, hidden_states,verbose=False,  env_id=""):
    #epsilon=params.get('epsilon', 0.05)
    epsilon=attack_config.epsilon_ball
    agent_inputs = model._build_inputs(batch, t)

    X_adv = Variable(agent_inputs.data, requires_grad=True)
    eta_0 = 2 * epsilon * torch.randn(agent_inputs.size()) - epsilon

    X_adv.data = Variable(X_adv.data + eta_0, requires_grad=True)

    eta_0 = torch.clamp(X_adv.data- agent_inputs.data.to('cpu'), -epsilon, epsilon)
    X_adv.data = agent_inputs.data.to('cpu') + eta_0
    # print(X_adv - X)
    return X_adv.cpu().data.numpy()


def attack_gd(model, batch, actions, opt, attack_config, t, t_env, hidden_states, loss_func=nn.CrossEntropyLoss()):

    method = attack_config.attack_method
    verbose = attack_config.verbose

    # y = model.soft(obs=X1, agents_available_actions=available_batch)
    if method == 'fgsm' or method=='adv_reg':
        atk = fgsm
    elif method == 'pgd':
        atk = pgd
    elif method=='rand_noise':
        atk = rand_noise
    elif method=='gaussian':
        atk =gaussian_noise
    adv_X = atk(model, batch, actions, opt, attack_config, t, t_env, hidden_states, verbose=verbose)
    return adv_X


