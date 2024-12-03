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
# use_cuda = torch.cuda.is_available()
# # device = torch.device("cuda" if use_cuda else "cpu")
# device = torch.device("cpu")

def pgd(model, X, y, opt, verbose=False, params={}, env_id=""):
    epsilon = params.get('epsilon', 0.075)
    # print(epsilon)
    niters = params.get('niters', 20)
    x_min = X.min()
    x_max = X.max()
    # print(x_min, x_max)
    loss_func = params.get('loss_func', nn.KLDivLoss(reduction='sum',reduce=False))
    # loss_fn = torch.nn.MSELoss(reduce=True, size_average=True) #torch.nn.L1Loss(reduction='sum')
    step_size = 0.01#epsilon * 1.0 / niters
    rand = False
    y = Variable(y, requires_grad=True).to('cpu')
    # X = torch.FloatTensor(X).to('cpu')
    if rand:
        noise_0 = 2 * epsilon * torch.rand(X.size()) - epsilon
        X_adv = X.data + noise_0
        noise_0 = torch.clamp(X_adv.data- X.data, -epsilon, epsilon)
        X_adv = X.data + noise_0
        X_adv = Variable(X_adv, requires_grad=True)
    else:
        X_adv = Variable(X, requires_grad=True)

        for i in range(niters):
            # print(i)
            #logits =  Variable(model.soft(x=X_adv),requires_grad=True)
            logits =  model.soft(x=X_adv)#.requires_grad_(True)#.squeeze(0)
            loss =  F.kl_div(logits.log(),y,reduce="sum")

            opt.zero_grad()
            # print(loss)
            loss.backward()
            eta_0 = step_size * X_adv.grad.data.sign()
            X_adv.data = Variable(X_adv.data + eta_0, requires_grad=True)

            eta_0 = torch.clamp(X_adv.data- X.data, -epsilon, epsilon)
            X_adv.data = X.data + eta_0

            # X_adv.data = torch.clamp(X_adv.data, x_min, x_max)
        print(X_adv - X)
    return X_adv.squeeze(0).data


def fgsm(model, batch, actions, tar_actions,opt, attack_config, t, t_env, hidden_state, verbose=False,  env_id=""):
    loss_func = nn.CrossEntropyLoss()
    epsilon=attack_config.epsilon_ball
    agent_inputs = model._build_inputs(batch, t)
    avail_actions = batch["avail_actions"][:, t]

    X_adv = Variable(agent_inputs.data, requires_grad=True)

    logits, hid = model.soft(X_adv, avail_actions, batch, hidden_state, t_ep=t, t_env=t_env, test_mode=True)
    
    # loss_func = params.get('loss_func', nn.CrossEntropyLoss())
    # y  = torch.LongTensor(np.argmax(y.cpu().data.numpy(), -1))
    # y_target = torch.LongTensor(target).squeeze(0).squeeze(1)
    # loss_1 = loss_func(logits.to('cpu'),torch.LongTensor(y))
    # print(loss_1)
    # print(y.size())
    # aaa

    loss = -loss_func(logits[0],tar_actions[0]) + loss_func(logits[0], actions[0])
    opt.zero_grad()
    loss.backward(retain_graph=True)

    eta_0 = epsilon * X_adv.grad.data.sign()
    X_adv.data = Variable(X_adv.data + eta_0, requires_grad=True)

    eta_0 = torch.clamp(X_adv.data- agent_inputs.data, -epsilon, epsilon)
    X_adv.data = agent_inputs.data + eta_0
    # print(X_adv - X)

    return X_adv.cpu().data.numpy()

def rand_nosie(model, X, y, opt,  available_batch, verbose=False, params={}, env_id=""):
    epsilon=params.get('epsilon', 0.05)

    X_adv = Variable(X.data, requires_grad=True).to('cpu')
    eta_0 = 2 * epsilon * torch.rand(X.size()) - epsilon

    X_adv.data = Variable(X_adv.data + eta_0, requires_grad=True)

    eta_0 = torch.clamp(X_adv.data- X.data.to('cpu'), -epsilon, epsilon)
    X_adv.data = X.data.to('cpu') + eta_0
    # print(X_adv - X)
    return X_adv.cpu().data.numpy()

def attack_target(model, batch, actions, tar_actions, opt, attack_config, t, t_env, hidden_state, loss_func=nn.CrossEntropyLoss()):

    method = attack_config.attack_method
    verbose = attack_config.verbose

    # y = model.soft(obs=X1, agents_available_actions=available_batch)
    if method == 'rand_nosie':
        atk = rand_nosie
    elif method == 'pgd':
        atk = pgd
    else:
        atk = fgsm
    adv_X = atk(model, batch, actions, tar_actions, opt, attack_config, t, t_env, hidden_state, verbose=verbose)
    return adv_X



