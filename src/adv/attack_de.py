import os
#from socket import ALG_SET_AEAD_ASSOCLEN
import sys
import numpy as np

import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from torch.autograd import Variable

from .differential_evolution import differential_evolution

def perturb_actions(xs, actions): # 扰动动作

	if xs.ndim < 2:
		xs = np.array([xs])
	batch = len(xs)
	actions = actions.repeat(batch, 1,1)
	xs = xs.astype(int)

	count = 0
	for x in xs:
		pixels = np.split(x, len(x)/2)
		
		for pixel in pixels:
			x_pos, r = pixel
			# print('1',actions.data[count, x_pos])
			
			actions.data[count, x_pos] = torch.tensor(r)
			# actions[count, x_pos] = torch.LongTensor(r)#.to(device)
			# print('2',actions[count, x_pos])
			# print('3',r)
			# aaa
		count += 1
	# actions[agents_available_actions == 0] = 10.0
	return actions
# self.qmix_algo.qmixer(value_action.view(self.actors, self.n_agents, -1), torch.FloatTensor(state_batch).to(device)) 
def predict_classes(xs, img, net, action_values, n_agents, n_actor, state_batch,device,agents_available_actions):
	imgs_perturbed = perturb_actions(xs, img.clone())
	# print(imgs_perturbed)
	# print(action_values[:, :-1])
	action_values[agents_available_actions == 0] = 10000000.0
	action_values = action_values.repeat(len(xs), 1,1)
	
	value_perturbed = torch.gather(action_values, dim=2, index=imgs_perturbed).squeeze(0)#.view(-1,n_actor, n_agents, )

	input = Variable(value_perturbed, volatile=True)#.no_grad()#.cuda()
	# print(value_perturbed.size())
	# print(torch.FloatTensor(state_batch).repeat(len(xs), 1).size())
	predictions = net(input.cpu(), torch.FloatTensor(state_batch).repeat(len(xs), 1).cpu())
	predictions = predictions.data.cpu().numpy()
	return predictions #if minimize else 1 - predictions

def attack_success(x, img, target_calss, net, action_values, n_agents, n_actor, state_batch, verbose=False):

	attack_image = perturb_actions(x, img.clone())
	# print(len(x))
	# print(attack_image)
	# print(action_values.unsqueeze(0))
	# print(len(x)).repeat(len(x), 1,1)
	input = torch.gather(action_values.unsqueeze(0), dim=2, index=attack_image).squeeze(0).view(-1,n_actor, n_agents)
	input = Variable(input, volatile=True)#.cuda()

	# print(input)
	# aaa
	q_tot = net(input, torch.FloatTensor(state_batch)).data.cpu().numpy()[0][0][0]

	if (verbose):
		print ("q_tot: %.4f"%q_tot)
	if (q_tot < target_calss.data.cpu().numpy()[0][0][0]):
		return True # 如果比之前小很多 就算是成功了 阈值的设计？

# xs, img, net, action_values, n_agents, n_actor, state_batch,device
def attack_de(img, label, net, action_values,n_agent, n_action,n_actor, state_batch,device,agents_available_actions,target=None, pixels=1, maxiter=75, popsize=400, verbose=False):

	targeted_attack = target is not None
	target_calss = target if targeted_attack else label
	# print(agents_available_actions)
	bounds = [(0,n_agent), (0,n_action)] * pixels # len(bounds) = 5
	# print((0,n_agent))
	# aaa
	popmul = max(1, popsize//len(bounds))

	predict_fn = lambda xs: predict_classes(
		xs, img, net, action_values, n_agent, n_actor, state_batch,device,agents_available_actions) # 要最小化的目标函数
	callback_fn = lambda x, convergence: attack_success(
		x, img, target_calss, net, action_values, n_agent, n_actor,state_batch, verbose)

	inits = np.zeros([popmul*len(bounds), len(bounds)])
	for init in inits: # 随机初始化
		for i in range(pixels):
			init[i*2+0] = np.random.random()*n_agent
			#init[i*5+1] = np.random.random()*32
			init[i*2+1] = np.random.randint(0,n_action,1) 
			#init[i*5+3] = np.random.normal(128,127)# 均值和方差 初始化动作空间 几个智能体可能不一样
			#init[i*5+4] = np.random.normal(128,127)
	# print(init)
	attack_result = differential_evolution(predict_fn, bounds, maxiter=maxiter, popsize=popmul,
		recombination=1, atol=-1, callback=callback_fn, polish=False, init=inits)
	# print(attack_result.x.astype(int)) 

	attack_image = perturb_actions(attack_result.x.astype(int), img.clone())
	# print(attack_image)
	# aaa
	#print(attack_image.cpu().data.numpy().reshape(n_actor, n_agent, -1))
	return attack_image.cpu().data.numpy().reshape(n_actor, n_agent, -1), attack_result.x.astype(int)

