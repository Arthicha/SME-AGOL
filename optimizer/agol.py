# ------------------- import modules ---------------------

# standard modules
import time, sys, os
from copy import deepcopy
import configparser

# math-related modules
import numpy as np # cpu array
import torch # cpu & gpu array
from torch.distributions import Normal

# modular network
from optim import Optim

# ------------------- configuration variables ---------------------
EPS = 1e-6

# ------------------- class AddedGradientOnlineLearning ---------------------
class AddedGradientOnlineLearning(Optim):
	# -------------------- constructor -----------------------
	# (private)

	def setup(self,config):
		self.vnet = None

		# initialize replay buffer
		self.__sigma = float(config["HYPERPARAM"]["SIGMA"])
		self.__sigmas = self.zeros(self.W.shape[0],self.W.shape[1]) + self.__sigma
		self.__min_grad = float(config["HYPERPARAM"]["MINGRAD"])
		self.__lr = float(config["HYPERPARAM"]["LR"])

		# reset everything before use
		self.reset()

	def attach_valuenet(self,vnet):
		self.vnet = vnet

	def attach_rewardfunction(self,func):
		self.get_reward = func

	# ------------------------- update and learning ----------------------------
	# (public)

	
	def update(self,states,weights,feedbacks,grads):
		
			
		update = torch.abs(grads)*(weights-self.W)
		
		rewards = self.get_reward(feedbacks)
		advantage = (rewards-torch.mean(rewards,dim=0).unsqueeze(0))
		std_advantage =  advantage / (torch.std(rewards,dim=0).unsqueeze(0)+EPS)
		std_advantage[std_advantage < 0] *= 0.1
		std_advantage = torch.clamp(std_advantage,-3.0,3.0)
		
		update = torch.sum(update * std_advantage, dim=[0,1])/torch.pow(self.__sigmas,2)
		dw = torch.clamp(self.__lr*0.1*1e-4*update,-self.__min_grad,self.__min_grad)

		dsigma = std_advantage*torch.abs(grads)*(torch.pow(weights-self.W,2)-torch.pow(self.__sigmas,2))/torch.pow(self.__sigmas,3)
		dsigma = self.lr*1e-4*torch.sum(dsigma,dim=[0,1])

		with torch.no_grad():
			self.W += (dw).detach()
			self.__sigmas = torch.clamp(self.__sigmas + dsigma,0.001,0.05)

		print('w',self.W[0,0])
			
	# -------------------- apply noise -----------------------

	def wnoise(self):
		self.dist = Normal(loc=self.W.detach()*0,scale=self.__sigma)
		noise = self.dist.rsample()
		#self.Wn = self.W + noise 
		return noise




	
		




