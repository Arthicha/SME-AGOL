'''
Class: GD
created by: arthicha srisuchinnawong
e-mail: arsri21@student.sdu.dk
date: 2 August 2022

Gradient Descent
'''


# ------------------- import modules ---------------------

# standard modules
import time, sys, os
from copy import deepcopy
import colorama 
from colorama import Fore
import configparser

# math-related modules
import numpy as np # cpu array
import torch # cpu & gpu array
from torch.autograd import Variable
from torch.distributions import Normal, Categorical

# modular network
from optim import Optim

# experience replay
from utils.utils import TorchReplay as Replay

#plot
import matplotlib.pyplot as plt

# ------------------- configuration variables ---------------------
EPSILON = 1e-6
# ------------------- class GD ---------------------

class GD(Optim):

	
	# -------------------- constructor -----------------------
	# (private)

	def setup(self,config):

		self.Wn = self.W

		self.__replaylength = int(config["REPLAYPARAM"]["NREPLAY"])
		self.__triallength = int(config["REPLAYPARAM"]["NTIMESTEP"])
		self.__sigma = float(config["HYPERPARAM"]["SIGMA"])
		self.__min_grad = float(config["HYPERPARAM"]["MINGRAD"])

		self.__reward_replay = Replay(self.__replaylength,shape=(self.__triallength,6,1))
		self.__observation_replay = Replay(self.__replaylength,shape=(self.__triallength,1,1))

		self.noise_mean = self.zeros(self.W.shape[0],self.W.shape[1])

		# reset everything before use
		self.reset()

	def update_experience_replay(self,reward, observation):
		# update experience replay
		self.__reward_replay.add(reward)
		self.__observation_replay.add(observation)

		# reset gradient
		self.reset() 

	def attach_valuenet(self,vnet):
		self.vnet = vnet

	
	# ------------------------- update and learning ----------------------------
	# (public)

	def get_noise(self):
		self.dist = Normal(loc=self.noise_mean,scale=0.0)
		noise = self.dist.rsample()
		self.Wn = self.W + noise
		return noise

	
	def update(self):
		if not self.freeze:
			obs = self.__observation_replay.get_data()
			obs = torch.reshape(obs,(obs.shape[0],obs.shape[1],1,1))
			
			

			for i in range(10):
				self.vnet.zero_grad()

				predicted_value = self.vnet(obs)
				predicted_value = torch.transpose(predicted_value, -1, -2)

				loss = torch.sum(torch.pow(self.__reward_replay.get_data()-predicted_value,2))
				loss.backward()
				grad = self.W.grad.clone()
				with torch.no_grad():
					self.W += -(self.lr*grad).detach()

			print("\tloss:",loss.item())

		self.reset()


	
		




