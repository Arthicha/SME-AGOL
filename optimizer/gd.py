# ------------------- import modules ---------------------

# standard modules
import time, sys, os
from copy import deepcopy
import configparser

# math-related modules
import numpy as np # cpu array
import torch # cpu & gpu array
from torch.autograd import Variable

# modular network
from optim import Optim

# experience replay
from utils.utils import TorchReplay as Replay

import matplotlib.pyplot as plt


# ------------------- configuration variables ---------------------
EPSILON = 1e-6
PLOT = False # plot actual return vs predicted value
# ------------------- class GD ---------------------

class GradientDescent(Optim):

	
	# -------------------- constructor -----------------------
	# (private)

	def setup(self,config):

		self.__lr = float(config["CRITICOPTIM"]["LR"])
		self.__iteration = int(config["CRITICOPTIM"]["ITERATION"])

		# reset everything before use
		self.reset()

	def attach_valuenet(self,vnet):
		self.vnet = vnet

	def attach_returnfunction(self,func):
		self.compute_return = func

	
	# ------------------------- update and learning ----------------------------
	# (public)

	
	def update(self,states,rewards):
		#values = values*0+0.5
		for i in range(self.__iteration):

			self.vnet.zero_grad()

			predicted_value = self.compute_return(self.vnet(states))
			values = self.compute_return(rewards)

			loss = torch.mean(torch.pow(values-predicted_value,2))
			loss.backward()


			with torch.no_grad():
				self.W -= (self.__lr*self.W.grad).detach()
			self.vnet.apply_noise(0)

		print("\tvalue loss:",loss.item())
		
		if PLOT:
			predicted_value = self.compute_return(self.vnet(states))
			plt.clf()
			plt.plot(np.transpose(self.numpy(values[:,:,0,0])),c='tab:blue')
			plt.plot(np.transpose(self.numpy(predicted_value[:,:,0,0])),c='tab:orange')
			plt.plot(np.transpose(self.numpy(torch.mean(values[:,:,0,0],dim=0))),c='tab:red')
			plt.savefig('value.jpg')
		self.reset()

	
		




