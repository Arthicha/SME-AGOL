
# ------------------- import modules ---------------------

# standard modules
import time, sys, os
from copy import deepcopy
import configparser

# math-related modules
import numpy as np # cpu array
import torch # cpu & gpu array

# simulation
from interface.vrepinterfaze import VrepInterfaze

# control
from network.SME import SequentialMotionExecutor
from optimizer.agol import AddedGradientOnlineLearning
from optimizer.gd import GradientDescent
from utils.utils import TorchReplay as Replay

# ------------------- config variables ---------------------
NREPLAY = 8
NTIMESTEP = 30
NEPISODE = 10000
RESET = False
CONNECTION = torch.FloatTensor(np.array([[0,1,0,0],[0,0,1,0],[0,0,0,1],[1,0,0,0]])).cuda()

# ------------------- auxiliary functions ---------------------

cumsumgain = torch.arange(NTIMESTEP,0,-1).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).cuda()

def numpy(x):
	return x.cpu().detach().numpy()

def tensor(x):
	return torch.FloatTensor(x).to(torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

def compute_return(speed):
	global cumsumgain
	speed_flip = torch.flip(speed,dims=[1])
	speed_fliptilend = torch.cumsum(speed_flip,dim=1)
	speed_tilend = torch.flip(speed_fliptilend,dims=[1])
	return speed_tilend/cumsumgain

# ------------------- setup ---------------------

# initiliaze SME network
sme = SequentialMotionExecutor('network.ini',CONNECTION)

# initialize AGOL learning algorithm
agol = AddedGradientOnlineLearning(sme.mn.W,'optimizer.ini')
agol.attach_returnfunction(compute_return) # set return function
agol.attach_valuenet(sme.vn) # set value network (remove this if you want to use average baseline)

# initialzie GD learning algorithm for baseline estimation
gd = GradientDescent(sme.vn.W,'optimizer.ini')
gd.attach_returnfunction(compute_return)  # set return function
gd.attach_valuenet(sme.vn)

# initialize simulation interface
vrep = VrepInterfaze()

# initialize experience replay
reward_replay = Replay(NREPLAY,shape=(NTIMESTEP,1,1))
grad_replay = Replay(NREPLAY,shape= (NTIMESTEP,4,18))
weight_replay = Replay(NREPLAY,shape=(1,4,18))
observation_replay = Replay(NREPLAY,shape=(NTIMESTEP,1,CONNECTION.shape[0]))

# ------------------- start locomotion learning ---------------------
for i in range(NEPISODE):
	print('episode',i)

	if RESET: # reset the simulation/network
		vrep.reset()
		sme.reset()

	# episode-wise setup
	prepose = vrep.get_robot_pose()
	sme.explore(agol.wnoise())
	weight_replay.add(sme.mn.Wn)

	for t in range(NTIMESTEP):

		# update network
		output = sme.forward()
		basis = sme.get_basis(torch=True)

		# update environment
		vrep.set_robot_joint(numpy(output))
		vrep.update()

		# compute reward
		pose = vrep.get_robot_pose()
		dx = pose[0]-prepose[0]
		dy = pose[1]-prepose[1]
		reward = dx*np.cos(prepose[-1]) + dy*np.sin(prepose[-1]) 
		prepose = deepcopy(pose)

		# backpropagate output gradient
		sme.zero_grad()
		torch.sum(output).backward() 

		# append experience replay
		reward_replay.add(tensor([reward]).unsqueeze(0))
		observation_replay.add(basis)
		grad_replay.add(sme.mn.W.grad)

	print('\t episodic reward',torch.sum(reward_replay.data()[-1]).item())

	# update the network
	gd.update(observation_replay.data(),reward_replay.data())
	agol.update(observation_replay.data(),weight_replay.data(),reward_replay.data(),grad_replay.data())
	

	
	


