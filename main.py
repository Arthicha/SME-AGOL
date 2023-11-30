
# ------------------- import modules ---------------------

# standard modules
import time, sys, os
from copy import deepcopy
import configparser

# math-related modules
import numpy as np # cpu array
import torch # cpu & gpu array
import matplotlib.pyplot as plt

# simulation
from interface.vrepinterfaze import VrepInterfaze

# control
from network.SME import SequentialMotionExecutor
from optimizer.agol import AddedGradientOnlineLearning
from utils.utils import TorchReplay as Replay


# control variable
NREPLAY = 8
NTIMESTEP = 30
HORIZON = 15
RESET = True

def numpy(x):
	return x.cpu().detach().numpy()

def tensor(x):
	return torch.FloatTensor(x).to(torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

def get_reward(speed):
	global HORIZON
	speed_flip = torch.flip(speed,dims=[1])
	speed_fliptilend = torch.cumsum(speed_flip,dim=1)
	speed_tilend = torch.flip(speed_fliptilend,dims=[1])
	print(speed_tilend[-1,0].item())
	
	return speed_tilend

connection = torch.FloatTensor(np.array([[0,1,0,0],[0,0,1,0],[0,0,0,1],[1,0,0,0]])).cuda()
sme = SequentialMotionExecutor('network.ini',connection)
agol = AddedGradientOnlineLearning(sme.mn.W,'optimizer.ini')
agol.attach_rewardfunction(get_reward)

vrep = VrepInterfaze()

reward_replay = Replay(NREPLAY,shape=(NTIMESTEP,1,1))
grad_replay = Replay(NREPLAY,shape= (NTIMESTEP,4,18))
weight_replay = Replay(NREPLAY,shape=(1,4,18))
observation_replay = Replay(NREPLAY,shape=(NTIMESTEP,1,connection.shape[0]))

for i in range(10000):
	if RESET:
		vrep.reset()
		sme.reset()
	prepose = vrep.get_robot_pose()

	print('episode',i)

	sme.explore(agol.wnoise())
	weight_replay.add(sme.mn.Wn)

	for t in range(NTIMESTEP):

		# update network
		output = sme.forward()
		basis = sme.get_basis(torch=True)

		# update environment
		vrep.set_robot_joint(numpy(output))
		vrep.update()

		# evaluate and update
		pose = vrep.get_robot_pose()
		dx = pose[0]-prepose[0]
		dy = pose[1]-prepose[1]
		reward = dx - np.abs(dy)
		propose = deepcopy(pose)

		torch.sum(output).backward()
		reward_replay.add(tensor([reward]).unsqueeze(0))
		observation_replay.add(basis)
		grad_replay.add(sme.mn.W.grad)

	agol.update(observation_replay.data(),
		weight_replay.data(),
		reward_replay.data(),
		grad_replay.data())
	
	


