from matplotlib import pyplot as plt
from agent import *
from env import *


if __name__ == '__main__':
	environment = Env()
	AI = AgentQlearning(environment, 0.9, 0.1, 0.1)
	AI.learning(500)
	AI2 = AgentSarsa(environment, 0.9, 0.1, 0.1)
	AI2.learning(500)
	
	plt.xlabel('Episodes')
	plt.ylabel('Average rewards of the recent 10 episodes')
	plt.legend()
	plt.savefig('./cliff_walking.png')
	plt.show()