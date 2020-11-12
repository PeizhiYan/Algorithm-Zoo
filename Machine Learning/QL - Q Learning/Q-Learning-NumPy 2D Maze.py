import numpy as np
import pandas as pd
import time
import os
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation


day = 0
moon = ['🌑','🌒','🌓','🌔','🌕','🌖','🌗','🌘']
def cls():
	global day
	os.system('clear')
	print(moon[int(day/10)%len(moon)]+' day:'+ str(int(day/10))) # display the moon
	day+=1

# initialize simulation parameters
MAZE_W = 8							   # width of the maze
MAZE_H = 8							   # height of the maze
ACTIONS = ['left', 'right', 'up', 'down'] # available actions
EPSILON = 0.95							# e-greedy policy
ALPHA = 0.1							   # learning rate
GAMMA = 0.8							   # discount factor
MAX_EPISODES = 100					   # maximum episodes
FRESH_TIME = 0.00001						 # fresh time for each move
AVOCADO = ( random.randint(1,MAZE_H-1), 
			random.randint(1,MAZE_W-1) )  # the location of avocado
WOLF = ( random.randint(1,MAZE_H-1), 
		 random.randint(1,MAZE_W-1) )	 # the location of the wolf

# genarate the maze
MAZE = np.zeros([MAZE_H, MAZE_W])
MAZE[-1][-1] = 1 # the position of the avocado

def display_maze(x, y):
	buffer = [['🌱' for i in range(MAZE_W+2)] for j in range(MAZE_H+2)]
	for i in range(len(buffer[0])):
		buffer[0][i] = '🌵'
		buffer[-1][i] = '🌵'
	for i in range(len(buffer)):
		buffer[i][0] = '🌵'
		buffer[i][-1] = '🌵'
	buffer[AVOCADO[0]][AVOCADO[1]] = '🥑'
	buffer[WOLF[0]][WOLF[1]] = '🐺'
	if x == AVOCADO[0] and y == AVOCADO[1]:
		buffer[x+1][y+1] = '💗'
	else:
		buffer[x+1][y+1] = '🐰'
	cls()
	for i in range(len(buffer)):
		for j in range(len(buffer[0])):
			print(buffer[i][j], end='')
		print()

display_maze(0,0)


def init_Q_table(maze, actions):
	Q_table = pd.DataFrame(
		np.zeros([MAZE_H*MAZE_W,len(actions)]), # Q-table initial values
		columns = actions,
	)
	return Q_table

def get_state(x, y):
	return x*MAZE_W + y

def get_position(S):
	return int(S/MAZE_W), S%MAZE_W

def choose_action(S, Q_table):
	state_actions = Q_table.iloc[S, :]
	if(np.random.uniform() > EPSILON or state_actions.all() == 0):
		action_name = np.random.choice(ACTIONS)
	else: # act greedy
		action_name = state_actions.argmax()
	return action_name


def take_action(S, A):
	x, y = get_position(S)
	def at(x, y):
		if x < 0 or x >= MAZE_H or y < 0 or y >= MAZE_W:
			return -2 # invalid position (out of boundary)
		elif x == AVOCADO[0] and y == AVOCADO[1]:
			return 1 # reach the acovado
		elif x == WOLF[0] and y == WOLF[1]:
			return -1 # eat by wolf, got -1 reward (punishment)
		else:
			return 0 # empty position

	def try_move(S, x, y):
		if at(x,y) == -2: # hit the wall
			S_ = S
			R = 0
		elif at(x,y) == -1: # eat by wolf
			S_ = 'terminal'
			R = -1
		elif at(x,y) == 1:
			S_ = 'terminal'
			R = 1
		else:
			S_ = get_state(x, y)
			R = 0
		return S_, R

	"""interact with the environment"""
	if A == 'left': # move left
		y -= 1
		return try_move(S, x, y)
	elif A == 'right':
		y += 1
		return try_move(S, x, y)
	elif A == 'up':
		x -= 1
		return try_move(S, x, y)
	else:
		x += 1
		return try_move(S, x, y)

def log_Q_table(Q_table, hist_):
	matrix = pd.DataFrame.copy(Q_table)
	hist_.append(matrix)
	return hist_


def plot_history(hist):
	for i in range(len(hist)):
		if (i % 10 == 0):
			print(i)
			matrix = hist[i]
			matrix = np.array(matrix.as_matrix())
			L = np.reshape(matrix[:,0], [MAZE_H, MAZE_W])
			R = np.reshape(matrix[:,1], [MAZE_H, MAZE_W])
			U = np.reshape(matrix[:,2], [MAZE_H, MAZE_W])
			D = np.reshape(matrix[:,3], [MAZE_H, MAZE_W])
			plt.cla()
			plt.subplot(2,2,1)
			plt.imshow(L, cmap='hot', interpolation='nearest')
			plt.subplot(2,2,2)
			plt.imshow(R, cmap='hot', interpolation='nearest')
			plt.subplot(2,2,3)
			plt.imshow(U, cmap='hot', interpolation='nearest')
			plt.subplot(2,2,4)
			plt.imshow(D, cmap='hot', interpolation='nearest')
			plt.savefig("./2D_results/iter_"+str(i+1))
		


def plot(Q_table, hist):
	matrix = np.array(Q_table.as_matrix())
	L = np.reshape(matrix[:,0], [MAZE_H, MAZE_W])
	R = np.reshape(matrix[:,1], [MAZE_H, MAZE_W])
	U = np.reshape(matrix[:,2], [MAZE_H, MAZE_W])
	D = np.reshape(matrix[:,3], [MAZE_H, MAZE_W])
	plt.subplot(3,2,1)
	plt.imshow(L, cmap='hot', interpolation='nearest')
	plt.subplot(3,2,2)
	plt.imshow(R, cmap='hot', interpolation='nearest')
	plt.subplot(3,2,3)
	plt.imshow(U, cmap='hot', interpolation='nearest')
	plt.subplot(3,2,4)
	plt.imshow(D, cmap='hot', interpolation='nearest')
	plt.subplot(3,2,5)
	#X = [i for i in range(len(hist))]
	#Y = hist
	#plt.plot(X, Y)
	plt.show()
	

avocado = 0
wolf = 0
hist = []
def rl():
	global avocado, wolf, hist
	Q_table = init_Q_table(MAZE, ACTIONS)
	for episode in range(MAX_EPISODES):
		try:
			S = random.randint(0,MAZE_H*MAZE_W)
			is_terminated = False
			x, y = get_position(S)
			display_maze(x, y)
			while not is_terminated:
				A = choose_action(S, Q_table)
				next_S, R = take_action(S, A)
				pred_Q = Q_table.loc[S,A]
				if next_S != 'terminal':
					targ_Q = R + GAMMA*Q_table.iloc[next_S, :].max()
				else: # next state is terminal
					targ_Q = R 
					is_terminated = True
				Q_table.loc[S,A] += ALPHA*(targ_Q - pred_Q) # update Q table
				S = next_S
				try:
					x, y = get_position(S)
				except:
					if R == -1:
						x = WOLF[0]
						y = WOLF[1]
						wolf += 1
					elif R == 1:
						x = AVOCADO[0]
						y = AVOCADO[1]
						avocado += 1
				#display_maze(x, y)
				print('Episode: ', episode)
				print('🥑: ', avocado)
				print('🐺: ', wolf)
				#hist.append(avocado)
			hist = log_Q_table(Q_table, hist)

		except:
			print('oops, something wrong')
			continue
	return Q_table


learned_Q_table = rl()

plot_history(hist)


cls()
print(learned_Q_table)
plot(learned_Q_table, hist)

