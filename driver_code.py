'''
Driver code for the cart pole and MS PacMan environments.
'''

import gymnasium as gym
from data_visualization import *
from agent import *
    
# Main program
    
# Define parameters
numOfEpisodes=3000
alpha=0.1
gamma=1
epsilon=0.7

#env = gym.make('CartPole-v1')
env = gym.make('ALE/MsPacman-v5')

'''
# Q-Learning agent
agent = Q_Learning_Agent(
                         env=env,
                         numOfEpisodes=numOfEpisodes, 
                         alpha=alpha, 
                         gamma=gamma, 
                         epsilon=epsilon, 
                         discrete_obs=False, 
                         discrete_actions=True, 
                         lowerBounds=[-2.4, -2, -1, -3.5], 
                         upperBounds=[2.4, 2, 1, 3.5], 
                         actions=(0,1)
                         )
'''

# DQN agent
agent = DQN_Agent(env,
                  gamma=gamma,
                  epsilon=epsilon,
                  numOfEpisodes=numOfEpisodes,
                  stateDimension=(1, 210, 160, 3),
                  actionDimension=9,
                  atari=True)

history = agent.simulate_learning()
#rand_rewards = agent.simulate_random()
generate_basic_plot(history['epsilon'], 'Episodes', 'epsilon parameter', 'epsilon_q_lerning', title="Epsilon (greedy) parameter")
generate_basic_plot(history['cumReward'], 'Episodes', 'cummulative reward', 'Cummulative_reward', title="Cummulative rewards per episode")
plot_moving_avg(history['cumReward'], 'Episodes', 'cummulative reward', 'Cummulative_reward_avg', title="Moving average cummulative rewards per episode", window_size=100)



#plot_double(history['cumReward'], rand_rewards, 'Episodes', 'Cummulative Reward', "Q_Learning", "Random", 'CumReward_QLearnng', title='Rewards with learning and random strategies')
#plot_double_moving_avg(history['cumReward'], rand_rewards, 'Episodes', 'Cummulative Reward',"Q-Learning", "Random", 'Rolling_Avg', window_size=500, title='Moving Average (window = 500 episodes)')
rewards,_ = agent.simulate_learned()
print(f"Cummulative reward of test episode: {rewards}")


