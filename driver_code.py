'''
Driver code for the cart pole and MS PacMan environments.
'''

import argparse
import gymnasium as gym
from data_visualization import *
from agent import *
 

parser = argparse.ArgumentParser(description="This is an implementation of Reinfonrcment Learning agents (Q-Learning and DQN) for the Cart Pole and MS Pac Man environments at openAI Gymnasium.")
parser.add_argument("-c", "--cartpole", action="store_true", help = "Will run the cart pole environment with Q-Learning and then DQN agents and save the results (graphs and video) in current directory")
parser.add_argument("-p", "--pacman", action="store_true", help = "Will run the Ms PacMan environment with DQN agent and save the results (graphs and video) in the current directory ")
parser.add_argument("-e", "--episodes", type=int, help = "Define the duration of training by setting the number of episodes. Default is 3000 episodes.")
# Main program

args = parser.parse_args()
EPISODES=3000
ALPHA=0.1
GAMMA=1
EPSILON=0.7

if args.episodes:
    EPISODES=args.episodes

if args.cartpole:
    env = gym.make('CartPole-v1')
    agent = Q_Learning_Agent(
                         env=env,
                         numOfEpisodes=EPISODES, 
                         alpha=ALPHA, 
                         gamma=GAMMA, 
                         epsilon=EPSILON, 
                         discrete_obs=False, 
                         discrete_actions=True, 
                         lowerBounds=[-2.4, -2, -1, -3.5], 
                         upperBounds=[2.4, 2, 1, 3.5], 
                         actions=(0,1)
                         )

    
    history = agent.simulate_learning()
    rand_rewards = agent.simulate_random()
    plot_double(history['cumReward'], rand_rewards, 'Episodes', 'Cummulative Reward', "Q_Learning", "Random", 'CumReward_QLearnng', title='Rewards with learning and random strategies')
    plot_double_moving_avg(history['cumReward'], rand_rewards, 'Episodes', 'Cummulative Reward',"Q-Learning", "Random", 'Rolling_Avg', window_size=500, title='Moving Average (window = 500 episodes)')
    generate_basic_plot(history['epsilon'], 'Episodes', 'epsilon parameter', 'epsilon_qlearning', title="Epsilon (greedy) parameter")
    rewards,_ = agent.simulate_learned()
    print(f"Cummulative reward of test episode: {rewards}")
    
    agent = DQN_Agent(env,
                  gamma=GAMMA,
                  epsilon=EPSILON,
                  numOfEpisodes=EPISODES,
                  stateDimension=4,
                  actionDimension=2,
                  atari=False)
    history = agent.simulate_learning()
    generate_basic_plot(history['epsilon'], 'Episodes', 'epsilon parameter', 'epsilon_dqm', title="Epsilon (greedy) parameter")
    generate_basic_plot(history['cumReward'], 'Episodes', 'cummulative reward', 'Cummulative_reward_dqn', title="Cummulative rewards per episode")
    plot_moving_avg(history['cumReward'], 'Episodes', 'cummulative reward', 'Cummulative_reward_avg_dqn', title="Moving average cummulative rewards per episode", window_size=100)
    
else:
    env = gym.make('ALE/MsPacman-v5')
    agent = DQN_Agent(env,
                  gamma=GAMMA,
                  epsilon=EPSILON,
                  numOfEpisodes=EPISODES,
                  stateDimension=(1, 210, 160, 3),
                  actionDimension=9,
                  atari=True)
    history = agent.simulate_learning()
    generate_basic_plot(history['epsilon'], 'Episodes', 'epsilon parameter', 'epsilon_dqm_pacman', title="Epsilon (greedy) parameter")
    generate_basic_plot(history['cumReward'], 'Episodes', 'cummulative reward', 'Cummulative_dqn_pacman', title="Cummulative rewards per episode")
    plot_moving_avg(history['cumReward'], 'Episodes', 'cummulative reward', 'Cummulative_reward_avg_dqn_pacman', title="Moving average cummulative rewards per episode", window_size=100)
    rewards,_ = agent.simulate_learned()
    print(f"Cummulative reward of test episode: {rewards}")



