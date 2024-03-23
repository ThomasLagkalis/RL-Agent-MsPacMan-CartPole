# This is the testing suite. Build to validate the functionality of the project.

from agent import *
import unittest
import random
import gymnasium as gym

# Global variables and initializations used by various tests.
random.seed(42)
testCases=1000
lowerBounds=[-2.4, -2, -1, -3.5]
upperBounds=[2.4, 2, 1, 3,5]
actions = (0,1)
env = gym.make("CartPole-v1")
agent = Q_Learning_Agent(env, 1000, 1, 0.25, 0.99, False, True, lowerBounds, upperBounds, actions)
obs, info = env.reset()

class TestQLearningAgent(unittest.TestCase):

    def test_get_state_(self):
        # Test that get_state function returns int correctly.
        states = []
        for _ in range(testCases):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
        
            if terminated or truncated:
                observation, info = env.reset()

            state = agent.get_state(obs) 
            self.assertEqual(type(state), type(1))

    
