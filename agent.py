import pandas as pd 
import numpy as np
import random
import time
import gymnasium as gym
from collections import deque
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow import gather_nd
from tensorflow.keras.losses import mean_squared_error


class Q_Learning_Agent:
    '''
    This class is an an abstract illustration of the Q-Learning algorithm containing all usefull funtionts to run 
    the Q-Learning algorithm in an openAI's gym environment.
    Note: works only for 1-dimensional observaiton spaces amd when all number of bins are < 10   

    Class contructor with parameters:
    - env: The environment (openAI gym's) on which the algorith will run.
    - numOfEpisodes: The total number of episodes.
    - gamma: Discount factor.
    - alpha: Step rate.
    - epsilon: Greedy parameter (for the e-Greedy algorithm).
    - discrete_obs: boolean, True if observation space is descrere
    - discrete_actions: boolean, True if actions space is descrere (Currently not used)
    - lowerBounds: lower limits for discretization.
    - upperBounds: upper limits for discretization.
    - actions: The valid actions in each time step for the agent (tuple).
    '''

    def __init__(self, env, numOfEpisodes, gamma, alpha, lowerBounds, upperBounds
                 , actions, epsilon=1.0, discrete_obs=False, discrete_actions=True):
        self.env=env
        self.numOfEpisodes=numOfEpisodes
        self.gamma=gamma
        self.alpha=alpha
        self.epsilon=epsilon
        self.Qtable=dict()
        self.discrete_obs=discrete_obs
        self.discrete_actions=discrete_actions
        self.lowerBounds=lowerBounds
        self.upperBounds=upperBounds
        self.actions=actions
   

    def get_state(self, obs):
        # This function gets the observation from the environment and returns the state (as an int)
        # ready to pass it in Q table.
        #
        # Agrs:
        # - obs: observation list from environment with format specified by the environment.
        # Return:
        # - state: the state mapped to an integer
        # Note: works only for 1-dimensional observaiton spaces and when all number of bins are < 10)
       
        size = self.env.observation_space.shape[0];
        bins = []
        state = []
        for i in range(size):
            bins.append(pd.cut([self.lowerBounds[i],self.upperBounds[i]], bins=10, retbins=True)[1][1:-1])
            state.append(np.digitize(x=obs[i], bins=bins[i]))
        
        i = ''.join(map(str, state))
        return int(i) 
        
    def check_Qtable(self, state):
        # Checks if state is in the Q table and if not initializes the stete,
        # Args: -state: the state to initialize in the Q table.

        if state not in self.Qtable.keys():
            self.Qtable[state] = dict()
            for action in self.actions:
                self.Qtable[state][action] = 0.0
        return 
    
    # Adapted for the Cart Pole env.
    def max_Q(self, state):
        # Returns the maximum Q-Value among the actions in state 
        # Args: -state: the state from which is selecting action.
        if self.Qtable[state][0] >= self.Qtable[state][1]:
            return self.Qtable[state][0]
        else:
            return self.Qtable[state][1]

    def max_Q_index(self, state):
        # Returns the index (aka action) with the highest Q-Value from the Q Table.
        # Args: -state: the state from which is selecting action.
        if self.Qtable[state][0] >= self.Qtable[state][1]:
            return 0
        else:
            return 1

    def get_action(self, state, curEpisode):
        # Chooses an action based on epsilon greedy approach (GLIE: Greedy in the Limit of Infinite Exploration)
        # Args: 
        # - state: currnet state.
        # - curEpisode: current episode.

        # For the first 100 episodes explore.
        if curEpisode<500: 
            action = self.env.action_space.sample() 
            return action
        
        # Generate a random float number in [0,1] 
        randomNumber = random.random()
        
        # Afte 7000 episodes, slowly decreasing epsilon factor.
        if curEpisode>7000:
            self.epsilon = self.epsilon * 0.9999
        
        # If the random number is less than epsilon, explore.
        if randomNumber<self.epsilon:
            action = self.env.action_space.sample() 
            return action
        
        # Else select greedy
        # Find max Q value
        max_Q = self.max_Q(state)
        actions = []
        for key, value in self.Qtable[state].items():
            if value == max_Q:
                actions.append(key)
        if len(actions) != 0:
            action = random.choice(actions)
        return action

    def simulate_learning(self):
        # Run all episodes amd update the Q-Values on Q table
        # return: - history: a dictionary with 'epsilon' epsilon value in each episode
        # and 'cumReward' the cummulative reward in each episode 
        # These data are provided for the plots.
        history = {'epsilon': [],  'cumReward': []}
        for curEpisode in range(self.numOfEpisodes):
            rewards = []
            
            # Before starting the episode reset the environment 
            (obs, _) = self.env.reset()

            print("Simulating learning episode {}".format(curEpisode))
            history['epsilon'].append(self.epsilon)

            terminate=False
            firstStep=True
            while not terminate:
                # In this loop first choose action and then update the Q table (aka learn)
                state = self.get_state(list(obs))
                self.check_Qtable(state)
                action = self.get_action(state, curEpisode)

                obs, reward, terminate, _, _ = self.env.step(action)
                rewards.append(reward)

                # If reached the terminal state then:
                # Q(s,a) <-- Q(s,a) + alpha*(reward - Q(s,a))
                # Else:
                # Q(s,a) <-- Q(s,a) + alpha*(reward + gamma*next_Qmax - Q(s,a))
                 
                next_Qmax = self.max_Q(state)
                if not terminate and not firstStep:
                    self.Qtable[prev_state][prev_action] = self.Qtable[prev_state][prev_action] + self.alpha*(prev_reward + self.gamma*next_Qmax - self.Qtable[prev_state][prev_action])
                elif not firstStep:
                    self.Qtable[prev_state][prev_action] = self.Qtable[prev_state][prev_action] + self.alpha*(prev_reward - self.Qtable[prev_state][prev_action])
                prev_state = state
                prev_action = action
                prev_reward = reward
                firstStep=False
            history['cumReward'].append(sum(rewards))
        return history
        
 
    def simulate_learned(self):
        test_env = gym.make('CartPole-v1', render_mode='rgb_array')
        test_env1 = gym.wrappers.RecordVideo(env=test_env, video_folder='./visual' , name_prefix='cart_pole')
        (obs, _) = test_env.reset()
        test_env1.render()
        state = self.get_state(list(obs))
        test_env1.start_video_recorder()
        timeSteps = 1000
        rewards = []

        for t in range(timeSteps):
            # Select only greedy actions
            self.check_Qtable(state)
            action = self.max_Q_index(state)
            state, reward, terminated, truncated, _ = test_env1.step(action)
            test_env1.render()
            state = self.get_state(list(state))
            rewards.append(reward)
            time.sleep(0.05)
            if (terminated):
                time.sleep(1)
                break
        test_env1.close_video_recorder()
        test_env1.close()
        test_env.close()
        return sum(rewards), test_env

    def simulate_random(self):
        '''
        A simulation of random strategy for comparisons
        '''
        cumReward = []

        for curEpisode in range(self.numOfEpisodes):
            rewards = []
            
            # Before starting the episode reset the environment 
            (obs, _) = self.env.reset()

            print("Simulating random  episode {}".format(curEpisode))

            terminate=False
            while not terminate:
                # In this loop first choose action randomly and then make step.
                action = self.env.action_space.sample() 
                obs, reward, terminate, _, _ = self.env.step(action)
                rewards.append(reward)
            cumReward.append(sum(rewards))
        return cumReward
        


class DQN_Agent:
    

    def __init__(self, env, gamma, epsilon, numOfEpisodes, stateDimension, actionDimension, atari=False):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilonMin = 0.15
        self.numOfEpisodes = numOfEpisodes
        self.stateDimension = stateDimension
        self.actionDimension = actionDimension
        self.atari=atari
        # Define the replay and batch buffers sizes.
        self.replayBuffSize = 500
        self.batchSize = 32
        # The period of episodes to update the target network
        self.targetNetPeriod = 10
        self.replayBuff=deque(maxlen=self.replayBuffSize)
        # Create the two networks. Both networks have the same structure.
        self.onlineNet=self.build_network()
        self.targetNet=self.build_network()
        # Copy the initial weights to targetNetwork
        self.targetNet.set_weights(self.onlineNet.get_weights())
        # Action used in training.
        self.usedActions = []


    def build_network(self):
        '''
        This function creates the neural network.Note that for atari games we need
        three extra convolutional layers to extract the useful features.
        Returns the model (neural network). 
        '''
        if self.atari:
            model = Sequential()
            model.add(Conv2D(32, (8, 8), strides=4, padding='same'))
            model.add(Activation('relu'))
            model.add(Conv2D(64, (4, 4), strides=2, padding='same'))
            model.add(Activation('relu'))
            model.add(Conv2D(64, (3, 3), strides=1, padding='same'))
            model.add(Activation('relu'))
            model.add(Flatten())
            mmodel.add(Dense(32,input_dim=self.stateDimension,activation='relu'))
            model.add(Dense(32,activation='relu'))
            model.add(Dense(self.actionDimension,activation='linear'))
            model.compile(loss='mse', optimizer=Adam())
            return model

        model = Sequential()
        model.add(Dense(32,input_dim=self.stateDimension,activation='relu'))
        model.add(Dense(32,activation='relu'))
        model.add(Dense(self.actionDimension,activation='linear'))
        # compile the network with the custom loss defined in my_loss_fn
        model.compile(optimizer = Adam(), loss = "mse", metrics = ['accuracy'])
        return model


    def loss_fn(self, y_true, y_pred):
        '''
        Funtion which implements the loss funtction of the DQN neural network (NN).
        Computes the mean squared error for one action (of the N actions), e.g. for cart pole N=2.
        Args:
            - y_true: a matrix of of dimension (self.batchReplayBufferSize, self.actionDimension) which corresponds to the target value.
            - y_pred: a matrix of of dimension (self.batchReplayBufferSize, self.actionDimension) which corresponds to the prediction value.
        '''
        s1,s2=y_true.shape
        indices=np.zeros(shape=(self.batchSize,self.actionDimension))
        indices[:,0]=np.arange(self.batchSize)
        indices[:,1]=self.usedActions
        loss = mean_squared_error(gather_nd(y_true,indices=indices.astype(int)), gather_nd(y_pred,indices=indices.astype(int)))
        return loss


    def train_net(self, curEpisode):
        '''
        This function implementing the training of the NNs.
        Args: curEpisode: current index of episodes, used to update
                         the target NN.
        '''

        # If the replay buffer is full then train the model.
        if len(self.replayBuff) > self.batchSize:
            # sample randomly from replay buffer into training batch buffer.
            miniBatch = random.sample(self.replayBuff, self.batchSize)
            
            for state, action, reward, next_state, done in miniBatch:
                if not done:
                    target_Q = reward + self.gamma * np.amax(self.targetNet.predict(next_state)[0], verbose=0)
                else:
                    target_Q = reward

                Q_values = self.onlineNet.predict(state, verbose=0)
                Q_values[0][action] = target_Q
                self.onlineNet.fit(state, Q_values, epochs=1, verbose=0)
             
            # Check if it's time to update the target NN parametres.
            if (curEpisode % self.targetNetPeriod == 0):
                self.targetNet.set_weights(self.onlineNet.get_weights())
                print("Target network updated!")
            return 
        else:
            return

    
    def get_action(self, state, curEpisode):
        ''' 
        This funciton chooses an action for the state on curEpisode 
        based on the epsilon-greedy approach.

        Args: 
             - state: state for which to compute the aciton.
             - curEpisode: the current episode index.
        '''

        if curEpisode<500:
            return np.random.choice(self.actionDimension)   
        
        randomNumber=np.random.random()

        if curEpisode>700 and self.epsilon > self.epsilonMin:
            self.epsilon=0.999999*self.epsilon

        if randomNumber < self.epsilon:
            # returns a random action selected from: 0,1,...,actionNumber-1
            return np.random.choice(self.actionDimension)

        else: 
            Q_values = self.onlineNet.predict(state, verbose=0)
            return np.random.choice(np.where(Q_values[0,:]==np.max(Q_values[0,:]))[0])

    def simulate_learning(self):
        '''
        Function to simulate the training (learning) process.
            
        return: A dictionary containing the history of the training 
        i.e. sum of rewards in each episdo and epsilon parameter per episode. 
        Both of these histories are lists.
        '''
        file = open('data.txt', 'w')

        history = {'epsilon': [],  'cumReward': []}
        for curEpisode in range(self.numOfEpisodes):
            rewards = []                

            print("Simulating learning episode {}".format(curEpisode))
            file.write("Training ep: {}:  ".format(curEpisode))

            # Reset the environment.
            (curState, _) = self.env.reset()
            curState = np.expand_dims(curState, axis=0)
           # curState = np.reshape(curState,[1,self.stateDimension])
            terminalState = False
            
            # 2000 time steps maximum.
            for t in range(3500):
                    
                # Select action.
                action = self.get_action(curState, curEpisode)

                # Step in the environment
                (nextState, reward, terminalState,_,_) = self.env.step(action) 
                nextState = np.expand_dims(nextState, axis=0)
                #nextState = np.reshape(nextState,[1,self.stateDimension])
                rewards.append(reward)
                    
                # Add (s,a,r,s', terminalState) to the replay buffer (aka memory)
                if not t==0:
                    self.replayBuff.append((curState, action, reward, nextState, terminalState))
                curState = nextState
                if terminalState:
                    break
             
            self.train_net(curEpisode)
            history['epsilon'].append(self.epsilon)
            history['cumReward'].append(sum(rewards))
            print(f"Rewards: {sum(rewards)}, Epsilon: {self.epsilon}")
            file.write("Sum rewards: {}\n".format(sum(rewards)))
        return history

    def simulate_learned(self):
        test_env = gym.make('ALE/MsPacman-v5', render_mode='rgb_array')
        test_env1 = gym.wrappers.RecordVideo(env=test_env, video_folder='./visual' , name_prefix='PacMan')
        (state, _) = test_env.reset()
        state = np.expand_dims(state, axis=0)
        test_env1.render()
        test_env1.start_video_recorder()
        timeSteps = 1000
        rewards = []

        for t in range(timeSteps):
            # Select only greedy actions
            Q_values = self.onlineNet.predict(state, verbose=0)
            action = np.random.choice(np.where(Q_values[0,:]==np.max(Q_values[0,:]))[0])
            state, reward, terminated, truncated, _ = test_env1.step(action)
            state = np.expand_dims(state, axis=0)
            test_env1.render()
            rewards.append(reward)
            time.sleep(0.05)
            if (terminated):
                time.sleep(1)
                break
        test_env1.close_video_recorder()
        test_env1.close()
        test_env.close()
        return sum(rewards), test_env
