# Reinforcement Learning agents for Cart Pole and Ms Pac Man  
The primary objective of this report is to explore the capabilities and behaviors of Q-Learning and DQN agents within two distinct environments: the Cart Pole and Pac-Man. .

## Dependencies

To run this project, you'll need the following dependencies:

- Python 
- Gymnasium
- numpy
- pandas
- matplotlib
- moviepy
- tensorflow

## Usage 

After you have installed all dependencies you can either download the .zip file manualy from GitHub or clone the repo with this command:

```console 
git clone https://github.com/ThomasLagkalis/RL-Agent-MsPacMan-CartPole.git
```

By running the following command (with the deafault parameters) it will run at the Ms Pac Man environment for 3000 episodes

```console 
python3 driver_code.py
```

To show all the available options you can run the following command:
```console
python3 driver_code.py --help
```

To run the cart pole environment add the following option:
```console
python3 driver_code.py --cartpole
```

To specify the duration (number of episodes) of the training process add the -d EPISODES argument. For example:
```console
python3 driver_code.py --cartpole -e 1000
```

If you want to edit the hyperparameters you can find them in drive_code.py (epsilon, gamma and alpha) and agent.py (minibatch size etc.).

## Results 

After you run the programm the results i.e. the graphs and an inscance of the agent navigating the environment are saved in ./visual



https://github.com/ThomasLagkalis/RL-Agent-MsPacMan-CartPole/assets/83509443/3b21b6c6-9061-4e8e-af5a-b03e433204ec



https://github.com/ThomasLagkalis/RL-Agent-MsPacMan-CartPole/assets/83509443/80e5d5e0-2fa0-4d9f-9af4-92c529214569





