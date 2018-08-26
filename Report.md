

# UDACITY REINFORCED LEARNING NANODEGREE: Project 1 Navigation  
---
# Report

  
## 1. First algorithm : DQN  

### 1.1 Description

#### 1.1.1 Deep Qlearning
The first chosen algorithm is DQN (see [Human-level control through deep reinforcement learning](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf))

DQN is a temporal-difference learning method meaning an agent does not until the end of the episode to learn but learn at the end of each time step. The proposed environment in this project provides a reward a each time step. If a yellow banana is collected a reward of +1 is received, if a blue is collected a reward of -1 is received and the game ends and in the other cases no reward is given. 
It is therefore appropriate to analyse if at each time step the taken action  increased the number of banana we collected.

DQN is based on Qlearning (Watkins, 1989) and leverages from neural networks to build a non linear function able to approximate the action-value function q*. Our observation space has a dimension of 37 in a continuous space with 4 possible discrete actions. As described in the paper previously referenced this method showed good result in Atari games environment equivalent to the one proposed here.

The evaluation of the action value is done using a Qnetwork initialised when creating the agent:

`self.qnetwork_local = QNetwork(state_size, action_size, seed).to(DEVICE)`

The action value will be retrieved using an epsilon greedy policy using the Qnetwork to evaluate the current state:

`action_values = self.qnetwork_local(state)`

#### 1.1.2 Experience Replay and Fixed Q targets
The algorithm also implements two other concepts:
* Experience replay
* Fixed Q target

*Experience replay* helps in removing the correlation between consecutive experiences. The reward obtained at a time step t could be linked to actions and states from the previous timestep. To prevent the algorithm from being induced in error, experiences are collected and stored in a memory. At a certain time step, the agent stops exploring a sample randomly its memory to learn.

`self.t_step = (self.t_step + 1) % UPDATE_EVERY`
`experiences = random.sample(self.memory, k=self.batch_size)`

*Fixed Q target* helps with the error computation stability. In Qlearning the target is approximated using the current Qnetwork. Meaning the target is updated each time we improve the network with new learning.  To avoid this correlation another Qnetwork to evaluate the target is created and updated with a rule minimising the effect of the learning

`self.qnetwork_target = QNetwork(state_size, action_size, seed).to(DEVICE) # declaration of Qnetwork used for targets`
`θ_target = τ*θ_local + (1 - τ)*θ_target # update rule for Q target`

### 1.2 Hyperparameters

* Parameters used for the epsilon greedy policy
Epsilon starts at 1 to favour exploration and will slowly decrease towards eps_end. 

 `eps_start=1.0` 
`eps_end=0.01`
`eps_decay=0.995`

* DQN parameters
Define the replay buffer size (number of stored experiences)

`BUFFER_SIZE = int(1e5)  # replay buffer size`

Define minibatch size used for learning

`BATCH_SIZE = 64         # minibatch size`

DQN uses a local QNetwork and a target QNetwork for stability purposes. The target network weights are updated with local weights using soft update using parametre TAU

`TAU = 1e-3              # for soft update of target parameters`

Learning rate used for optimiser 

`LR = 5e-4               # learning rate `

Learn every UPDATE_EVERY steps. Decide between learning and continuing exploring 

`UPDATE_EVERY = 4        # how often to update the network`

GAMMA used in expected reward computation. 

`GAMMA = 0.99            # discount factor`

* Optimizer
Adam optimizer is chosen over SGD as it converges faster in early stages of training (see [Improving Generalization Performance by Switching from Adam to SGD](https://arxiv.org/pdf/1712.07628.pdf)

`self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)`

### 1.3 Result
The environment is considered solved when the agent is able to harvest an average of 13 bananas over 100 episodes. In order to ensure the agent would pass the test outside the training session, the target was fixed to an average of 14.
The figure below shows that the Agent is able to reach an average score of at least 14 over 100 episodes in less than 600 episodes (here 554)
![Agent learning curve with DQN _](./images/DQN_result.jpg)

Several attempts were made and the score of 14 is reached in average after 550 episodes

### 1.4 Improvement

DQN can be improved using methods such as :
* DDQN (see :[Double DQN paper](https://arxiv.org/abs/1509.06461)) to avoid overestimating action values
* Prioritised experience replay to select memories according to their learning potential.(see : [Prioritised replay](https://arxiv.org/abs/1511.05952))

The next section describe my attempt with the implementation of prioritised replay.

## 2. DQN and prioritised replay  

This implementation impacts the memory structure and the construction of the minibatch.  Instead of having a standard buffer, each memory is now stored with its priority which is equal to the absolute value of the TD-error.


