# UDACITY REINFORCED LEARNING NANODEGREE: Project 1 Navigation  
  
### 0. Project Details  
  
This project is part of the RL learning nanodegree from UDACITY. In this project we build an Agent able to play a "Banana Collection game"  where the goal is to collect as many yellow bananas as possible while avoiding blue bananas.  
The Agent will evolve in an environment built using [Unity](https://blogs.unity3d.com/2017/09/19/introducing-unity-machine-learning-agents/) 
  
#### The Environment  
  
We train an agent to navigate (and collect bananas!) in a large, square world.  The agent's goal is to collect as many yellow bananas as possible while avoiding blue bananas.  
  ![BananaWorld](./images/banana_world.jpg)
##### Environment Observation space  
The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction. Given this information, the agent has to learn how to best select actions.  
  
##### Action space  
To navigate this banana world, four discrete actions are available, corresponding to:  
- **`0`** - move forward.  
- **`1`** - move backward.  
- **`2`** - turn left.  
- **`3`** - turn right.  
  
##### Reward mechanism  
A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana  
  
##### Criteria for solving the environment  
The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.  
  
### 1. Getting Started  
  
#### Preamble  

This project was developed on a **Windows 64 bits** platform using **CPU** computation.

To launch the project you will require installing Unity environment and its dependencies. The Windows installation is covered [here](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation-Windows.md)

#### Installing

As suggested in the Unity walkthrough, it is recommended creating a new *conda*  environment. Below are some additional recommendations where we used "p1_navigation" as the name for our environment:

* Download this project and decompress it in a local folder (e.g: p1_navigation)
* Open the command windows (cmd) and select the project folder as your current directory (cd "path")
* After executing the following steps from the procedure
 `conda env create -n p1_navigation`
 `activate p1_navigation`
 `pip install tensorflow==1.7.1`
 Install the git package to clone the ml-agent repository
 `conda install git`
Clone the ml-agent git repository
 `git clone https://github.com/Unity-Technologies/ml-agents.git`
* Go into the ml-agents python folder and perform the installation: 
 `cd ml-agents\python`  
 `pip install .`  
 
#### Adding Pytorch
This project uses the Pytorch library and you will need to install it. Instructions can be found [here](https://pytorch.org/)
 With your environment activated enter following commands:
`conda install pytorch-cpu -c pytorch`
`pip3 install torchvision`


#### One last step
For the project to run you will require the Windows version of the Banana runtime. This has been provided in the project folder and just need to be uncompressed in the same folder `./Banana_Windows_x86_64/`

### Run the project  
 
 #### Structure

 The project folder contains three different sub-folder  
1. 0_Simple_DQN : DQN implementation solving the environment. DQN algorithm is discussed in this [paper](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
2. 1_Prio_replay : DQN implementation using prioritised replay. An improvement over DQN where experiences that are likely to provide better "lessons" are prioritised when sampling. This algorithm is discussed in this [paper](https://arxiv.org/abs/1511.05952) 
3. 2_Pixel_DQN : DQN implementation now only using the visual information.

#### Execution

To execute the project go to into the folder 
`cd 0_Simple_DQN `
And launch the jupyter notebook
`jupyter notebook Navigation_DQN.ipynb`

