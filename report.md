[//]: # (Image References)

[image1]: https://github.com/schambon77/DRLND-Continuous-Control/blob/master/Scores.JPG "Plot of Rewards"

# Project 2: Continuous Control

## Technical Details

The solution is based in most parts on the [DDPG implementation for the pendulum environment](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum)
 provided in the Udacity [deep-reinforcement-learning](https://github.com/udacity/deep-reinforcement-learning).

### Learning Algorithm

The [DDPG](https://arxiv.org/abs/1509.02971) (Deep Deterministic Policy Gradient) algorithm is an Actor-Critic type of
reinforcement learning algorithm, with similarities to 
[DQN](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) (Deep Q-Networks).

One limit of the DQN algorithm is its inability to deal with continuous action space. DDPG implements the actor-critic 
concept to palliate this problem, where:
- the actor trains a neural network to estimate the optimal deterministic policy (best believed action for any given state)
- and the critic trains a neural network to estimate the action-value function using the estimated optimal policy

As the actor is prone to high varience, and the critic to high bias, the combination of the 2 helps for an efficient
reinforcement learning algorithm.

To avoid instabilities in training non-linear function approximators, DDPG uses 2 fundamental improvements illustrated 
by DQN:
- the use of a replay buffer
- and the use of target networks to stabilize the training of local networks for both actor and critic

Furthermore, DDPG adds the concept of soft updates to slowly blend the trained local networks weights into 
the target networks at every timestep, rather than regular although abrupt weight updates.

Here are the main steps of the DDPG algorithm:
- initialize critic and actor local networks
- copy the local weights to both target networks
- initialize a replay buffer
- iterate over N episodes
- for each episode:
    - get first observation from environment
    - iterate over T timesteps
    - for each timestep, until terminal state encountered:
        - select an action with the local actor network (with noise added for exploration purposes)
        - execute the selected action in the environment and observe the new state and reward
        - store the experience tuple in the replay buffer
        - sample a random minibatch from the replay buffer
        - compute the target action-value = reward + discounted action-value of next state and action using critic target network
        - compute the local action-value using the local critic network 
        - update local critic weights by minimizing the critic loss between local and target action-value
        - compute the predicted action with local actor network
        - compute the actor loss using the local critic network with the state and predicted action
        - update local actor weights by minizing the actor loss
        - perform a soft-update of both critic and actor target network weights

#### Networks

This section reuses the provided source code for the DDPG pendulum as is with no modification.

[model.py](https://github.com/schambon77/DRLND-Continuous-Control/blob/master/model.py)

##### Actor

The `Actor` class implements a neural network responsible to capture the policy model, i.e. providing a mapping 
between the current state and actions to take.

Compared to the DQN technique, this network is necessary to deal with the continuous nature the action space.

The network is defined as:

| Input Size        | Layer           | Output Size |
| ------------- |:-------------:| -----:|
| 33       |  Fully Connected    | 400 |
| 400      | RELU     |   400 |
| 400 | Fully Connected     |    300 |
| 300      | RELU     |   300 |
| 300 | Fully Connected     |    4 |
| 4 | tanh     |    4  |

The `forward` pass of the model takes the current state as input of the first layer, and returns the recommended actions
at the output of the last activation layer


##### Critic
   
The `Critic` class implements a neural network responsible to capture the action-value function, i.e. predicting 
the expected return for a given (state, action) pair.

   
| Input Size        | Layer           | Output Size |
| ------------- |:-------------:| -----:|
| 33       |  Fully Connected    | 400 |
| 400      | RELU     |   400 |
| **404** | Fully Connected     |    300 |
| 300      | RELU     |   300 |
| 300 | Fully Connected     |   1 |

The `forward` pass of the model takes the current state as input of the first layer, injects the actions taken at the
 second fully connected layer and returns the expected return.


#### Agent

This section reuses the provided source code for the DDPG pendulum with some modifications to adapt to 20 agents.

[ddpg_agent.py](https://github.com/schambon77/DRLND-Continuous-Control/blob/master/ddpg_agent.py)

The `Agent` class constructor `__init__` takes the size of the state (33) and action (4) space as input, but also the number of agents
that are interacting with the environment in parallel (20). A local and target instance of each `Actor` and `Critic` models
are created, as well as an instance of a unique `ReplayBuffer` (of size 1e5) in order to store all experience tuples.

The `act` function is passed the current observed states for all 20 agents. The local `Actor` model is called for each agent
with its respective state to get the recommended actions to take.
- Note1: some noise defined as *Ornstein-Uhlenbeck process* is added in order to help the agents explore the environment.
- Note2: the noise generation is altered from the original source code after reviewing helpful discussions on the Udacity
dedicated Slack channels, replacing `np.random.random` by `np.random.standard_normal`.

The `step` function is passed the past and new states from the environment for all 20 agents, once their actions are taken, 
as well as the corresponding reward. Each separate agent experience tuple is stored in the `ReplayBuffer` instance. 
Once its size has reached the minimum size, a batch of size 128 is sampled from it in order to update the local and 
target networks for both `Actor` and `Critic` modules. A discount rate of 0.99 is used for the update. We end up having 
at every timestep 20 new experience tuples, but only 1 update of the models. This turned out to be a good ratio to avoid 
too many updates and training instability issues.

The `learn` function implements the models update. It uses Adam optimizers to minimze the MSE
error loss function between expected and target weights.
- Note1: as suggested in the project benchmark implementation, we added a clipping of the `Critic` local parameters with a
call to `clip_grad_norm_`.
- Note2: the update of target network parameters from local network is performed a soft update technique using the 
parameter tau set to 1e-3.

#### Training

[Continuous_Control.ipynb](https://github.com/schambon77/DRLND-Continuous-Control/blob/master/Continuous_Control.ipynb)

The first cells set up the environment and create the `Agent` instance based on the state and action space size, and the 
number of separate agents interacting with the environment.

The function `ddpg` implements the training of the `Agent` instance.

The main loop performs episodes iteratively. At the start of each episode, we reset:
- the `Agent`
- the environment
- and the agent scores

At each timestep, the current states are passed to the `Agent` function `act` to get the next actions to take.

The environment returns the new states, rewards, and terminal statuses.

These new states are passed to the `Agent` function `step` to enrich its experience and perform one update of its models.

The rewards for the 20 agents are accumulated throughout the episode and averaged to compute a global score appended to 
a list for future display, and a queue which keeps the last 100 scores.

We iterate through each episode until a terminal state is found in the environment.

The training problem is considered solved when the average of the last 100 scores is above 30, at which point we save the 
local models parameters.

#### Demonstration

A short `play` function is implemented to demonstrate over 3 episodes of 2000 timesteps how the agents 
have learnt how to interact successfully with the environment. 

### Plot of Rewards

The training reaches the acceptance criteria after 283 episodes.

![Plot of Rewards][image1]


### Ideas for Future Work

The training was performed on a local CPU, and took a couple of hours. Interestingly, 
the GPU-enabled online workspace from Udacity didn't help much in speeding the training.

For future work, I'd be very interested in implementing a technique like D4PG
more prone to GPU (hence faster) training. I also understand that having separate agents
gettting trained independently and sharing updates at regular times could help getting a 
better (and faster) model. To be investigated (time-permitting of course!).
