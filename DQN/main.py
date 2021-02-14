# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

# Define gym environment
env = gym.make('CartPole-v0').unwrapped

# Set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# Set up GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Maps the (state, action) pair to its (next_state, reward) result
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# Class that holds the transitions observed recently
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
    
    def push(self, *args):
        '''Saves a transition'''
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity # increase the position by one, but when it reaches the capacity set it back to 0

    def sample(self, batch_size):
        '''Get a random batch (of size = batch_size) of transitions for training'''
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)



# BUILD THE MODEL #
class DQN(nn.Module):
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=(5, 5), stride=2)
        self.bn1 = nn.BatchNorm2d(num_features=16) # Applies batch normalization over a 4D input
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(5, 5),  stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(5, 5), stride=2)
        self.bn3 = nn.BatchNorm2d(32)


        # Calculate the output of the convolutional layer
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32

        # Linear output layer
        self.head = nn.Linear(linear_input_size, outputs)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))



# INPUT EXTRACTION #

# Define transformation steps
resize = T.Compose([
    T.ToPILImage(),
    T.Resize(40, interpolation=Image.CUBIC),
    T.ToTensor()
])

# Get location of the cart function
def get_cart_location(screen_width):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # Middle of the cart

# Handle screen 
def get_screen():
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    # Strip off the top and bottom of the screen
    _, screen_height, screen_width = screen.shape
    screen = screen[:, int(screen_height * 0.4):int(screen_height * 0.8)]
    view_width = int(screen_width * 0.6)
    cart_location = get_cart_location(screen_width)

    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2, cart_location + view_width // 2)

    # Strip off the edges, so that we have a square image centered on a cart
    screen = screen[:, :, slice_range]
    # Convert to float and rescale
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    # Convert to torch tensor
    screen = torch.from_numpy(screen)
    # Resize by applying transformations defineed above
    screen = resize(screen)
    # Add batch dimesion
    screen = screen.unsqueeze(0)

    return screen.to(device)


env.reset()
plt.figure()
plt.imshow(get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(), interpolation='none')
plt.title('Example extracted screen')
plt.show()



# TRAINING #

BATCH_SIZE = 128
GAMMA = 0.999        # Discount rate --> calculates the future discounted reward
EPS_START = 0.9      # Initial exploration rate --> the initial rate at which the agent randomly picks the action
EPS_END = 0.05       # Minimum exploration rate --> the minimum rate at which the agent randomly picks the action
EPS_DECAY = 200      # Decay of exploration rate --> the rate at which the exploration rate will be reduced
TARGET_UPDATE = 10

# Screen initial set up
init_screen = get_screen()
_, _, screen_height, screen_width = init_screen.shape

# Get number of actions from gym action space
n_actions = env.action_space.n

policy_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net = DQN(screen_height, screen_width, n_actions).to(device)
# Load into "target_net" the architecture along with the learned parameters of "policy_net"
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
# Init memory with a capacity of 10000
memory = ReplayMemory(10000) 

steps_done = 0

# Function to select which action to take given a state
def select_action(state):
    global steps_done
    # Get a random number between 0.0 and 1.0
    sample = random.random()
    # Get threshold needed to decided if we want to explore or take a decision
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    # Increase the epoch
    steps_done += 1

    # If the random number is over the threshold -> the agent takes an action
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    # If the random number is not over the threshold -> the action is taken randomly, hence the environment is explored
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

episode_durations = []

def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())

    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
    
    # Pause a bit so that plots are updated
    plt.pause(0.001) 
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


# TRAINING LOOP #

def optimize_model():
    # If there is not enough data in the memory return
    if len(memory) < BATCH_SIZE:
        return

    # Get a batch of transition from memory
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch -> this converts batch-array of Transitions to Transition of batch-arrays
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    
    # The model computes Q(s_{t}), then we select the columns of actions taken
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA)  + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


num_episodes = 50

for i_episode in range(num_episodes):
    # Init the environment and state
    env.reset()
    last_screen = get_screen()
    current_screen = get_screen()
    state = current_screen - last_screen

    for t in count():
        # Select and perform an action
        action = select_action(state)
        _, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)

        # Observe new state
        last_screen = current_screen
        current_screen = get_screen()
        if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None
        
        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        optimize_model()
        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break
        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

print('Complete')
env.render()
env.close()
plt.ioff()
plt.show()
