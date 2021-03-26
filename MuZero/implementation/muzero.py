import torch
import torch.nn as nn
import torch.nn.functional as F
import typing
from typing import Dict, List
from resnet import ResNet



class Action(object):
    """ Holds logic for taking actions """
    def __init__(self, index: int):
        self.index = index
    
    def __hash__(self):
        return self.index

    def __eq__(self, other):
        return self.index == other.index

    def __gt__(self, other):
        return self.index >  other.index


class NetworkOutput(typing.NamedTuple):
    """ Defines network output type """
    value: float
    reward: float
    policy_logits: Dict[Action, float]
    hidden_state: List[float]

class RepresentationFunction(nn.Module):
    """ Implementation of representation function -> outputs hidden state """
    def __init__(self):
        super().__init__()

        # DOWNSAMPLE
        self.conv1 = nn.Conv2d(3, 128, kernel_size=(3, 3), stride=2)
        self.residual_block1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=2),
            nn.BatchNorm2d(128)
        )
        self.conv2 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=2)
        self.residual_block2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=2),
            nn.BatchNorm2d(256)
        )
        self.avgpool = nn.AvgPool2d(kernel_size=(3, 3), stride=2)

        # RESNET
        self.resnet = ResNet()

    def _downsample(self, x):
        x = self.conv1(x)
        x1 = self.residual_block1(x)
        x += x1   # Residual connection
        x = F.relu(x)

        x = self.conv2(x)
        x2 = self.residual_block2(x)
        x += x2   # Residual connection
        x =  F.relu(x)

        x = self.avgpool(x)
        x = self.resnet(x)

        return x

    def forward(self, x):
        # Downsample
        x = self._downsample(x)
        # ResNet
        x = self.resnet(x)
        # Normalization [0, 1] per channel
        x = x / 255.0

        return x

class PredictionFunction(nn.Module):
    """ Implementation of prediction function -> outputs value and policy """
    def __init__(self, action_space_size):
        super().__init__()

        # RESNET
        self.resnet = ResNet()

        # DOWNSAMPLING CONVOLUTION
        self.conv1x1 = nn.Conv2d(512, 64, kernel_size=1, stride=1, bias=False)

        # VALUE FULLY CONNECTED NETWORK
        self.fc_value_net = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, ?)
        )

        # POLICY FULLY CONNECTED NETWORK
        self.fc_policy_net = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_space_size)
        )

    def forward(self, x):
        # ResNet + downsampling + flattening
        x = self.resnet(x)
        x = self.conv1x1(x)
        x = torch.flatten(x, 1)
        # Get policy
        policy = self.fc_policy_net(x)
        policy = F.softmax(policy)
        # Get value
        value = self.fc_value_net(x)

        return policy, value

class DynamicsFunction(nn.Module):
    """ Implementation of dynamics function -> outputs reward """
    def __init__(self, action_space_size):
        super().__init__()

        # ONE-HOT-ENCODING
        self.one_hot_enc = ?

        # DOWNSAMPLING CONVOLUTIONS
        self.conv1x1_1 = nn.Conv2d(action_space_size, 64, kernel_size=1, stride=1, bias=False)
        self.conv1x1_2 = nn.Conv2d(512, 64, kernel_size=1, stride=1, bias=False)

        # RESNET
        self.resnet = ResNet()

        # REWARD FULLY CONNECTED NETWORK
        self.fc_reward_layers = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, ?)
        )

    def action_encoding_go(self, action_space_size, action):
        encoding = torch.zeros(action_space_size)
        encoding[torch.argmax(action)] = 1.0
        return encoding

    def forward(self, x):
        # One-hot-encoding
        x = self.one_hot_enc(x)
        # Downsample
        x = self.conv1x1_1(x)
        # ResNet
        out_res = self.resnet(x)
        # Downsample
        x = self.conv1x1_2(out_res)
        x = torch.flatten(x, 1)
        # Get reward
        x = self.fc_reward_layerss(x)
        # Normalize resnet output
        out_res = out_res / 255.0

        return out_res, x

class MuZero(nn.Module):
    def __init__(self):
        super().__init__()

        self.representation = RepresentationFunction()
        self.prediction = PredictionFunction(action_space_size=)
        self.dynamics = DynamicsFunction(action=)


    def initial_inference(self, image) -> NetworkOutput:
        return NetworkOutput(0, 0, {}, [])
    
    def recurrent_inference(self, hidden_state, action) -> NetworkOutput:
        # Dynamics + prediction function
        return NetworkOutput(0, 0, {}, [])

    def get_weights(self):
        # Returns the weights of this network
        return []

    def training_steps(self) -> int:
        # How many steps/batches the network has been trained for
        return 0
