# https://github.com/ikostrikov/pytorch-a3c

import math
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------- MODEL ----------------------------------- #

def normalized_columns_initializer(weights: torch.Tensor, std: float = 1.0) -> torch.Tensor:
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True))
    return out

class ActorCritic(nn.Module):
    def __init__(self, num_inputs: int, action_space) -> None:
        super().__init__()

        # Define convolutional layeres
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        # Define LSTM Cell
        self.lstm = nn.LSTMCell(input_size=32 * 3 * 3, hidden_size=256)

        # Define linear layers
        num_outputs = action_space.n
        self.critic_linear = nn.Linear(256, 1)
        self.actor_linear = nn.Linear(256, num_outputs)

        # Init weights
        self.apply(self._init_weights)
        self.actor_linear.weight.data = normalized_columns_initializer(self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = normalized_columns_initializer(self.critic_linear.weight.data, 0.01)
        self.critic_linear.bias.data.fill_(0)
        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

    def _init_weights(self, module) -> None:
        """ Handles the weights initialization """
        if isinstance(module, nn.Conv2d):
            # Convolutional layer
            weight_shape = list(module.weight.data.size())
            fan_in = np.prod(weight_shape[1:4])
            fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
            w_bound = np.sqrt(6. / (fan_in + fan_out))
            module.weight.data.uniform_(-w_bound, w_bound)
            module.bias.data.fill_(0)
        elif isinstance(module, nn.Linear):
            # Linear layer
            weight_shape = list(module.weight.data.size())
            fan_in = weight_shape[1]
            fan_out = weight_shape[0]
            w_bound = np.sqrt(6. / (fan_in + fan_out))
            module.weight.data.uniform_(-w_bound, w_bound)
            module.bias.data.fill_(0)


    def forward(self, inputs):
        inputs, (hx, cx) = inputs
        
        # Pass inputs into convolutional layers
        x = F.elu(self.conv1(inputs))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))

        x = x.view(-1, 32 * 3 * 3)

        # Pass inputs into the LSTM layers
        hx, cx = self.lstm(x, (hx, cx))
        x = hx

        return self.critic_linear(x), self.actor_linear(x), (hx, cx)


# ----------------------------------- OPTIMIZATION ----------------------------------- #

class SharedAdam(torch.optim.Adam):
    """ Adam algorithm with shared states """
    def __init__(
        self, 
        params, 
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999), 
        eps: float = 1e-8, 
        weight_decay: float = 0
    ) -> None:
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        # Iterate over the tensors in the current parameter group
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1)
                state['exp_avg'] = p.data.new().resize_as_(p.data).zero_()
                state['exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_()
        
    def share_memory(self) -> None:
        """ 
            Handles the sharing of memory in multiprocessing. 
            After tensors are shared, it will be possible to send them to other processes without making any copies. 
        """
        # Iterate over the tensors in the current parameter group
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]                 # Get the current parameter from state
                state['step'].share_memory_()         # Moves the state's step to shared memory
                state['exp_avg'].share_memory_()      # Moves the state's exp_avg to shared memory
                state['exp_avg_sq'].share_memory_()   # Moves the state's exp_avg_sq to shared memory

    def step(self, closure=None):
        """
        Performs a single optimization step.

        Arguments:
            - closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """
        
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                # Init grad and state
                grad = p.grad.data
                state = self.state[p]

                # Get exp_avg and exp_avg_sq from state
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                # Geta beta values from group
                beta1, beta2 = group['betas']

                # Update state's step
                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                # Handle epsilon
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                # Handle biases
                bias_correction1 =  1 - beta1 ** state['step'].item()
                bias_correction2 =  1 - beta2 ** state['step'].item()
                # Handle step size
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss
