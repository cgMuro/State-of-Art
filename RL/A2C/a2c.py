# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/algo/kfac.py
# For the complete version see the repository liked above.

import torch
import torch.nn as nn


class A2C():
    def __init__(
        self,
        actor_critic,                  # Function that defines the policy
        value_loss_coef: float = 0.5,  # Value of the loss coefficient
        entropy_coef: float = 0.01,    # Entropy term coefficient
        lr: float = 7e-4,              # Learning rate
        eps: float = 1e-5,             # RMSprop optimizer epsilon
        alpha: float = 0.99,           # RMSprop optimizer alpha
        max_grad_norm: float = 0.5     # Max norm of gradients
    ) -> None:

        self.actor_critic = actor_critic
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        # Init RMSprop optimizder
        self.optimizer = torch.optim.RMSprop(actor_critic.parameters(), lr=lr, eps=eps, alpha=alpha)

    def update(self, rollouts):
        observation_shape = rollouts.obs.size()[2:]             # Get size of oberservation space in environment
        action_shape = rollouts.actions.size()[-1]              # Get size of action space in environment
        num_steps, num_processes, _ = rollouts.rewards.size()   # Get size of reward space in environment


        # Evaluate actions using current policy
        values, action_log_probs, dist_entropy, _ =  self.actor_critic.evaluate_actions(
            rollouts.obs[:-1].view(-1, *observation_shape),
            rollouts.recurent_hidden_states[0].view(-1, self.actor_critic.recurrent_hidden_state_size),
            rollouts.masks[:-1].view(-1, 1),
            rollouts.actions.view(-1, action_shape)
        )

        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        # Calculate advantages
        advantages = rollouts.returns[:-1] - values
        # Calculate value's loss and action's loss
        value_loss = advantages.pow(2).mean()
        action_loss = -(advantages.detach() * action_log_probs).mean()

        # Zero out gradients
        self.optimizer.zero_grad()
        # Calculate gradients
        (value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef).backward()
        # Clip gradients
        nn.utils.clip_grad.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
        # Update parameters
        self.optimizer.step()


        return value_loss.item(), action_loss.item(), dist_entropy.item()
