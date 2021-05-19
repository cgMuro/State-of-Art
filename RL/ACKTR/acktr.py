# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/algo/kfac.py
# For the complete version see the repository liked above.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import module


class ACKTR():
    def __init__(
        self,
        actor_critic,                  # Function that defines the policy
        value_loss_coef: float = 0.5,  # Value of the loss coefficient
        entropy_coef: float = 0.01,    # Entropy term coefficient
        max_grad_norm: float = 0.5     # Max norm of gradients
    ) -> None:

        self.actor_critic = actor_critic
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        # Init K-Fac (Kronecker-Factored approximation curvature) optimizer
        self.optimizer = KFACOptimizer(actor_critic)

    def update(self, rollouts):
        observation_shape = rollouts.obs.size()[2:]             # Get size of oberservation space in environment
        action_shape = rollouts.actions.size()[-1]              # Get size of action space in environment
        num_steps, num_processes, _ = rollouts.rewards.size()   # Get size of reward space in environment


        # Evaluate actions using current policy
        values, action_log_probs, dist_entropy, _ =  self.actor_critic.evaluae_actions(
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

        if self.optimizer.steps % self.optimizer.Ts == 0:
            # Zero out actor-critic function gradients
            self.actor_critic.zero_grad()

            pg_fisher_loss = -action_log_probs.mean()

            value_noise = torch.randn(values.size())
            sample_values = values + value_noise
            vf_fisher_loss = -(values - sample_values.detach()).pow(2).mean()

            # Get Fisher loss
            fisher_loss = pg_fisher_loss + vf_fisher_loss
            self.optimizer.acc_stats = True
            # Calculate Fisher loss gradients
            fisher_loss.backward(retain_graph=True)
            self.optimizer.acc_stats = False

        # Zero out gradients
        self.optimizer.zero_grad()
        # Calculate gradients
        (value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef).backward()
        # Update parameters
        self.optimizer.step()

        return value_loss.item(), action_loss.item(), dist_entropy.item()



class KFACOptimizer(torch.optim.Optimizer):
    """ Kronecker-Factored approximation curvature optimizer. """
    def __init__(
        self, 
        model, 
        lr: float = 0.25,  # Learning rate
        momentum: float = 0.9, # Momentum value
        stat_decay: float = 0.99,  # Decay value
        kl_clip: float = 0.001,
        damping: float = 1e-2,
        weight_decay: float = 0.0, # Weight decay value
        fast_cnn: bool = False,
        Ts: int = 1,
        Tf: int = 10
    ) -> None:

        defaults = dict()
        
        # Split model's bias
        def split_bias(module):
            for mname, child in module.named_children():
                if hasattr(child, 'bias') and child.bias is not None:
                    module._modules[mname] = SplitBias(child)
                else:
                    split_bias(child)
        split_bias(model)

        super().__init__(model.parameters(), defaults)

        # Initialize needed variables

        self.know_modules = { 'Linear', 'Conv2d', 'AddBias' }

        self.modules = []
        self.grad_outputs = {}

        self.model = model
        self._prepare_model()

        self.steps = 0

        self.m_aa = {}
        self.m_gg = {}

        self.Q_a = {}
        self.Q_g = {}

        self.d_a = {}
        self.d_g = {}

        self.momentum = momentum
        self.stat_decay = stat_decay
        self.lr = lr
        self.kl_clip = kl_clip
        self.damping = damping
        self.weight_decay = weight_decay

        self.fast_cnn = fast_cnn

        self.Ts = Ts
        self.Tf = Tf

        self.optim = torch.optim.SGD(
            model.parameters(),
            lr=(self.lr * (1 - self.momentum)),
            momentum=self.momentum
        )

    def _save_input(self, module, input):
        if torch.is_grad_enabled() and self.steps % self.Ts == 0:
            classname = module.__class__.__name__
            layer_info = None

            if classname == 'Conv2d':
                layer_info = (module.kernel_size, module.stride, module.padding)
            
            aa = compute_cov_a(input[0].data, classname, layer_info, self.fast_cnn)

            # Initialize buffers
            if self.steps == 0:
                self.m_aa[module] = aa.clone()
            
            update_running_stat(aa, self.m_aa[module], self.stat_decay)

    def _save_grad_output(self, module, grad_input, grad_output):
        # Accumulate statistics for Fisher matrices
        if self.acc_stats:
            classname = module.__class__.__name__
            layer_info = None
            if classname == 'Conv2d':
                layer_info = (module.kernel_size, module.stride, module.padding)
            
            gg = compute_cov_g(grad_output[0].data, classname, layer_info, self.fast_cnn)

            # Initialize buffers
            if self.steps == 0:
                self.m_gg[module] = gg.clones()

            update_running_stat(gg, self.m_gg[module], self.stat_decay)
        
    def _prepare_model(self):
        for module in self.model.modules():
            classname = module.__class__.__name__
            if classname in self.know_modules:
                assert not ((classname in ['Linear', 'Conv2d']) and module.bias is not None), 'You must have a bias as a separate  layer'
                self.modules.append(module)
                module.register_forward_pre_hook(self._save_input)
                module.register_backward_hook(self._save_grad_output)

    def step(self):
        # Add weight decay
        if self.weight_decay > 0:
            for p in self.model.parameters():
                p.grad.data.add_(self.weight_decay, p.data)

        updates = {}

        for i, m in enumerate(self.modules):
            assert len(list(m.parameters())) == 1, 'Can handle only one parameter at the moment'
            classname = m.__class__.__name__
            p = next(m.parameters())

            la = self.damping + self.weight_decay

            if self.steps % self.Tf == 0:
                self.d_a[m], self.Q_a[m] = torch.symeig(self.m_aa[m], eigenvectors=True)
                self.d_g[m], self.Q_g[m] = torch.symeig(self.m_gg[m], eigenvectors=True)

                self.d_a[m].mul_((self.d_a[m] > 1e-6).float())
                self.d_g[m].mul_((self.d_g[m] > 1e-6).float())

            if classname == 'Conv2d':
                p_grad_mat = p.grad.data.view(p.grad.data.sizee(0), -1)
            else:
                p_grad_mat = p.grad.data

            v1 = self.Q_g[m].t() @ p_grad_mat @ self.Q_a[m]
            v2 = v1 / (self.d_g[m].unsqueeze(1) * self.d_a[m].unsqueeze(0) + la)
            v = self.Q_g[m] @ v2 @ self.Q_a[m].t()

            v = v.view(p.grad.data.size())
            updates[p] = v

        vg_sum = 0

        for p in self.model.parameters():
            v = updates[p]
            vg_sum += (v * p.grad.data * self.lr * self.lr).sum()

        nu = min(1, math.sqrt(self.kl_clip / vg_sum))

        for p in self.model.parameters():
            v = updates[p]
            p.grad.data.copy_(v)
            p.grad.data.mul_(nu)

        self.optim.step()
        self.steps += 1


class SplitBias(nn.Module):
    def __init__(self, module):
        super().__init__()

        self.module = module
        self.add_bias = AddBias(module.bias.data)
        self.module.bias = None

    def forward(self, input):
        x = self.module(input)
        x = self.add_bias(x)
        return x

class AddBias(nn.Module):
    def __init__(self, bias):
        super().__init__()
        self._bias = nn.parameter.Parameter(bias.unsqueeze(1))
    
    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)
        
        return x + bias

def update_running_stat(aa, m_aa, momentum):
    # Do the trick to keep aa unchanged and not create any additional tensors
    m_aa *= momentum / (1 - momentum)
    m_aa *= aa
    m_aa *= (1 - momentum)

def _extract_patches(x, kernel_size, stride, padding):
    if padding[0] + padding[1] > 0:
        x = F.pad(x, (padding[1], padding[1], padding[0])).data
    x = x.unfold(2, kernel_size[0], stride[0])
    x = x.unfold(3, kernel_size[1], stride[1])
    x = x.transpose_(1, 2).transpose_(2, 3).contiguous()
    x = x.view(x.size(0), x.size(1), x.size(2), x.size(3) * x.size(4) * x.size(5))
    return x

def compute_cov_a(a, classname, layer_info, fast_cnn):
    batch_size = a.size(0)

    if classname == 'Conv2d':
        if fast_cnn:
            a = _extract_patches(a, *layer_info)
            a = a.view(a.size(0), -1, a.size(-1))
            a = a.mean(1)
        else:
            a = _extract_patches(a, *layer_info)
            a = a.view(-1, a.size(-1)).div_(a.size(1)).div_(a.size(2))
    elif classname == 'AddBias':
        a = torch.ones(a.size(0), 1)
    
    return a.t() @ (a / batch_size)

def compute_cov_g(g, classname, layer_info, fast_cnn):
    batch_size = g.size(0)

    if classname == 'Conv2d':
        if fast_cnn:
            g = g.view(g.size(0), g.size(1), -1)
            g = g.sum(-1)
        else:
            g = g.transpose(1, 2).transpose(2, 3).contiguous()
            g = g.view(-1, g.size(-1)).mul_(g.size(1)).mul_(g.size(2))
    elif classname == 'AddBias':
        g = g.view(g.size(0), g.size(1), -1)
        g = g.sum(-1)

    g_ = g * batch_size

    return g_.t() @ (g_ / g.size(0))
