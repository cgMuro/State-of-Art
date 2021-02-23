# https://pytorch.org/docs/master/_modules/torch/optim/adam.html

import torch
import torch.optim.functional as F
from torch.optim import Optimizer

# Inherit from Optimizer class which handles all the general optimizer functionalities
class Adam(Optimizer):
    """
    Implements Adam algorithm.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.
    The implementation of the L2 penalty follows changes proposed in
    `Decoupled Weight Decay Regularization`_.

    Args:
        -->   params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        -->   lr (float, optional): learning rate (default: 1e-3)
        -->   betas (Tuple[float, float], optional): coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999))
        -->   eps (float, optional): term added to the denominator to improve numerical stability (default: 1e-8)
        -->   weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        -->   amsgrad (boolean, optional): whether to use the AMSGrad variant of this algorithm from the paper `On the Convergence of Adam and Beyond`_ (default: False)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False):
        # Handle invalid attribute values
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        # Init default dictionary
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_dacay=weight_decay, amsgrad=amsgrad)

        super(Adam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Adam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad() # No grad decorator
    def step(self, closure=None):
        """
        Performs a single optimization step.
           Args:
             --> closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """
        # Init loss
        loss = None

        # Check if the closure is already defined
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Iterate through the parameters. A parameter is an iterable that specifies what Tensors should be optimized.
        for group in self.param_groups:
            # Init all the variables that we need to do the computation
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            state_sums = []
            max_exp_avg_sqs = []
            state_steps = []

            # Iterate over the tensors in the current parameter
            for p in group['params']:
                # Check if the tensor's gradient is already defined
                if p.grad is not None:
                    params_with_grad.append(p)
                    # Check if the tensor's gradient is sparse
                    if p.grad.is_sparse:
                        raise RuntimeError('Adam does not support sparse gradients, please consideer SparseAdam instead')

                    grads.append(p.grad)

                    # Get the current optimizer state
                    state = self.state[p]
                    # Lazy state initialization -> check if it's the first time we access the state, if it is then we need to set some attributes
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p,  memory_format=torch.preserve_format)
                        if group['amsgrad']:
                            # Maintains max of all exponential moving averages of sq. grad. values
                            state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])

                    if group['amsgrad']:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                    # Update the steps for each param group update
                    state['step'] += 1
                    # Record the step after step update
                    state_steps.append(state['step'])

            beta1, beta2 = group['betas']

            # Functional API that performs Adam algorithm computation
            F.adam(
                params=params_with_grad,
                grads=grads,
                exp_avgs=exp_avgs,
                exp_avg_sqs=exp_avg_sqs,
                max_exp_avg_sqs=max_exp_avg_sqs,
                state_steps=state_steps,
                amsgrad=group['amsgrad'],
                beta1=beta1,
                beta2=beta2,
                lr=group['lr'],
                weight_decay=group['weight_decay'],
                eps=group['eps']
            )

        return loss
