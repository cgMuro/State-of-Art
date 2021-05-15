* [*Asynchronous Methods for Deep Reinforcement Learning*](https://arxiv.org/abs/1602.01783) (Paper)
* [PyTorch Implementations of Asynchronous Advantage Actor Critic](https://github.com/ikostrikov/pytorch-a3c) by Ilya Kostrikov


## Pseudocode
```
1. Global parameters θ and w  |  Thread specific parameters θ' and w'
2. Initialize time step t = 1
3. While T <= T_max
        (1) Reset gradients: dθ = 0 and dw = 0
        (2) Synchronous thread specific params with global params: θ' = θ and w' = w
        (3) t_start = t and sample a starting state s_t
        (4) While (s_t ≠ TERMINAL) and t - t_start <= t_max
               1. pick the action A_t ~ π_θ'(A_t|S_t) and receive reward R_t
               2. Update t = t + 1 and T = T + 1
        (5) Initialize the variable that holds the return estimation:                 
            R = 0 if s_t is TERMINAL else V_w'(s_t)
        (6) For i = t-1, ..., t_start:
               1. R <- γR + R_i  , R = Monte Carlo measure for G_i
               2. Accumulate gradients with respect to θ':
                            dθ <- dθ + ∇_θ'log(π_θ(a_i|s_i))(R - V_w'(s_i))
                  Accumulate gradients with respect to w':
                            dw <- dw + 2(R - V_w'(s_i))∇_w'(R - V_w'(s_i))
        (7) Update asynchronously θ using dθ and w using dw
```