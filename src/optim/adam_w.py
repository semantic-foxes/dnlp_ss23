from typing import Callable, Iterable, Tuple

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        # Sanity check
        if lr < 0.0:
            message = f'Invalid learning rate: {lr} - should be >= 0.0'
            raise ValueError(message)
        if not 0.0 <= betas[0] < 1.0:
            message = f'Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0]'
            raise ValueError(message)
        if not 0.0 <= betas[1] < 1.0:
            message = f'Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0]'
            raise ValueError(message)
        if not 0.0 <= eps:
            message = f'Invalid epsilon value: {eps} - should be >= 0.0'
            raise ValueError(message)

        defaults = {
            'lr': lr,
            'betas': betas,
            'eps': eps,
            'weight_decay': weight_decay,
            'correct_bias': correct_bias
        }
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, '
                                       'please consider SparseAdam instead')

                # State should be stored in this dictionary
                state = self.state[p]

                # Access hyperparameters from the `group` dictionary
                alpha = group['lr']

                # First step handling
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(grad)
                    state['v'] = torch.zeros_like(grad)

                # m and v update
                state['step'] += 1
                m, v = state['m'], state['v']
                beta_1, beta_2 = group['betas']

                m = beta_1 * m + (1 - beta_1) * grad
                v = beta_2 * v + (1 - beta_2) * grad**2
                state['m'] = m
                state['v'] = v

                a = alpha
                if group['correct_bias']:
                    a *= torch.sqrt(1 - beta_2 ** state['step']) / (1 - beta_1 ** state['step'])

                # Parameter update
                p.data.add_(-a * m / (torch.sqrt(v) + group['eps']))
                # Weight decay
                p.data.add_(-p.data * alpha * group['weight_decay'])

        return loss
