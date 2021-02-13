import numpy as np

class ScheduledOptim(object):
    """ A simple wrapper class for learning rate scheduling """

    def __init__(self, optimizer, n_warmup_steps):
        self.optimizer = optimizer
        self.d_model = 64
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.delta = 1

    def step(self):
        "Step by the inner optimizer"
        self.optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self.optimizer.zero_grad()

    def increase_delta(self):
        self.delta *= 2

    def update_learning_rate(self):
        "Learning rate scheduling per step"

        self.n_current_steps += self.delta
        new_lr = np.power(self.d_model, -0.5) * np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr

    def state_dict(self):
        ret = {
            'd_model': self.d_model,
            'n_warmup_steps': self.n_warmup_steps,
            'n_current_steps': self.n_current_steps,
            'delta': self.delta,
        }
        ret['optimizer'] = self.optimizer.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        self.d_model = state_dict['d_model']
        self.n_warmup_steps = state_dict['n_warmup_steps']
        self.n_current_steps = state_dict['n_current_steps']
        self.delta = state_dict['delta']
        self.optimizer.load_state_dict(state_dict['optimizer'])

