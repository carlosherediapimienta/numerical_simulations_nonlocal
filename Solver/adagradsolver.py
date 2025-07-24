import jax.numpy as jnp
from .base.common import _NonlocalSolverBase, DTYPE
import numpy as np
from typing import Callable



class AdaGrad:
    def __init__(self, dL: Callable, lr: float = 0.01, epsilon: float = 1e-8, lr_decay: float = 0,
                 lambda_l2: float = 0, epochs: int = 1000):
        self.lr = lr
        self.epsilon = epsilon
        self.lr_decay = lr_decay
        self.lambda_l2 = lambda_l2
        self.dL = dL
        self.accumulated_gradients = 0
        self.iteration = 1
        self.epochs = epochs
        self.global_error_tolerance = 1e-5
        self.theta_result = []
        self.accumulated_gradients_result = []

    def __global_error__(self, theta_new: float, theta_old: float) -> float:
        diff = theta_new - theta_old
        return np.abs(diff)

    def solve(self, theta_initial):
        theta = theta_initial
        while self.iteration <= self.epochs:
            
            self.theta_result.append(theta)
            self.accumulated_gradients_result.append(self.accumulated_gradients)

            theta_old = theta
            dL_value = self.dL(theta)

            if self.lambda_l2 != 0:
                dL_value += self.lambda_l2 / 2 * theta

            self.accumulated_gradients += dL_value ** 2
            update = self.lr * dL_value / (np.sqrt(self.accumulated_gradients) + self.epsilon)

            if self.lr_decay != 0:
                theta -= 1/(1 + (self.iteration - 1) * self.lr) * update
            else:
                theta -= update
            
            global_error = self.__global_error__(theta_new=theta, theta_old=theta_old)

            if self.iteration % 50 == 0:
                print(f'Epoch: {self.iteration}, Error: {global_error}.')

            self.iteration += 1

        print(f'Last epoch: {self.iteration}, Error: {global_error}.')

        return self.theta_result, self.accumulated_gradients_result, self.iteration
    

class NonlocalSolverAdaGrad(_NonlocalSolverBase):
    def __init__(self, *args, lr_decay: float = 0.0, **kw):
        super().__init__(*args, **kw)
        self.lr_decay = DTYPE(lr_decay)
        self.eps      = DTYPE(1e-8)
        self._alpha_t = lambda τ: 1. / (1. + τ * (self.lr_decay / self.alpha))

    def _build_stats(self, y):
        interp = self._interp(y)
        g      = jax.vmap(lambda τ: self.dL(interp(τ))
                        + 0.5 * self.lambda_ * interp(τ))(self.t)
        G_sqrt = jnp.sqrt(jnp.cumsum(g*g))
        return (y, g, G_sqrt, jax.vmap(self._alpha_t)(self.t))

    def _rhs(self, t, y_prev, idx, y_fix, g, G_sqrt, a_t):
        y_val = interp1d(t, self.t, y_fix, method="cubic")
        return self.f(t, y_val) - a_t[idx] * g[idx] / (G_sqrt[idx] + self.eps)
    
