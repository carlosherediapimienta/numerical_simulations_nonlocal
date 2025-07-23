import jax.numpy as jnp
from .base.common import _NonlocalSolverBase, _ema, DTYPE
import numpy as np
from typing import Callable

try:
    import jax
    import jax.numpy as jnp
    device = jax.devices()[0]
    print(f"[Solver] Usando JAX en: {device.device_kind} ({device.platform})")
except ImportError:
    raise ImportError("[Solver] JAX no está instalado. Instálalo para usar este optimizador.")

class RMSPropMomentum:
    def __init__(self, dL:Callable, lr:float=0.001, beta:float=0.9, epsilon:float=1e-8, weight_decay: float = 0, lambda_l2: float = 0, epochs: int = 1000):
        self.lr = lr
        self.beta = beta
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.lambda_l2 = lambda_l2
        self.v = 0
        self.dL = dL
        self.iteration = 1
        self.epochs = epochs
        self.global_error_tolerance = 1e-5
        self.theta_result = []
        self.v_result = []

    def __global_error__(self, theta_new: float , theta_old: float) -> float:
            diff = theta_new - theta_old
            return np.abs(diff)

    def solve(self, theta_initial):

        theta = theta_initial
        while self.iteration <= self.epochs:
            
            self.theta_result.append(theta)
            self.v_result.append(self.v)

            theta_old = theta
            dL_value = self.dL(theta)
            
            if self.lambda_l2 != 0:
                dL_value  += self.lambda_l2 / 2 * theta

            self.v = self.beta * self.v + (1 - self.beta) * (dL_value ** 2)            
            update = self.lr * dL_value / (np.sqrt(self.v) + self.epsilon)
            
            if self.weight_decay != 0:
                theta = (1 - self.weight_decay) * theta - update
            else:
                theta -= update
            
            global_error = self.__global_error__(theta_new=theta, theta_old=theta_old)

            if self.iteration % 50 == 0:
                print(f'Epoch: {self.iteration}, Error: {global_error}.')

            self.iteration += 1

        print(f'Last epoch: {self.iteration}, Error: {global_error}.')

        return self.theta_result, self.v_result, self.iteration
    
    
class NonlocalSolverMomentumRMSProp(_NonlocalSolverBase):
    def __init__(self, *args, beta: float = 0.99, **kw):
        super().__init__(*args, **kw)
        self.beta = DTYPE(beta)
        self.eps  = DTYPE(1e-8)

    def _build_stats(self, y):
        interp = self._interp(y)
        g = jax.vmap(lambda τ:
                     self.dL(interp(τ)) + 0.5 * self.lambda_ * interp(τ))(self.t)
        v = _ema(self.beta, g * g)
        return interp, g, jnp.sqrt(v)

    def _rhs(self, t, y_prev, idx, interp, g, v_sqrt):
        return self.f(t, interp(t)) - g[idx] / (v_sqrt[idx] + self.eps)