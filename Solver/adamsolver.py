import numpy as np
from typing import Callable
from typing import Tuple
import jax.numpy as jnp
from interpax import interp1d
from .base.common import _NonlocalSolverBase, _ema, DTYPE

try:
    import jax
    import jax.numpy as jnp
    device = jax.devices()[0]
    print(f"[Solver] Usando JAX en: {device.device_kind} ({device.platform})")
except ImportError:
    raise ImportError("[Solver] JAX no está instalado. Instálalo para usar este optimizador.")

class AdamMomentum:
    def __init__(self, dL: Callable, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8, 
                 weight_decay: float = 0, lambda_l2: float = 0, epochs: int = 1000):

        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.lambda_l2 = lambda_l2
        if not hasattr(dL, 'lower'):
            self.dL = jax.jit(dL)
        else:
            self.dL = dL
        self.epochs = epochs
        self.global_error_tolerance = 1e-5
        self.reset_state()

    def reset_state(self):
        self.m = 0.0
        self.v = 0.0
        self.iteration = 1
        self.theta_result = []
        self.m_result = []
        self.v_result = []

    @staticmethod
    @jax.jit
    def __global_error__(theta_new, theta_old):
        import jax.numpy as jnp
        diff = theta_new - theta_old
        return jnp.abs(diff)

    def solve(self, theta_initial):
        self.reset_state()
        theta = jnp.array(theta_initial)
        while self.iteration <= self.epochs:
            self.theta_result.append(float(theta))
            self.m_result.append(float(self.m))
            self.v_result.append(float(self.v))

            theta_old = theta
            dL_value = self.dL(theta)
            if self.lambda_l2 != 0:
                dL_value += self.lambda_l2 / 2 * theta

            self.m = self.beta1 * self.m + (1 - self.beta1) * dL_value
            self.v = self.beta2 * self.v + (1 - self.beta2) * (dL_value ** 2)
            m_hat = self.m / (1 - self.beta1 ** self.iteration)
            v_hat = self.v / (1 - self.beta2 ** self.iteration)
            update = self.lr * m_hat / (jnp.sqrt(v_hat) + self.epsilon)

            if self.weight_decay != 0:
                theta = (1 - self.weight_decay) * theta - update
            else:
                theta -= update

            global_error = self.__global_error__(theta_new=theta, theta_old=theta_old)

            if self.iteration % 50 == 0:
                print(f'Epoch: {self.iteration}, Error: {float(global_error)}.')

            self.iteration += 1

        print(f'Last epoch: {self.iteration-1}, Error: {float(global_error)}.')

        return self.theta_result, self.m_result, self.v_result, self.iteration
    

    
class NonlocalSolverMomentumAdam(_NonlocalSolverBase):
    def __init__(self, *args,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 **kw):
        super().__init__(*args, **kw)
        self.beta1, self.beta2 = map(DTYPE, betas)
        self._alpha_t = lambda t: jnp.where(t <= 1e-12, 1., jnp.sqrt(1 - self.beta2 ** (t / self.alpha)) / (1 - self.beta1 ** (t / self.alpha)))
        self._eps_t = lambda t: jnp.sqrt(1 - self.beta2 ** (t / self.alpha)) * DTYPE(1e-8)

    # ---------- hooks concretos ----------
    def _build_stats(self, y):
        interp = self._interp(y)
        g = jax.vmap(lambda t: self.dL(interp(t)) + 0.5 * self.lambda_ * interp(t))(self.t)
        m = _ema(self.beta1, g)
        v = _ema(self.beta2, g**2)
        v_sqrt  = jnp.sqrt(v)
        self._last_m = jnp.stack((self.t, m), axis=1)      
        self._last_v = jnp.stack((self.t, v), axis=1)  
        return (y, m, v_sqrt, jax.vmap(self._alpha_t)(self.t), jax.vmap(self._eps_t)(self.t))

    def _rhs(self, t, y_prev, idx, y_fix, m, v_sqrt, a_t, eps_t):
        y_val = interp1d(t, self.t, y_fix, method="cubic")
        denom = v_sqrt[idx] + eps_t[idx]
        return self.f(t, y_val) - a_t[idx] * (m[idx] / denom)
    