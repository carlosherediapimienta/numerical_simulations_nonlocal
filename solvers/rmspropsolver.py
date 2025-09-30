import jax.numpy as jnp
from .base.common import _NonlocalSolverBase, DTYPE, fixed_quad_jax
import numpy as np
from typing import Callable
from interpax import interp1d
from jax import lax

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
    r"""
    θ̇(t) = - ḡ(t) / ( √v̄(t) + ε(t) )

    con  
      ḡ(t) = dL(θ(t)) + ½λ θ(t)  
      v̄(t) = ∫₀ᵗ (1-β)/α · e^{-(1-β)(t-tau)/α} · g(tau)² dtau  
            ≈ fixed_quad_jax(...)
    """
    def __init__(self, *args, beta: float = 0.99, eps_base: float = 1e-8,**kw):
        super().__init__(*args, **kw)
        self.beta      = DTYPE(beta)
        self.eps_base  = DTYPE(eps_base)
        self.lam       = (1. - self.beta) / self.alpha

    def _build_stats(self, y):
        if self.verbose:
            jax.debug.print("¿NaNs en y?: {}", jnp.isnan(y).any())

        interp = self._interp(y)
        t_vec = self.t
        lam   = self.lam
        nGL   = int(1e3)

        def g_fun(tau):
            return self.dL(interp(tau)) + 0.5*self.lambda_ * interp(tau)

        @jax.jit
        def _v_single(t):
            def _compute(_t):
                ub = _t + self.alpha
                ker = lambda tau: jnp.exp(-lam * (ub - tau))
                f_v = lambda tau: lam * ker(tau) * g_fun(tau)**2
                v_k = fixed_quad_jax(f_v, 1e-12, ub, nGL, verbose=self.verbose)
                if self.verbose:
                    jax.debug.print("step values → nGL ={}  t={}  v={}", nGL, t, v_k)
                return v_k
            return lax.cond(t < 1e-12, lambda _: DTYPE(0.), _compute, t)

        # vectorizado sobre la malla
        g     = jax.vmap(g_fun)(t_vec)
        v     = jax.vmap(_v_single)(t_vec)

        if self.verbose:                             
            # imprime 3 primeros y el último punto como muestra
            jax.debug.print("g[0]={:.3e}, v[0]={:.3e}", g[0], v[0])
            jax.debug.print("g[1]={:.3e}, v[1]={:.3e}", g[1], v[1])
            jax.debug.print("g[-1]={:.3e}, v[-1]={:.3e}", g[-1], v[-1])

        self._last_v = jnp.stack((t_vec, v), axis=1)

        # devuelve stats: y, g_vector, √v
        return (y, g, jnp.sqrt(v))

    def _rhs(self, t, y_prev, idx, y_fix, g, v_sqrt):
        """ θ̇_i(t) = - g_i(t) / ( √v_i(t) + ε ) """
        y_val = interp1d(t, self.t, y_fix, method="cubic", extrap=True)
        denom = v_sqrt[idx] + self.eps_base
        return self.f(t, y_val) - g[idx] / denom
    