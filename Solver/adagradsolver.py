import jax.numpy as jnp
from jax import lax
import jax
from .base.common import _NonlocalSolverBase, DTYPE, fixed_quad_jax
import numpy as np
from interpax import interp1d
from typing import Callable
from functools import partial

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
    
def chunked_vmap(fn, chunk_size: int):
    """
    Aplica fn sobre el primer eje de x en bloques de chunk_size, reduciendo
    pico de memoria. fn debe aceptar (x_i) y devolver y_i.
    """
    @partial(jax.jit, static_argnames=('chunk_size',))
    def _apply(x, chunk_size=chunk_size):
        N = x.shape[0]
        pad = (-N) % chunk_size
        # Relleno con el último valor para igualar tamaños
        pad_width = ((0, pad),) + ((0, 0),) * (x.ndim - 1)
        x_pad = jnp.pad(x, pad_width, mode='edge')
        x_mat = x_pad.reshape((-1, chunk_size) + x.shape[1:])  # (num_chunks, chunk, ...)

        # Cuerpo por chunk: vmap sobre el chunk
        def per_chunk(x_chunk):
            return jax.vmap(fn)(x_chunk)

        y_mat = lax.map(per_chunk, x_mat)  # (num_chunks, chunk, ...)
        # Deshacer reshape y quitar padding
        y_pad = y_mat.reshape((x_pad.shape[0],) + y_mat.shape[2:])
        return y_pad[:N]

    return _apply
    

class NonlocalSolverAdaGrad(_NonlocalSolverBase):
    r"""
    θ̇(t) = - α(t) · g(t) / ( √G(t+α) + ε )

    con
      g(t)   = ∂f(θ(t)) + ½λ θ(t)
      G(t+α) = (1/α) ∫₀^{t+α} g(tau)² dtau      ← factor 1/α
      α(t)   = 1 / (1 + (lr_decay/α)·t)
    """
    def __init__(self, *args, lr_decay: float = 0.0, eps_base: float = 1e-8, **kw):
        super().__init__(*args, **kw)
        self.lr_decay = DTYPE(lr_decay)
        self.eps_base = DTYPE(eps_base)
        self.chunk_size = 512
        self.nGL        = int(1e3)

    def _build_stats(self, y):
        if self.verbose:
            jax.debug.print("¿NaNs en y?: {}", jnp.isnan(y).any())

        interp = self._interp(y)
        t_vec  = self.t
        lam    = 1. / self.alpha

        def g_fun(tau):
            return self.dL(interp(tau)) + 0.5 * self.lambda_ * interp(tau)
        
        @jax.jit
        def _G_single(t):
            ub = t + self.alpha
            def compute(upper):
                integrand = lambda tau: g_fun(tau)**2
                return lam * fixed_quad_jax(integrand, 1e-12, upper, self.nGL, verbose=self.verbose)     
            return lax.cond(t < 1e-12, lambda _: DTYPE(0.), compute, ub)


        g = jax.vmap(g_fun)(t_vec)
        G = chunked_vmap(_G_single, self.chunk_size)(t_vec)
        G_sqrt = jnp.sqrt(G)

        if self.verbose:                             
            # imprime 3 primeros y el último punto como muestra
            jax.debug.print("g[0]={:.3e}, G[0]={:.3e}", g[0], G[0])
            jax.debug.print("g[1]={:.3e}, G[1]={:.3e}", g[1], G[1])
            jax.debug.print("g[-1]={:.3e}, G[-1]={:.3e}", g[-1], G[-1])

        self._last_G = jnp.stack((t_vec, G), axis=1)
        return (y, g, G_sqrt)

    def _rhs(self, t, y_prev, idx, y_fix, g, G_sqrt):
        y_val = interp1d(t, self.t, y_fix, method="cubic", extrap=True)
        denom = G_sqrt[idx] + self.eps_base
        return self.f(t, y_val) - g[idx] / denom
    
