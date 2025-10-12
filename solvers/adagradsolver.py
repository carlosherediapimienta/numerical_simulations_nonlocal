import jax.numpy as jnp
from jax import lax
import jax
from .base.common import _NonlocalSolverBase, DTYPE, fixed_quad_jax
import numpy as np
from interpax import interp1d
from typing import Callable
from functools import partial

class AdaGrad:
    """
    Simple 1D AdaGrad optimizer (scalar parameter theta).

    Parameters
    ----------
    dL : Callable
        Gradient function dL(theta). Should return the derivative wrt theta.
    lr : float
        Base learning rate (alpha).
    epsilon : float
        Small constant added to the denominator for numerical stability.
    lr_decay : float
        If non-zero, applies a simple (1 / (1 + (iter-1)*lr)) decay factor
        to the update magnitude (note: uses `lr` in the formula as provided).
    lambda_l2 : float
        L2 regularization coefficient. If non-zero, adds (lambda/2)*theta to dL.
    epochs : int
        Maximum number of iterations.

    Attributes
    ----------
    accumulated_gradients : float
        Running sum of squared gradients G_t = Sum g_t^2.
    theta_result : list
        History of θ across iterations.
    accumulated_gradients_result : list
        History of accumulated gradients across iterations.
    global_error_tolerance : float
        Unused stopping criterion placeholder (kept for compatibility).
    """
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
        """Absolute difference between successive theta values."""
        diff = theta_new - theta_old
        return np.abs(diff)

    def solve(self, theta_initial):
        """
        Run AdaGrad for `epochs` steps starting from `theta_initial`.

        Notes
        -----
        - Uses scalar theta (no vector/matrix support here).
        - L2 term, if enabled, adds (lambda/2)*theta to the gradient.
        - If `lr_decay != 0`, scales the update by 1 / (1 + (iter-1)*lr).
        """
        theta = theta_initial
        while self.iteration <= self.epochs:
            
            # Keep history for analysis/plotting
            self.theta_result.append(theta)
            self.accumulated_gradients_result.append(self.accumulated_gradients)

            theta_old = theta
            dL_value = self.dL(theta)

            # Optional L2 contribution (as per user's original formula)
            if self.lambda_l2 != 0:
                dL_value += self.lambda_l2 / 2 * theta

            # Accumulate squared gradient (denominator for AdaGrad)
            self.accumulated_gradients += dL_value ** 2

            # AdaGrad step size scaling
            update = self.lr * dL_value / (np.sqrt(self.accumulated_gradients) + self.epsilon)

            # Optional learning-rate decay on the update magnitude
            if self.lr_decay != 0:
                theta -= 1/(1 + (self.iteration - 1) * self.lr) * update
            else:
                theta -= update
            
            # Simple convergence monitor (not used to stop)
            global_error = self.__global_error__(theta_new=theta, theta_old=theta_old)

            if self.iteration % 50 == 0:
                print(f'Epoch: {self.iteration}, Error: {global_error}.')

            self.iteration += 1

        print(f'Last epoch: {self.iteration}, Error: {global_error}.')

        return self.theta_result, self.accumulated_gradients_result, self.iteration
    
def chunked_vmap(fn, chunk_size: int):
    """
    Apply `fn` over the first axis of `x` in memory-friendly chunks.

    This reduces peak memory usage by splitting the batch dimension into
    chunks of size `chunk_size`, vmapping within each chunk, and then
    stitching the results back together.

    Requirements
    ------------
    - `fn` must accept a single element x_i and return y_i.
    - The leading dimension of `x` is treated as the batch dimension.
    """
    @partial(jax.jit, static_argnames=('chunk_size',))
    def _apply(x, chunk_size=chunk_size):
        N = x.shape[0]
        pad = (-N) % chunk_size  # number of items to pad to reach multiple
        
        # Pad along the leading dimension using the edge value to preserve shape
        pad_width = ((0, pad),) + ((0, 0),) * (x.ndim - 1)
        x_pad = jnp.pad(x, pad_width, mode='edge')

        # Reshape into (num_chunks, chunk_size, ...)
        x_mat = x_pad.reshape((-1, chunk_size) + x.shape[1:])  

        # Process each chunk: vmap over the chunk axis
        def per_chunk(x_chunk):
            return jax.vmap(fn)(x_chunk)

        # Map over chunks; result has shape (num_chunks, chunk_size, ...)
        y_mat = lax.map(per_chunk, x_mat) 

        # Undo the reshape and remove the padding
        y_pad = y_mat.reshape((x_pad.shape[0],) + y_mat.shape[2:])
        return y_pad[:N]

    return _apply
    

class NonlocalSolverAdaGrad(_NonlocalSolverBase):
    """
    Continuous-time AdaGrad-like nonlocal dynamics for a scalar parameter θ(t):

        dot theta(t) = f(t, theta(t)) - g(t) / ( sqrt{G(t+alpha)} + epsilon )

    with
        g(t)   = partial f(theta(t)) + 1/2 lambda theta(t)
        G(t+alpha) = (1/alpha) int_0^{t+alpha} [g(tau)]^2 dtau    <--- note the 1/alpha factor
        alpha(t)   = 1 / (1 + (lr_decay/alpha) · t)   (handled externally via step)

    Notes
    -----
    - Inherits time-stepping / relaxation scaffolding from `_NonlocalSolverBase`.
    - Builds a cubic interpolant over the current iterate y(t) to evaluate
      nonlocal terms at arbitrary tau during quadrature.
    - G(t+alpha) is computed by Gauss-Legendre quadrature via `fixed_quad_jax`.
    """
    def __init__(self, *args, lr_decay: float = 0.0, eps_base: float = 1e-8, **kw):
        super().__init__(*args, **kw)
        self.lr_decay = DTYPE(lr_decay)
        self.eps_base = DTYPE(eps_base)
        self.chunk_size = 512      # chunk size for memory-friendly vmaps
        self.nGL        = int(1e3) # Gauss–Legendre order for the integral

    def _build_stats(self, y):
        """
        Precompute:
          - g(t)      : local gradient term along the current path y(t)
          - G_sqrt(t) : sqrt of the cumulative integral G(t+alpha)
          - store (t, G(t+alpha)) for inspection/debugging

        Returns
        -------
        (y, g, G_sqrt) : tuple of arrays consumed by `_rhs`.
        """
        if self.verbose:
            jax.debug.print("¿NaNs in y?: {}", jnp.isnan(y).any())

        interp = self._interp(y)     # cubic interpolant y(τ) over the grid
        t_vec  = self.t
        lam    = 1. / self.alpha     # 1/alpha factor in the definition of G

        def g_fun(tau):
            # g(tau) = dL(y(tau)) + 1/2 lambda y(tau)
            return self.dL(interp(tau)) + 0.5 * self.lambda_ * interp(tau)
        
        @jax.jit
        def _G_single(t):
            """
            Compute G(t+alpha) for a single t using fixed-order GL quadrature.
            For t sim 0, return 0 to avoid degeneracy near the lower bound.
            """
            ub = t + self.alpha
            def compute(upper):
                integrand = lambda tau: g_fun(tau)**2
                # Integrate from a tiny positive lower bound to avoid issues at 0
                return lam * fixed_quad_jax(integrand, 1e-12, upper, self.nGL, verbose=self.verbose)
            # If t is extremely small, short-circuit to zero     
            return lax.cond(t < 1e-12, lambda _: DTYPE(0.), compute, ub)

        # Evaluate g(t) over the grid
        g = jax.vmap(g_fun)(t_vec)

        # Compute G(t+alpha) over the grid using chunked vmaps to reduce memory
        G = chunked_vmap(_G_single, self.chunk_size)(t_vec)
        G_sqrt = jnp.sqrt(G)

        if self.verbose:                             
            # Print first two and last entries as a quick sanity check
            jax.debug.print("g[0]={:.3e}, G[0]={:.3e}", g[0], G[0])
            jax.debug.print("g[1]={:.3e}, G[1]={:.3e}", g[1], G[1])
            jax.debug.print("g[-1]={:.3e}, G[-1]={:.3e}", g[-1], G[-1])

        # Keep (t, G) for external inspection if needed
        self._last_G = jnp.stack((t_vec, G), axis=1)
        return (y, g, G_sqrt)

    def _rhs(self, t, y_prev, idx, y_fix, g, G_sqrt):
        """
        Right-hand side for the explicit Euler step at time `t`.

        Parameters
        ----------
        t : DTYPE
            Current time.
        y_prev : DTYPE
            Previous value .
        idx : int
            Index of `t` on the time grid.
        y_fix : jnp.ndarray
            Current iterate y(t) .
        g : jnp.ndarray
            Samples of g(t) along the time grid.
        G_sqrt : jnp.ndarray
            Samples of sqrt(G(t+alpha)) along the time grid.

        Returns
        -------
        dy/dt : DTYPE
            Continuous-time AdaGrad-like update evaluated at t.
        """

        # Interpolate y at the exact time `t`
        y_val = interp1d(t, self.t, y_fix, method="cubic", extrap=True)

        # Denominator sqrt{G(t+alpha)} + eps for stability
        denom = G_sqrt[idx] + self.eps_base

        # Combine local dynamics f(t, y(t)) with normalized gradient term
        return self.f(t, y_val) - g[idx] / denom
    
