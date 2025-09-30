import jax.numpy as jnp
from .base.common import _NonlocalSolverBase, DTYPE, fixed_quad_jax
import numpy as np
from typing import Callable
from interpax import interp1d
from jax import lax

# Optional device info (kept as in the original). Raises a clear error if JAX
# is not present. Messages are left in Spanish per your original code.
try:
    import jax
    import jax.numpy as jnp
    device = jax.devices()[0]
    print(f"[Solver] Using JAX in: {device.device_kind} ({device.platform})")
except ImportError:
    raise ImportError("[Solver] JAX is not installed. Install it to use this optimizer.")

class RMSPropMomentum:
    """
    Scalar RMSProp optimizer for a single parameter theta.

    Parameters
    ----------
    dL : Callable
        Gradient function dL(theta). Must accept and return scalars.
    lr : float
        Base learning rate alpha.
    beta : float
        Exponential decay rate for the second-moment estimate (0 < beta < 1).
    epsilon : float
        Small constant for numerical stability in the denominator.
    weight_decay : float
        Multiplicative weight decay coefficient (decoupled style):
        theta ← (1 - weight_decay) theta - update.
    lambda_l2 : float
        L2 term coefficient. If non-zero, adds (lambda/2) theta to the gradient.
    epochs : int
        Number of iterations to run.

    Attributes
    ----------
    v : float
        Exponential moving average of squared gradients (second moment).
    theta_result : list[float]
        History of theta values across iterations.
    v_result : list[float]
        History of v values across iterations.
    iteration : int
        1-based iteration counter.
    global_error_tolerance : float
        Reserved for potential early stopping (not used here).
    """
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
        """Absolute difference |theta_new - theta_old|; useful as a progress metric."""
        diff = theta_new - theta_old
        return np.abs(diff)

    def solve(self, theta_initial):
        """
        Run RMSProp optimization starting at `theta_initial`.

        Notes
        -----
        - This implementation is scalar (1D) on purpose.
        - L2 regularization (if enabled) modifies the gradient as
          g ← g + (lambda/2) theta.
        - Weight decay (if enabled) applies a multiplicative shrink
          to theta before subtracting the update.
        """

        theta = theta_initial
        while self.iteration <= self.epochs:

            # Record history for analysis/plotting
            self.theta_result.append(theta)
            self.v_result.append(self.v)

            theta_old = theta
            dL_value = self.dL(theta)
            
            # Optional L2 contribution (as per original formula)
            if self.lambda_l2 != 0:
                dL_value  += self.lambda_l2 / 2 * theta

            # Exponential moving average of squared gradients
            self.v = self.beta * self.v + (1 - self.beta) * (dL_value ** 2)

            # RMSProp update (normalized by sqrt(v) + eps)            
            update = self.lr * dL_value / (np.sqrt(self.v) + self.epsilon)
            
            # Optional decoupled weight decay
            if self.weight_decay != 0:
                theta = (1 - self.weight_decay) * theta - update
            else:
                theta -= update
            
            # Progress metric (not used as a stopping condition)
            global_error = self.__global_error__(theta_new=theta, theta_old=theta_old)

            if self.iteration % 50 == 0:
                print(f'Epoch: {self.iteration}, Error: {global_error}.')

            self.iteration += 1

        print(f'Last epoch: {self.iteration}, Error: {global_error}.')

        return self.theta_result, self.v_result, self.iteration
    
    
class NonlocalSolverMomentumRMSProp(_NonlocalSolverBase):
    """
    Continuous-time RMSProp-like nonlocal dynamics for a scalar parameter θ(t):

        dot theta(t) = f(t, theta(t)) - g(t) / ( sqrt v(t) + ε )

    where
        g(t) = dL(theta(t)) + 1/2 lambda theta(t)
        v(t) = int^t_0 [(1-beta)/alpha] · exp(-(1-beta)(t-tau)/alpha) · [g(tau)]^2 dtau
              ~ computed via fixed-order Gauss-Legendre quadrature

    Notes
    -----
    - This class inherits the time grid, Euler stepping, relaxation loop,
      and error control from `_NonlocalSolverBase`.
    - The kernel for v(t) is an exponential EMA in continuous time with
      rate (1-beta)/alpha. We precompute v(t) on the solver time grid using
      `fixed_quad_jax`.
    """
    def __init__(self, *args, beta: float = 0.99, eps_base: float = 1e-8,**kw):
        super().__init__(*args, **kw)
        self.beta      = DTYPE(beta)
        self.eps_base  = DTYPE(eps_base)
        # lambda = (1−beta) / alpha: exponential kernel rate scaled by step size
        self.lam       = (1. - self.beta) / self.alpha

    def _build_stats(self, y):
        """
        Precompute quantities needed by the RHS:
          - g(t):   local gradient term along the current path y(t)
          - v(t):   EMA-like integral of g(t)^2 with exponential kernel
          - sqrt v(t):  square root used in the denominator
          - store (t, v(t)) for external inspection/debugging

        Returns
        -------
        (y, g, sqrt_v) : tuple consumed by `_rhs`.
        """
        if self.verbose:
            jax.debug.print("¿NaNs in y?: {}", jnp.isnan(y).any())

        interp = self._interp(y) # cubic interpolant y(tau) over the grid
        t_vec = self.t
        lam   = self.lam
        nGL   = int(1e3)         # Gauss–Legendre order for integration

        def g_fun(tau):
            # g(tau) = dL(y(tau)) + 1/2 lambda y(tau)
            return self.dL(interp(tau)) + 0.5*self.lambda_ * interp(tau)

        @jax.jit
        def _v_single(t):
            """
            Compute v(t) using a fixed-order Gauss-Legendre quadrature and
            an exponential kernel with rate `lam`. For very small t, return 0
            to avoid degeneracy near the lower limit.
            """
            def _compute(_t):
                ub = _t + self.alpha # integrate up to t + alpha 
                ker = lambda tau: jnp.exp(-lam * (ub - tau))
                f_v = lambda tau: lam * ker(tau) * g_fun(tau)**2
                v_k = fixed_quad_jax(f_v, 1e-12, ub, nGL, verbose=self.verbose)
                if self.verbose:
                    jax.debug.print("step values → nGL ={}  t={}  v={}", nGL, t, v_k)
                return v_k
            
            # Short-circuit to zero near t ~ 0
            return lax.cond(t < 1e-12, lambda _: DTYPE(0.), _compute, t)

        # Evaluate g(t) and v(t) across the entire time grid
        g     = jax.vmap(g_fun)(t_vec)
        v     = jax.vmap(_v_single)(t_vec)

        if self.verbose:                             
            # Print a small sample for sanity checks
            jax.debug.print("g[0]={:.3e}, v[0]={:.3e}", g[0], v[0])
            jax.debug.print("g[1]={:.3e}, v[1]={:.3e}", g[1], v[1])
            jax.debug.print("g[-1]={:.3e}, v[-1]={:.3e}", g[-1], v[-1])

        # Keep (t, v) for inspection if needed
        self._last_v = jnp.stack((t_vec, v), axis=1)

        # Return stats: current iterate y, g samples, and sqrt(v)
        return (y, g, jnp.sqrt(v))

    def _rhs(self, t, y_prev, idx, y_fix, g, v_sqrt):
        """
        Right-hand side for the explicit Euler step:

            dot theta(t) = f(t, theta(t)) - g(t) / ( sqrt v(t) + eps )

        Parameters
        ----------
        t : DTYPE
            Current time.
        y_prev : DTYPE
            Previous value.
        idx : int
            Index corresponding to `t` on the time grid.
        y_fix : jnp.ndarray
            Current iterate y(t).
        g : jnp.ndarray
            Samples of g(t) along the time grid.
        v_sqrt : jnp.ndarray
            Samples of sqrt v(t) along the time grid.

        Returns
        -------
        dy/dt : DTYPE
            The instantaneous rate used by the Euler integrator.
        """
        # Interpolate y at the exact continuous time `t`
        y_val = interp1d(t, self.t, y_fix, method="cubic", extrap=True)

        # Denominator sqrt v(t) + eps for stability
        denom = v_sqrt[idx] + self.eps_base

        # Combine local dynamics with normalized gradient term
        return self.f(t, y_val) - g[idx] / denom
    