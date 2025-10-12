import numpy as np
from typing import Callable
from typing import Tuple
from jax import lax
import jax.numpy as jnp
from interpax import interp1d
from .base.common import _NonlocalSolverBase, DTYPE, fixed_quad_jax

# Device info (kept as in your original). If JAX is missing, raise a clear error.
try:
    import jax
    import jax.numpy as jnp
    device = jax.devices()[0]
    print(f"[Solver] Using JAX in: {device.device_kind} ({device.platform})")
except ImportError:
    raise ImportError("[Solver] JAX is not installed. Install it to use this optimizer.")

class AdamMomentum:
    """
    Scalar Adam optimizer with weight decay and optional L2 term.

    Parameters
    ----------
    dL : Callable
        Gradient function dL(theta). If not already lowered, it will be JIT-compiled.
    lr : float
        Base learning rate (alpha).
    beta1 : float
        Exponential decay rate for the first moment (0 < beta1 < 1).
    beta2 : float
        Exponential decay rate for the second moment (0 < beta2 < 1).
    epsilon : float
        Small constant in the denominator for numerical stability.
    weight_decay : float
        Decoupled weight decay coefficient: theta ← (1 - wd)theta - update.
    lambda_l2 : float
        L2 coefficient; if non-zero, adds (lambda/2)theta to the gradient.
    epochs : int
        Number of iterations to run.

    Attributes
    ----------
    m, v : float
        First/second moment accumulators.
    iteration : int
        1-based iteration counter.
    theta_result, m_result, v_result : list[float]
        Per-epoch histories for analysis/plotting.
    """
    def __init__(self, dL: Callable, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8, 
                 weight_decay: float = 0, lambda_l2: float = 0, epochs: int = 1000):

        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.lambda_l2 = lambda_l2

        # JIT dL unless it is already a lowered JAX function
        if not hasattr(dL, 'lower'):
            self.dL = jax.jit(dL)
        else:
            self.dL = dL
        self.epochs = epochs
        self.global_error_tolerance = 1e-5 # placeholder; not used to stop
        self.reset_state()

    def reset_state(self):
        """Reset optimizer state and histories."""
        self.m = 0.0
        self.v = 0.0
        self.iteration = 1
        self.theta_result = []
        self.m_result = []
        self.v_result = []

    @staticmethod
    @jax.jit
    def __global_error__(theta_new, theta_old):
        """Absolute difference |theta_new - theta_old|; handy as a progress metric."""
        import jax.numpy as jnp
        diff = theta_new - theta_old
        return jnp.abs(diff)

    def solve(self, theta_initial):
        """
        Run Adam for `epochs` steps starting from `theta_initial`.

        Notes
        -----
        - Scalar implementation (1D).
        - Bias-corrected moments m, v are used.
        - If `weight_decay != 0`, apply decoupled weight decay before subtracting `update`.
        """
        self.reset_state()
        theta = jnp.array(theta_initial)

        while self.iteration <= self.epochs:
            # Log histories 
            self.theta_result.append(float(theta))
            self.m_result.append(float(self.m))
            self.v_result.append(float(self.v))

            theta_old = theta
            dL_value = self.dL(theta)

            # Optional L2 (as per your original formula)
            if self.lambda_l2 != 0:
                dL_value += self.lambda_l2 / 2 * theta

            # Adam moment updates
            self.m = self.beta1 * self.m + (1 - self.beta1) * dL_value
            self.v = self.beta2 * self.v + (1 - self.beta2) * (dL_value ** 2)

            # Bias corrections
            m_hat = self.m / (1 - self.beta1 ** self.iteration)
            v_hat = self.v / (1 - self.beta2 ** self.iteration)

            # Adam update
            update = self.lr * m_hat / (jnp.sqrt(v_hat) + self.epsilon)

            # Decoupled weight decay (if enabled)
            if self.weight_decay != 0:
                theta = (1 - self.weight_decay) * theta - update
            else:
                theta -= update

            # Progress metric
            global_error = self.__global_error__(theta_new=theta, theta_old=theta_old)

            if self.iteration % 50 == 0:
                print(f'Epoch: {self.iteration}, Error: {float(global_error)}.')

            self.iteration += 1

        print(f'Last epoch: {self.iteration-1}, Error: {float(global_error)}.')

        return self.theta_result, self.m_result, self.v_result, self.iteration
    

    
class NonlocalSolverMomentumAdam(_NonlocalSolverBase):
    """
    Continuous-time Adam-like nonlocal dynamics for a scalar parameter θ(t):

        dot theta(t) = f(t, theta(t)) - alpha(t) hatm(t) / ( sqrt{hatv(t) + eps(t)} )

    with  hatm, hatv defined by exponential kernels K_1, K_2
           K_a(t) = (1 - beta_a)/alpha exp(-(1 - β_a) t / alpha),   a in {1,2}.

    Notes
    -----
    - Inherits grid construction, Euler stepping, relaxation, and error control
      from `_NonlocalSolverBase`.
    - alpha(t) and eps(t) are bias-correction factors (analogs to discrete Adam).
    - Precomputed weight matrices `_exp1/_exp2` are prepared from the time grid;
      they are not used in the current quadrature-based implementation but kept
      for potential fast-matrix versions.
    """
    def __init__(self, *args,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps_base: float = 1e-8,
                 **kw):
        super().__init__(*args, **kw)
        self.beta1, self.beta2 = map(DTYPE, betas)
        self.eps_base = DTYPE(eps_base)

        # Bias-correction factors (continuous-time analogs)
        # alpha(t): scales the first moment; eps(t): scales the denominator
        self._alpha_t = lambda t: jnp.where(
            t <= 1e-12,
            1.,
            jnp.sqrt(1. - self.beta2 ** (t / self.alpha)) / (1. - self.beta1 ** (t / self.alpha))
        )
        self._eps_t = lambda t: jnp.where(
            t <= 1e-12,
            self.eps_base,                                   
            self.eps_base * jnp.sqrt(1. - self.beta2 ** (t / self.alpha))
        )

        # --- Precompute weight matrices on the current time grid -----------
        dt        = self.t[:, None] - self.t[None, :]    # t_i - t_j
        self._tri = dt >= 0                              # causal mask 

        self.lam1 = (1. - self.beta1) / self.alpha       # lambda_1 = (1-beta_1)/alpha
        self.lam2 = (1. - self.beta2) / self.alpha       # lambda_2 = (1-beta_2)/alpha

         # Exponential kernels e^{-lambda (t_i - t_j)} (kept for potential matrix path)
        self._exp1 = jnp.exp(-self.lam1 * dt)
        self._exp2 = jnp.exp(-self.lam2 * dt)

    # ------------------------------------------------------------------
    def _build_stats(self, y):
        """
        Precompute along the current iterate y(t):
          - m(t): exponential moving average of g(t)
          - v(t): exponential moving average of g(t)^2
          - sqrt(v(t)), alpha(t), eps(t): quantities needed by the RHS
          - Save (t, m) and (t, v) for inspection/debugging

        Returns
        -------
        (y, m, v_sqrt, a_t, eps_t)
        """
        if self.verbose:
            jax.debug.print("¿NaNs in y?: {}", jnp.isnan(y).any())

        interp = self._interp(y)
        # Local copies of rates for the integrals (could reuse self.lam1/lam2)
        lam1 = (1. - self.beta1) / self.alpha
        lam2 = (1. - self.beta2) / self.alpha
        nGL  = int(1e3)  # Gauss–Legendre order

        def g_fun(tau):
            # g(tau) = dL(y(tau)) + 1/2 lambda y(tau)
            return self.dL(interp(tau)) + 0.5*self.lambda_ * interp(tau)
        
        @jax.jit
        def _moments_single(t,lam):
            """
            Compute (m(t), v(t)) for a single time t via fixed-order GL quadrature.
            For very small t, short-circuit to (0, 0) to avoid boundary issues.
            """
            def _compute(_t):
                ker = lambda tau: jnp.exp(-lam * (_t - tau))
                f_m = lambda tau: lam * ker(tau) * g_fun(tau)
                f_v = lambda tau: lam * ker(tau) * g_fun(tau)**2
                m_k = fixed_quad_jax(f_m, 1e-12, _t, nGL, verbose=self.verbose)
                v_k = fixed_quad_jax(f_v, 1e-12, _t, nGL, verbose=self.verbose)
                if self.verbose:
                    jax.debug.print("step values → nGL ={}  t={}  m={}  v={}", nGL, t, m_k, v_k)
                return m_k, v_k
            return lax.cond(t < 1e-12, lambda _: (DTYPE(0.), DTYPE(0.)), _compute, t)

        # Vectorized evaluation over the solver grid
        m = jax.vmap(lambda t: _moments_single(t,lam1)[0])(self.t)
        v = jax.vmap(lambda t: _moments_single(t,lam2)[1])(self.t)

        if self.verbose:                             
            # Print a small sample for sanity checks
            jax.debug.print("m[0]={:.3e}, v[0]={:.3e}", m[0], v[0])
            jax.debug.print("m[1]={:.3e}, v[1]={:.3e}", m[1], v[1])
            jax.debug.print("m[-1]={:.3e}, v[-1]={:.3e}", m[-1], v[-1])

        # Save timeseries for inspection/plotting
        self._last_m = jnp.stack((self.t, m), axis=1)
        self._last_v = jnp.stack((self.t, v), axis=1)

        v_sqrt = jnp.sqrt(v)
        a_t    = jax.vmap(self._alpha_t)(self.t)     # bias-correction factor
        eps_t  = jax.vmap(self._eps_t)(self.t)       # scaled epsilon

        return (y, m, v_sqrt, a_t, eps_t)

    # ------------------------------------------------------------------
    def _rhs(self, t, y_prev, idx, y_fix, m, v_sqrt, a_t, eps_t):
        """
        Right-hand side for the explicit Euler step:

            dot theta(t) = f(t, theta(t)) - alpha(t) m(t) / ( sqrt{v(t)} + eps(t) )

        Parameters
        ----------
        t : DTYPE
            Current time.
        y_prev : DTYPE
            Previous value .
        idx : int
            Index of `t` on the solver time grid.
        y_fix : jnp.ndarray
            Current iterate y(t) .
        m : jnp.ndarray
            Samples of the first moment along the grid.
        v_sqrt : jnp.ndarray
            Samples of sqrt(second moment) along the grid.
        a_t : jnp.ndarray
            Bias-correction factor alpha(t) along the grid.
        eps_t : jnp.ndarray
            Scaled eps(t) along the grid.

        Returns
        -------
        dy/dt : DTYPE
            Instantaneous rate used by the Euler integrator.
        """
        # Interpolate y at the exact time t
        y_val = interp1d(t, self.t, y_fix, method="cubic", extrap=True)

        # Denominator \sart{v(t)} + eps(t) for stability
        denom = v_sqrt[idx] + eps_t[idx]

        # Combine local dynamics with normalized moment term
        return self.f(t, y_val) - a_t[idx] * (m[idx] / denom)