"""
Common utilities shared by all non-local solvers.
This module is not meant to be instantiated directly.
"""
from __future__ import annotations
from functools import partial
from typing import Callable, Tuple, Any
import jax, jax.numpy as jnp
from jax import lax
from interpax import interp1d         
import numpy as _npx  

# Use 64-bit floats by default so numerical integration/iteration is stable
jax.config.update("jax_enable_x64", True)         
DTYPE = jnp.float64

# Small cache of Gauss–Legendre nodes/weights (NumPy computes them once).
# Precomputed for n in [2, 1000); fallback computes on demand if needed.
_GL_CACHE = {n: tuple(map(jnp.asarray, _npx.polynomial.legendre.leggauss(n))) for n in range(2, int(1e3))}

def fixed_quad_jax(fun, a, b, n, *, verbose=False, tol=1e-12):
    """
    Fixed-order Gauss–Legendre quadrature of order `n`.

    Notes
    -----
    - `a` and `b` can be scalars or broadcast-compatible vectors. Each pair
      (a[i], b[i]) defines one integral of `fun` over that interval.
    - The integral is evaluated with change of variables from [-1, 1] to [a, b].
    - If `b < a`, the sign is flipped automatically.
    - Very short intervals (|b - a| < tol) are treated as zero to avoid
      numerical noise.
    - Vectorization: evaluation points are fed through `jax.vmap(fun)`.

    Parameters
    ----------
    fun : Callable
        Integrand; must map array-like `t` to values broadcastable under sum.
    a, b : float or array-like
        Integration bounds (broadcast-compatible).
    n : int
        Quadrature order (nodes/weights). Cached for n < 1000.
    verbose : bool
        If True, prints intermediate results via `jax.debug.print`.
    tol : float
        Threshold below which intervals are treated as zero-length.

    Returns
    -------
    res : jnp.ndarray
        Integral(s) with shape compatible with broadcasting of `a` and `b`.
        If `a` and `b` are scalars, returns a scalar.
    """
    try:
        xg, wg = _GL_CACHE[n]
    except KeyError:
        # Fallback: compute with NumPy (outside JIT) and wrap as JAX arrays
        xg, wg = map(jnp.asarray, _npx.polynomial.legendre.leggauss(n))

    # Normalize shapes for vmapping over intervals
    a_arr = jnp.reshape(a, (-1,))          # ()  -> (1,)
    b_arr = jnp.reshape(b, (-1,))          # idem

    def _int(lo, hi):
         # Ensure correct orientation and keep track of sign if bounds reversed
        sign = jnp.where(hi < lo, -1., 1.)
        lo,  hi = jnp.minimum(lo, hi), jnp.maximum(lo, hi)

        # Affine map from [-1, 1] to [lo, hi]
        mid  = 0.5 * (hi + lo)
        half = 0.5 * (hi - lo)
        pts  = mid + half * xg

        # Evaluate integrand at quadrature nodes (vectorized)
        vals = jax.vmap(fun)(pts)

        # Weighted sum with Jacobian factor
        res  = sign * half * jnp.sum(wg * vals, axis=0)
        if verbose:                       
            jax.debug.print("fixed_quad: a={} b={}  res={}", a, b, res)

        # Treat near-zero-length intervals as zero to avoid spurious noise
        return jnp.where(jnp.abs(hi-lo) < tol, 0., res)
    
    res = jax.vmap(_int)(a_arr, b_arr)
    return res[0] if jnp.ndim(a) == 0 else res

class _NonlocalSolverBase:
    """
    Shared boilerplate for non-local ODE-like solvers:
      - time grid construction
      - explicit Euler time-stepping via `lax.scan` (single JIT compile)
      - fixed-point relaxation with smoothing
      - global error monitoring and tolerance/iteration controls

    Subclasses MUST implement:
      * _build_stats(y) -> Tuple[Any, ...]
          Precompute any structures ("stats") needed to evaluate the RHS
          at all times, given the current iterate y(t).

      * _rhs(t, y_prev, idx, *stats) -> dy/dt
          Right-hand side function to advance one Euler step.
          `idx` is the integer time-index (0-based) corresponding to `t`.

    Attributes (key ones)
    ---------------------
    f : Callable
        Local RHS f(t, y) used for the initial (no-integral) solution.
    dL : Callable
        Additional functional/derivative used by specific non-local models.
    t : jnp.ndarray
        Time grid [t0, tf) with spacing `alpha` (tf is excluded).
    y0 : DTYPE
        Initial condition (scalar).
    alpha : DTYPE
        Time step size.
    lambda_ : DTYPE
        Optional model parameter exposed to subclasses.
    smoothing : DTYPE
        Current relaxation factor s in y_next = s*y_new + (1-s)*y_cur.
    global_tol : float
        Convergence tolerance for the global error metric.
    max_iteration : int
        Safety cap on fixed-point iterations.
    verbose : bool
        If True, prints progress every 20 iterations.
    """
    def __init__(self,
                 f: Callable,
                 dL: Callable,
                 t_span: Tuple[float, float],
                 y0,
                 alpha: float,
                 lambda_: float = 0.,
                 verbose: bool = False):

        # JIT f and dL unless they are already lowered/compiled
        self.f  = jax.jit(f)  if not hasattr(f, "lower") else f
        self.dL = jax.jit(dL) if not hasattr(dL, "lower") else dL

        # Scalar initial condition (force dtype/shape)
        self.y0      = jnp.asarray(y0, dtype=DTYPE).ravel()[0]
        self.alpha   = DTYPE(alpha)
        self.lambda_ = DTYPE(lambda_)
        self.verbose = verbose

        self.t0, self.tf = t_span
         # Time grid is [t0, tf) with uniform spacing alpha (tf not included)
        self.t = jnp.arange(self.t0, self.tf, self.alpha, dtype=DTYPE)

        # ---------- relaxation (under-relaxed fixed-point iteration) ----
        self.smoothing     = DTYPE(0.5)    # starting smoothing factor
        self.smooth_max    = DTYPE(0.9999) # hard cap to avoid overshoot
        # Monotone schedule of candidate smoothing increases
        self.increments    = jnp.linspace(self.smoothing,
                                          self.smooth_max,
                                          1_000, dtype=DTYPE)
        self._max_inc_hit  = False         # flag once the max is reached

        # ---------- stopping criteria / safeguards ----------------------
        self.global_tol    = 1e-4
        self.max_iteration = int(3e3)

    # ---------- helpers -------------------------------------------------
    def _interp(self, y: jnp.ndarray):
        """Cubic interpolator over the current time grid.
        Not jitted on purpose to avoid recompilations when shapes change."""
        return lambda t: interp1d(t, self.t, y, method="cubic", extrap=True)

    # ---------------------------------------------------------------
    @staticmethod
    @partial(jax.jit, static_argnums=(3,)) 
    def _integrate(alpha: DTYPE,
                   y0: DTYPE,
                   t_vec: jnp.ndarray,
                   rhs: Callable[[DTYPE, DTYPE, jnp.int32, Tuple[Any, ...]],DTYPE],
                   stats: Tuple[Any, ...]) -> jnp.ndarray:
        """
        Explicit Euler integrator over `t_vec` using a single `lax.scan`.
        Compiled once (JIT) for the whole solve to minimize overhead.

        Parameters
        ----------
        alpha : DTYPE
            Time step size.
        y0 : DTYPE
            Initial value at t_vec[0].
        t_vec : jnp.ndarray
            Monotone time grid.
        rhs : Callable
            Function (t, y_prev, idx, *stats) -> dy/dt.
        stats : tuple
            Precomputed data built from the current iterate y(t).

        Returns
        -------
        y_hist : jnp.ndarray
            Values of y at all points in t_vec (same length as t_vec).
        """
        idxs  = jnp.arange(len(t_vec), dtype=jnp.int32)
        ts_id = jnp.stack((t_vec[:-1], idxs[:-1]), axis=1)

        def step(y_prev, args):
            t, idx = args
            idx = idx.astype(jnp.int32)  # defensive cast inside JIT
            y_next = y_prev + alpha * rhs(t, y_prev, idx, *stats)
            return y_next, y_next 

        _, hist = jax.lax.scan(step, y0, ts_id)
        return jnp.concatenate([jnp.array([y0], dtype=DTYPE), hist])

    # ------------- error & smoothing -----------------------------------
    @staticmethod
    @jax.jit
    def _global_error(a: jnp.ndarray, b: jnp.ndarray):
        """Global error metric: Euclidean norm over the full time series."""
        return jnp.linalg.norm(a - b)

    @staticmethod
    @jax.jit
    def _mix(s: DTYPE, a: jnp.ndarray, b: jnp.ndarray):
        """Convex combination for relaxation: s*a + (1-s)*b."""
        return s * a + (1. - s) * b

    # -------- hooks que las subclases deben rellenar --------------------
    def _build_stats(self, y: jnp.ndarray) -> Tuple[Any, ...]:
        """Prepare any per-iteration precomputations consumed by `_rhs`."""
        raise NotImplementedError

    def _rhs(self, t, y_prev, idx, *stats):
        """Right-hand side of the (non-local) ODE at time `t` and index `idx`."""
        raise NotImplementedError

    # -------------------------------------------------------------------
    def _step(self, y: jnp.ndarray) -> jnp.ndarray:
        """Run one full pass of the explicit Euler integrator using current stats."""
        stats = self._build_stats(y)
        return self._integrate(self.alpha, self.y0, self.t,
                               self._rhs, stats)

    # -------------------------------------------------------------------
    def solve(self):
        """
        Fixed-point outer loop with under-relaxation.

        Algorithm sketch
        ----------------
        1) Build a purely local baseline solution (no integral term)
           by integrating y' = f(t, y).
        2) Given the current iterate y_cur, build stats and integrate the
           full non-local RHS once to get y_new.
        3) Relax: y_relax = mix(smoothing, y_cur, y_new).
        4) Repeat until the global error ||y_relax - y_new|| < global_tol
           or safety limits are hit. The smoothing factor can increase
           when the error rises (simple backoff strategy).
        """
        self.iteration = 0

        # Baseline solution without the non-local term
        y_cur = self._integrate(self.alpha, self.y0, self.t,
                                lambda t, y, idx, *_: self.f(t, y), ())
        y_new = self._step(y_cur)
        err   = self._global_error(y_cur, y_new)
        if self.verbose:
            print(f"Iter {self.iteration} – err {err}")

        last = err
        while err > self.global_tol:
            # Under-relaxed update
            y_relax = self._mix(self.smoothing, y_cur, y_new)

            # Re-integrate with stats from the relaxed iterate
            y_new   = self._step(y_relax)
            err     = self._global_error(y_relax, y_new)

            # ---------- safety / control logic ----------
            if jnp.isnan(err) or jnp.isinf(err):
                print("Divergence (NaN / Inf). Abort.")
                break
            
            # If error worsens, increase smoothing factor (toward 1)
            if err > last:                          # subir smoothing
                if self._max_inc_hit:
                    print("Maximum smoothing achieved. Stop.")
                    break
                try:
                    nxt = self.increments[
                        jnp.searchsorted(self.increments,
                                         self.smoothing, side="right")]
                except IndexError:
                    nxt = self.smooth_max
                    self._max_inc_hit = True
                self.smoothing = jnp.minimum(self.smooth_max, nxt)
            last  = err
            y_cur = y_relax
            self.iteration += 1

            if self.iteration % 20 == 0:
                print(f"Iter {self.iteration} – err {err}")

            if self.iteration >= self.max_iteration:
                print("Max iterations. Stop.")
                break

        print(f"Last iter {self.iteration} – err {err}")
        self.y            = y_new
        self.global_error = err
        return self.t, y_new
