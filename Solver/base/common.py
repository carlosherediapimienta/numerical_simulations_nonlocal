"""
Código común a todos los solvers no-locales.
No se debe instanciar directamente.
"""
from __future__ import annotations
from typing import Callable, Tuple, Any
import jax
import jax.numpy as jnp
from interpax import interp1d          # pip install interpax

jax.config.update("jax_enable_x64", True)
DTYPE = jnp.float64

@jax.jit
def _ema(beta: DTYPE, seq: jnp.ndarray) -> jnp.ndarray:
    """Exponential Moving Average vectorizado (una única compilación)."""
    def body(prev, x):
        new = beta * prev + (1. - beta) * x
        return new, new
    _, out = jax.lax.scan(body, DTYPE(0.), seq)
    return out


class _NonlocalSolverBase:
    """
    Boiler-plate compartido: mallado temporal, Euler+scan (jit único),
    relajación con smoothing, control de error y tolerancia.

    Las subclases **solo** implementan:
      * _build_stats(y)         -> Tuple[Any, ...]
      * _rhs(t, y_prev, idx, *stats) -> dy/dt
    """
    # -----------------------------------------------------------------
    def __init__(self,
                 f: Callable,
                 dL: Callable,
                 t_span: Tuple[float, float],
                 y0,
                 alpha: float,
                 lambda_: float = 0.,
                 verbose: bool = True):

        self.f  = jax.jit(f)  if not hasattr(f, "lower") else f
        self.dL = jax.jit(dL) if not hasattr(dL, "lower") else dL

        self.y0      = jnp.asarray(y0, dtype=DTYPE).ravel()[0]
        self.alpha   = DTYPE(alpha)
        self.lambda_ = DTYPE(lambda_)
        self.verbose = verbose

        self.t0, self.tf = t_span
        self.t = jnp.arange(self.t0, self.tf, self.alpha, dtype=DTYPE)

        # Relaxation params
        self.smoothing     = DTYPE(0.5)
        self.smooth_max    = DTYPE(0.9999)
        self.increments    = jnp.linspace(self.smoothing,
                                          self.smooth_max,
                                          1_000, dtype=DTYPE)
        self._max_inc_hit  = False

        self.global_tol    = 1e-4
        self.max_iteration = int(1e10)

    # ---------- helpers -------------------------------------------------
    def _interp(self, y: jnp.ndarray):
        """Interpolador cúbico – no se jittea (evita recompilaciones)."""
        return lambda τ: interp1d(τ, self.t, y, method="cubic")

    # ---------------------------------------------------------------
    @staticmethod
    @jax.jit
    def _integrate(alpha: DTYPE,
                   y0: DTYPE,
                   t_vec: jnp.ndarray,
                   rhs: Callable[[DTYPE, DTYPE, jnp.int32, Tuple[Any, ...]],
                                 DTYPE],
                   stats: Tuple[Any, ...]) -> jnp.ndarray:
        """Euler + lax.scan; una sola compilación para todo el proceso."""
        idxs  = jnp.arange(len(t_vec), dtype=jnp.int32)
        ts_id = jnp.stack((t_vec[:-1], idxs[:-1]), axis=1)

        def step(y_prev, args):
            t, i = args
            dy = rhs(t, y_prev, i.astype(jnp.int32), *stats)
            return y_prev + alpha * dy, y_prev

        _, hist = jax.lax.scan(step, y0, ts_id)
        return jnp.concatenate([jnp.array([y0], dtype=DTYPE), hist])

    # ------------- error & smoothing -----------------------------------
    @staticmethod
    @jax.jit
    def _global_error(a: jnp.ndarray, b: jnp.ndarray):
        return jnp.linalg.norm(a - b)

    @staticmethod
    @jax.jit
    def _mix(s: DTYPE, a: jnp.ndarray, b: jnp.ndarray):
        """Next y with smoothing factor."""
        return s * a + (1. - s) * b

    # -------- hooks que las subclases deben rellenar --------------------
    def _build_stats(self, y: jnp.ndarray) -> Tuple[Any, ...]:
        raise NotImplementedError

    def _rhs(self, t, y_prev, idx, *stats):
        raise NotImplementedError

    # -------------------------------------------------------------------
    def _step(self, y: jnp.ndarray) -> jnp.ndarray:
        stats = self._build_stats(y)
        return self._integrate(self.alpha, self.y0, self.t,
                               self._rhs, stats)

    # -------------------------------------------------------------------
    def solve(self):
        self.iteration = 0

        # solución sin término integral
        y_cur = self._integrate(self.alpha, self.y0, self.t,
                                lambda t, y, idx, *_: self.f(t, y), ())
        y_new = self._step(y_cur)
        err   = self._global_error(y_cur, y_new)
        if self.verbose:
            print(f"Iter {self.iteration} – err {err}")

        last = err
        while err > self.global_tol:
            y_relax = self._mix(self.smoothing, y_cur, y_new)
            y_new   = self._step(y_relax)
            err     = self._global_error(y_relax, y_new)

            # ---------- controles ----------
            if jnp.isnan(err) or jnp.isinf(err):
                print("Divergencia (NaN / Inf). Abort.")
                break

            if err > last:                          # subir smoothing
                if self._max_inc_hit:
                    print("Máx smoothing alcanzado. Stop.")
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

            if self.iteration % 20 == 0 and self.verbose:
                print(f"Iter {self.iteration} – err {err}")

            if self.iteration >= self.max_iteration:
                print("Máx iteraciones. Stop.")
                break

        print(f"Last iter {self.iteration} – err {err}")
        self.y            = y_new
        self.global_error = err
        return self.t, y_new
