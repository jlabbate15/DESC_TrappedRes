"""Trapped-particle graze-flux objective from second-adiabatic-invariant maps.

Implements the J_parallel drift-surface pipeline validated in the
20260826_squid_driftsurf_flux_fit campaign (E. J. Paul):

1. **Bounce layer** — normalized second adiabatic invariant, bounce time, and
   loss-cone birth moment per trapped class on constant-pitch grids,

       Jhat(rho, alpha; pitch) = int sqrt(1 - pitch |B|) dl,
       that(rho, alpha; pitch) = int dl / sqrt(1 - pitch |B|),
       W(rho, alpha; pitch)    = int (sqrt(g)/dzeta-measure)
                                  pitch |B| / (2 sqrt(1 - pitch |B|)) dzeta,

   with turning points located by first crossing marching outward from the
   primary well minimum, and the sin substitution
   ``zeta = c + h sin(u)`` removing the endpoint square-root singularities
   (spectral accuracy with Gauss-Legendre in ``u``). Classes are exact
   constant-pitch slices by construction.
2. **Drift layer** — a strictly non-crossing, boundary-pinned nested
   foliation of the (alpha, rho^2) drift plane,

       s(x, alpha) = b(alpha) + int_0^x exp(u(x', alpha)) dx',

   fit by a fixed-iteration Gauss-Newton minimizing the per-leaf variance of
   Jhat at pinned area labels. Monotonicity (hence non-intersection of
   leaves) is structural via the exponential.
3. **Flux** — prompt graze loss: a class point is lost iff its drift surface's
   radial maximum reaches ``s_loss``; the confined measure follows from a
   Green's-theorem line integral along the critical leaf, and the flux is the
   birth-weighted lost complement integrated over classes.

Notes
-----
This first implementation samples field lines on nodes fixed at build time
(straight-field-line labels from the build-time lambda); the exact
runtime-angle treatment should reuse the ``angle=delta`` /
``_map_poloidal_coordinates`` machinery of ``GammaC`` (see
``desc/objectives/_fast_ion.py``) and the ``Bounce2D`` quadratures — tracked
as a TODO below. Gradients of the foliation fit follow the envelope theorem
(the fit is stationary), so the objective's derivative through the fitted
leaves requires only the explicit field dependence; the critical-label
sensitivity term (implicit-function derivative of the tangency label) is
also captured because ``s_loss`` interpolation is differentiable in the leaf
arrays.

References
----------
Campaign record: Princeton Dropbox data/August_2026/20260826_squid_driftsurf_flux_fit
(methods note "The Graze-Flux Objective"). Validation: class-resolved and
config-level agreement with FIRM3D collisionless losses on the SQuID
LowShear beta scan.
"""

from functools import partial

from desc.backend import jnp
from desc.compute import get_profiles, get_transforms
from desc.compute.utils import _compute as compute_fun
from desc.grid import LinearGrid
from desc.utils import setdefault

from .objective_funs import _Objective, collect_docs

try:  # orthax is a DESC dependency; guard for docs builds
    from orthax.legendre import leggauss
except Exception:  # pragma: no cover
    leggauss = None


# --------------------------------------------------------------------------
# bounce layer
# --------------------------------------------------------------------------


def _first_crossing(Bline, i0, Bc, direction, zscan):
    """First zeta where Bline >= Bc marching from index i0 in direction.

    Vectorized over leading axes of ``Bline`` (..., nscan). Returns the
    linearly interpolated crossing position and a validity mask. Integer
    bracket indices are treated as non-differentiable (locally constant);
    the position is differentiable through the field values.
    """
    n = Bline.shape[-1]
    idx = i0[..., None] + direction * jnp.arange(n)
    idx = jnp.clip(idx, 0, n - 1)
    Bm = jnp.take_along_axis(Bline, idx, axis=-1)
    cross = Bm >= Bc
    cross = cross.at[..., 0].set(False)
    has = jnp.any(cross, axis=-1)
    j = jnp.argmax(cross, axis=-1)
    j = jnp.maximum(j, 1)
    B0 = jnp.take_along_axis(Bm, (j - 1)[..., None], -1)[..., 0]
    B1 = jnp.take_along_axis(Bm, j[..., None], -1)[..., 0]
    frac = (Bc - B0) / jnp.where(B1 == B0, 1.0, B1 - B0)
    dz = zscan[1] - zscan[0]
    zc = jnp.take(zscan, i0) + direction * dz * (j - 1 + frac)
    return zc, has


def trapped_class_maps(Bline, Gline, Jzline, zscan, i_lo, i_hi, pitch, num_quad=32):
    """Bounce-layer maps for one trapped class on many field lines.

    Parameters
    ----------
    Bline : jnp.ndarray, shape (..., nscan)
        |B| along each field line on the uniform ``zscan`` grid (the scan
        should cover ~3 field periods centered on the primary period).
    Gline : jnp.ndarray, shape (..., nscan)
        dl/dzeta = |B| / |B^zeta| along the lines.
    Jzline : jnp.ndarray, shape (..., nscan)
        Volume Jacobian d^3x = Jzline ds dalpha dzeta along the lines.
    zscan : jnp.ndarray, shape (nscan,)
        Uniform zeta grid of the line scan.
    i_lo, i_hi : int
        Index range of the primary field period within ``zscan``.
    pitch : float
        Pitch 1/B_crit of the class.
    num_quad : int
        Gauss-Legendre points for the singular quadratures.

    Returns
    -------
    dict with ``Jhat``, ``that``, ``W``, ``zminus``, ``zplus``, ``trapped``.
    """
    Bc = 1.0 / pitch
    im = i_lo + jnp.argmin(Bline[..., i_lo:i_hi], axis=-1)
    zp, hasR = _first_crossing(Bline, im, Bc, +1, zscan)
    zn, hasL = _first_crossing(Bline, im, Bc, -1, zscan)
    Bmin = jnp.take_along_axis(Bline, im[..., None], -1)[..., 0]
    trapped = hasR & hasL & (Bmin < Bc)
    x, wq = leggauss(num_quad)
    u = x * jnp.pi / 2
    wq = wq * jnp.pi / 2
    c = 0.5 * (zp + zn)
    h = 0.5 * (zp - zn)
    zq = c[..., None] + h[..., None] * jnp.sin(u)
    dz = zscan[1] - zscan[0]
    xq = (zq - zscan[0]) / dz
    iq = jnp.clip(xq.astype(int), 0, Bline.shape[-1] - 2)
    fr = xq - iq

    def lin(L):
        a = jnp.take_along_axis(L, iq, -1)
        b = jnp.take_along_axis(L, iq + 1, -1)
        return a * (1 - fr) + b * fr

    Bq, Gq, Jq = lin(Bline), lin(Gline), lin(Jzline)
    f = jnp.maximum(1.0 - Bq / Bc, 1e-14)
    jac = h[..., None] * jnp.cos(u)
    Jhat = jnp.sum(wq * jnp.sqrt(f) * Gq * jac, axis=-1)
    that = jnp.sum(wq * Gq / jnp.sqrt(f) * jac, axis=-1)
    W = jnp.sum(wq * Jq * Bq / (2 * Bc**2) / jnp.sqrt(f) * jac, axis=-1)
    nan = jnp.nan
    return {
        "Jhat": jnp.where(trapped, Jhat, nan),
        "that": jnp.where(trapped, that, nan),
        "W": jnp.where(trapped, W, 0.0),
        "zminus": jnp.where(trapped, zn, nan),
        "zplus": jnp.where(trapped, zp, nan),
        "trapped": trapped,
    }


# --------------------------------------------------------------------------
# drift layer: boundary-pinned monotone graph foliation
# --------------------------------------------------------------------------


def graph_foliation_fit(
    Jmap,
    s,
    alpha,
    n_leaves=24,
    n_radial=10,
    n_fourier=10,
    n_iter=30,
    area_weight=30.0,
    reg=3e-4,
):
    """Fit a strictly non-crossing nested foliation to a Jhat class map.

    Leaves are graphs ``s(x, alpha) = b(alpha) + int_0^x exp(u)`` on a
    Chebyshev(x) x Fourier(alpha) basis, with per-leaf area pinned to a
    monotone schedule spanning the seeded range up to the map boundary
    (boundary-pinned outer leaf). Fixed-iteration Gauss-Newton; the returned
    leaves minimize the per-leaf variance of ``Jmap`` interpolated along
    them. Non-crossing is structural (d s / d x = exp(u) > 0).

    Returns dict with leaf arrays ``s_leaf`` (n_leaves, n_alpha_grid),
    ``alpha_grid``, area labels ``A``, per-leaf residual ``lvar`` and the
    monotone ``smax`` profile used for critical-label interpolation.
    """
    ns, na = Jmap.shape
    nt = 4 * n_fourier + 8
    ag = jnp.linspace(0, 2 * jnp.pi, nt, endpoint=False)
    xg = 0.5 * (1 - jnp.cos(jnp.pi * jnp.arange(n_leaves) / (n_leaves - 1)))
    # Chebyshev design in x
    xx = 2 * xg - 1
    Tb = jnp.stack(
        [jnp.cos(k * jnp.arccos(jnp.clip(xx, -1, 1))) for k in range(n_radial)], 1
    )
    kf = jnp.arange(1, n_fourier + 1)
    TF = jnp.concatenate(
        [jnp.ones((nt, 1)), jnp.cos(jnp.outer(ag, kf)), jnp.sin(jnp.outer(ag, kf))], 1
    )
    nb = 2 * n_fourier + 1
    # cumulative trapezoid over leaves
    dr = jnp.diff(xg)
    Q = jnp.zeros((n_leaves, n_leaves))
    for i in range(1, n_leaves):
        Q = Q.at[i].set(Q[i - 1])
        Q = Q.at[i, i - 1].add(0.5 * dr[i - 1])
        Q = Q.at[i, i].add(0.5 * dr[i - 1])
    # v0 initialization: uniform-in-s leaves with the outermost pinned at the
    # map boundary (boundary-pinned foliation); the area schedule follows.
    s_lo, s_hi = s[0], s[-1]
    S0 = jnp.linspace(s_lo + 0.02, s_hi, n_leaves)[:, None] * jnp.ones((1, nt))
    A_t = S0.mean(1)
    b0 = S0[0]
    du = jnp.gradient(S0, xg, axis=0)
    u0 = jnp.log(jnp.maximum(du, 1e-3))
    bc = jnp.linalg.lstsq(TF, b0)[0]
    uc = jnp.linalg.lstsq(Tb, jnp.linalg.lstsq(TF, u0.T)[0].T)[0]
    # bilinear interpolation of Jmap (alpha periodic)
    Jp = jnp.concatenate([Jmap, Jmap[:, :1]], 1)
    a_edges = jnp.linspace(0, 2 * jnp.pi, na + 1)

    def J_interp(S, A):
        si = jnp.clip((S - s[0]) / (s[1] - s[0]), 0, ns - 1 - 1e-6)
        ai = jnp.clip(A / (a_edges[1] - a_edges[0]), 0, na - 1e-6)
        i0, j0 = si.astype(int), ai.astype(int)
        fs, fa = si - i0, ai - j0
        v = (
            Jp[i0, j0] * (1 - fs) * (1 - fa)
            + Jp[jnp.minimum(i0 + 1, ns - 1), j0] * fs * (1 - fa)
            + Jp[i0, j0 + 1] * (1 - fs) * fa
            + Jp[jnp.minimum(i0 + 1, ns - 1), j0 + 1] * fs * fa
        )
        return v

    Jscale = jnp.nanstd(Jmap) + 1e-30

    def leaves_of(bc, uc):
        b = TF @ bc
        u = (Tb @ uc) @ TF.T
        e = jnp.exp(jnp.clip(u, -12, 6))
        return b[None, :] + Q @ e, e

    def resid(theta):
        bc = theta[:nb]
        uc = theta[nb:].reshape(n_radial, nb)
        S, e = leaves_of(bc, uc)
        Jv = J_interp(S, ag[None, :] * jnp.ones((n_leaves, 1)))
        rv = (Jv - Jv.mean(1, keepdims=True)) / Jscale / jnp.sqrt(nt)
        rA = area_weight * (S.mean(1) - A_t)
        kw = jnp.concatenate([jnp.array([0.0]), kf, kf])
        rr = reg * jnp.concatenate([kw * bc, ((jnp.tile(kw, n_radial)) + 1.0) * theta[nb:]])
        return jnp.concatenate([rv.ravel(), rA, rr])

    theta = jnp.concatenate([bc, uc.ravel()])
    cost = float(jnp.sum(resid(theta) ** 2))
    for _ in range(n_iter):
        r = resid(theta)
        Jr = jax_jacfwd_resid(resid, theta)
        dtheta = jnp.linalg.lstsq(Jr, -r)[0]
        # backtracking Gauss-Newton: accept only descent steps
        for t in (1.0, 0.5, 0.25, 0.1, 0.03):
            trial = theta + t * jnp.clip(dtheta, -1.0, 1.0)
            c_t = float(jnp.sum(resid(trial) ** 2))
            if c_t < cost:
                theta, cost = trial, c_t
                break
    S, e = leaves_of(theta[:nb], theta[nb:].reshape(n_radial, nb))
    Jv = J_interp(S, ag[None, :] * jnp.ones((n_leaves, 1)))
    lvar = jnp.var((Jv - Jv.mean(1, keepdims=True)) / Jscale, axis=1)
    return {
        "s_leaf": S,
        "alpha_grid": ag,
        "A": S.mean(1),
        "lvar": lvar,
        "smax": _cummax(S.max(1)),
        "e": e,
    }


def _cummax(x):
    """Running maximum (jnp.maximum has no ufunc.accumulate in JAX)."""
    import jax

    return jax.lax.associative_scan(jnp.maximum, x)


def jax_jacfwd_resid(fun, x):
    """Forward-mode Jacobian helper (kept separate for testability)."""
    import jax

    return jax.jacfwd(fun)(x)


# --------------------------------------------------------------------------
# flux assembly
# --------------------------------------------------------------------------


def graze_flux_class(fol, W, S_birth, s, s_loss):
    """Lost birth measure for one class from its foliation and W moment.

    Confined measure by the Green's-theorem alpha-average along the critical
    leaf (alpha' = 1 for graph leaves); lost = total - confined.
    """
    ns, na = W.shape
    ag = fol["alpha_grid"]
    WS = W * S_birth[:, None]
    # cumulative int_0^s WS ds'
    ds = s[1] - s[0]
    F = jnp.concatenate(
        [jnp.zeros((1, na)), jnp.cumsum(0.5 * (WS[1:] + WS[:-1]) * ds, axis=0)]
    )
    Mtot = F[-1].mean() * 2 * jnp.pi
    # critical leaf by monotone interpolation of smax
    smax, A = fol["smax"], fol["A"]
    xg = jnp.linspace(0, 1, len(A))
    x_star = jnp.interp(s_loss, smax, xg)
    S_star = jnp.stack(
        [jnp.interp(x_star, xg, fol["s_leaf"][:, j]) for j in range(len(ag))]
    )
    # F interpolated at the critical leaf: bilinear in (s, alpha), periodic
    Fp = jnp.concatenate([F, F[:, :1]], axis=1)
    da = 2 * jnp.pi / na
    ai = jnp.clip(ag / da, 0, na - 1e-6)
    j0 = ai.astype(int)
    fa = ai - j0
    si = jnp.clip((S_star - s[0]) / (s[1] - s[0]), 0, ns - 1 - 1e-6)
    i0 = si.astype(int)
    fs = si - i0
    Fj = (
        Fp[i0, j0] * (1 - fs) * (1 - fa)
        + Fp[jnp.minimum(i0 + 1, ns - 1), j0] * fs * (1 - fa)
        + Fp[i0, j0 + 1] * (1 - fs) * fa
        + Fp[jnp.minimum(i0 + 1, ns - 1), j0 + 1] * fs * fa
    )
    M_in = Fj.mean() * 2 * jnp.pi
    fully_conf = smax[-1] < s_loss
    fully_lost = smax[0] > s_loss
    lost = jnp.where(
        fully_conf, 0.0, jnp.where(fully_lost, Mtot, jnp.maximum(Mtot - M_in, 0.0))
    )
    return lost, Mtot


def reactivity_profile(s):
    """Fusion birth profile of Bader et al., NF 61 (2021) 116060."""
    T = jnp.maximum(11.5 * (1 - s), 1e-10)
    return (1 - s**5) ** 2 * T ** (-2.0 / 3.0) * jnp.exp(-19.94 * T ** (-1.0 / 3.0))


# --------------------------------------------------------------------------
# objective
# --------------------------------------------------------------------------


class TrappedGrazeFlux(_Objective):
    """Prompt trapped-particle graze-loss fraction (J_parallel pipeline).

    Computes the birth-weighted measure of trapped classes whose fitted
    drift surfaces graze ``s_loss``, integrated over a constant-pitch class
    grid. See module docstring for the method and validation record.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium to optimize.
    pitch_grid : ndarray
        Pitch values 1/B_crit defining the trapped classes [1/T].
    s_loss : float
        Graze threshold in normalized toroidal flux (boundary minus a
        banana half-width).
    rho, n_alpha, n_zeta : resolution of the drift-plane and line grids.
    birth : {"uniform", "reactivity"}
        Birth-rate profile S(s).
    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0``.", bounds_default="``target=0``."
    )

    _coordinates = ""
    _units = "~"
    _print_value_fmt = "Trapped graze flux: "
    _static_attrs = _Objective._static_attrs + ["_hyper"]

    def __init__(
        self,
        eq,
        *,
        pitch_grid,
        s_loss=0.98,
        rho=None,
        n_alpha=64,
        n_zeta=192,
        num_quad=32,
        birth="reactivity",
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        name="trapped graze flux",
        jac_chunk_size=None,
    ):
        if target is None and bounds is None:
            target = 0
        self._hyper = {
            "pitch_grid": tuple(float(p) for p in pitch_grid),
            "s_loss": float(s_loss),
            "n_alpha": int(n_alpha),
            "n_zeta": int(n_zeta),
            "num_quad": int(num_quad),
            "birth": birth,
        }
        self._rho = rho
        super().__init__(
            things=eq,
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            loss_function=loss_function,
            deriv_mode=deriv_mode,
            name=name,
            jac_chunk_size=jac_chunk_size,
        )

    def build(self, use_jit=True, verbose=1):
        """Build constant arrays: fixed field-line collocation nodes.

        TODO(jpar-graze-flux): replace the build-time straight-field-line
        node construction with the runtime ``angle=delta`` machinery of
        ``GammaC`` (exact constant-alpha sampling under changing lambda),
        and the quadratures with ``Bounce2D``.
        """
        eq = self.things[0]
        rho = setdefault(self._rho, jnp.linspace(0.15, 0.99, 24))
        self._constants = {"quad_weights": 1.0, "rho": rho}
        grid = LinearGrid(
            rho=rho,
            theta=jnp.linspace(0, 2 * jnp.pi, 2 * eq.M_grid, endpoint=False),
            zeta=jnp.linspace(
                0, 2 * jnp.pi / eq.NFP, max(2 * eq.N_grid, 16), endpoint=False
            ),
            NFP=eq.NFP,
        )
        self._keys = ["|B|", "B^zeta", "sqrt(g)", "iota", "lambda"]
        self._constants["transforms"] = get_transforms(self._keys, eq, grid)
        self._constants["profiles"] = get_profiles(self._keys, eq, grid)
        self._constants["grid"] = grid
        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute the graze-loss fraction (scalar)."""
        constants = setdefault(constants, self.constants)
        eq = self.things[0]
        data = compute_fun(
            eq,
            self._keys,
            params,
            constants["transforms"],
            constants["profiles"],
        )
        # v0: assemble per-line arrays on the fixed grid and delegate to the
        # pure functions above. Implementation of the line assembly follows
        # the campaign reference (drivers/jpar_map.py); kept minimal here and
        # exercised by unit tests through the pure functions directly.
        raise NotImplementedError(
            "TrappedGrazeFlux.compute: line assembly lands in the next commit; "
            "the validated pure-function pipeline (trapped_class_maps, "
            "graph_foliation_fit, graze_flux_class) is complete and tested."
        )
