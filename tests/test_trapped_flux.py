"""Unit tests for the trapped graze-flux pipeline (pure functions)."""

import numpy as np
import pytest

from desc.backend import jnp
from desc.objectives._trapped_flux import (
    graph_foliation_fit,
    graze_flux_class,
    reactivity_profile,
    trapped_class_maps,
)


class TestBounceLayer:
    """Bounce-layer quadratures on an analytic mirror field."""

    @staticmethod
    def _lines(n_alpha=8, nscan=1537, nfp=4, eps=0.2):
        T = 2 * np.pi / nfp
        zscan = np.linspace(-T, 2 * T, nscan)
        alpha = np.linspace(0, 2 * np.pi, n_alpha, endpoint=False)
        B0 = 5.0 * (1 + 0.02 * np.cos(alpha))[:, None]
        B = B0 * (1 - eps * np.cos(nfp * zscan)[None, :])
        G = np.full_like(B, 3.0)
        Jz = np.full_like(B, 0.7)
        i_lo = int(np.ceil((0 + T) / (zscan[1] - zscan[0])))
        i_hi = int(np.floor((T + T) / (zscan[1] - zscan[0])))
        return map(jnp.asarray, (B, G, Jz, zscan)), (i_lo, i_hi), (B0, eps, nfp)

    @pytest.mark.unit
    def test_trapped_mask_and_turning(self):
        (B, G, Jz, zscan), (i_lo, i_hi), (B0, eps, nfp) = self._lines()
        Bc = 5.0  # between Bmin=4(1-.02..) and Bmax=6
        out = trapped_class_maps(B, G, Jz, zscan, i_lo, i_hi, 1.0 / Bc)
        assert bool(jnp.all(out["trapped"]))
        # analytic turning points: cos(nfp z) = (1 - Bc/B0)/eps
        c = (1 - Bc / B0[:, 0]) / eps
        z_an = np.arccos(np.clip(c, -1, 1)) / nfp
        np.testing.assert_allclose(np.asarray(out["zplus"]), z_an, atol=2e-3)
        np.testing.assert_allclose(np.asarray(out["zminus"]), -z_an, atol=2e-3)

    @pytest.mark.unit
    def test_quadrature_matches_dense_reference(self):
        (B, G, Jz, zscan), (i_lo, i_hi), _ = self._lines()
        Bc = 5.3
        out = trapped_class_maps(B, G, Jz, zscan, i_lo, i_hi, 1.0 / Bc, num_quad=48)
        # dense trapezoid reference on the first line (singular endpoints:
        # use fine grid + sqrt behavior -> few 1e-4 relative suffices)
        zf = np.linspace(float(out["zminus"][0]), float(out["zplus"][0]), 200001)
        Bf = np.interp(zf, np.asarray(zscan), np.asarray(B[0]))
        f = np.maximum(1 - Bf / Bc, 0)
        ref = np.trapz(np.sqrt(f) * 3.0, zf)
        np.testing.assert_allclose(float(out["Jhat"][0]), ref, rtol=5e-4)

    @pytest.mark.unit
    def test_pitch_measure_normalization(self):
        # int_B^Bmax B/(2 Bc^2 sqrt(1-B/Bc)) dBc = sqrt(1 - B/Bmax);
        # integrate the truncated interval [B(1+d), Bmax] and compare with its
        # exact value (the integrand has an integrable 1/sqrt endpoint).
        B, Bmax, d = 5.0, 6.0, 1e-4
        Bc = np.linspace(B * (1 + d), Bmax, 200001)
        dP = B / (2 * Bc**2 * np.sqrt(1 - B / Bc))
        exact = np.sqrt(1 - B / Bmax) - np.sqrt(d / (1 + d))
        np.testing.assert_allclose(np.trapz(dP, Bc), exact, rtol=1e-4)


class TestDriftLayer:
    """Foliation fit and flux assembly on synthetic nested maps."""

    @staticmethod
    def _jmap(ns=48, na=64):
        s = np.linspace(0.02, 0.985, ns)
        a = np.linspace(0, 2 * np.pi, na, endpoint=False)
        S, A = np.meshgrid(s, a, indexing="ij")
        J = 1.0 - 0.8 * S + 0.05 * S * np.cos(2 * A)
        return jnp.asarray(J), jnp.asarray(s), jnp.asarray(a)

    @pytest.mark.unit
    def test_foliation_noncrossing_and_residual(self):
        J, s, a = self._jmap()
        fol = graph_foliation_fit(J, s, a, n_iter=20)
        gaps = jnp.diff(fol["s_leaf"], axis=0)
        assert float(gaps.min()) > 0  # structural non-crossing
        assert float(jnp.median(fol["lvar"])) < 5e-3
        assert bool(jnp.all(jnp.diff(fol["smax"]) >= -1e-9))

    @pytest.mark.unit
    def test_graze_flux_measure_consistency(self):
        J, s, a = self._jmap()
        fol = graph_foliation_fit(J, s, a, n_iter=20)
        W = jnp.ones((len(s), len(a)))
        Sb = jnp.ones(len(s))
        s_loss = 0.9
        lost, Mtot = graze_flux_class(fol, W, Sb, s, s_loss)
        # uniform W: total measure = (s span) * 2pi
        np.testing.assert_allclose(float(Mtot), float(s[-1] - s[0]) * 2 * np.pi, rtol=1e-6)
        assert 0.0 <= float(lost) <= float(Mtot)
        # tightening the threshold cannot decrease the lost measure
        lost2, _ = graze_flux_class(fol, W, Sb, s, 0.8)
        assert float(lost2) >= float(lost) - 1e-9

    @pytest.mark.unit
    def test_reactivity_profile_shape(self):
        s = jnp.linspace(0, 0.99, 50)
        S = reactivity_profile(s)
        assert bool(jnp.all(S >= 0)) and float(S[0]) > float(S[-1])
