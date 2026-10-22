"""Tests for the linear local-PNG + Kaiser basis vectors of :py:mod:`desiwinds.convenience` and the basis names of :py:mod:`desiwinds.window`."""

import lsstypes as lss
import numpy as np
import pytest

pytest.importorskip("cosmoprimo")

from desiwinds.convenience import (
    PNG_ALL_BASIS_NAMES,
    PNG_KAISER_BASIS_NAMES,
    _png_shapes,
    get_growth_ratio,
    get_png_basis_vectors,
    get_png_kaiser_coefficients,
)
from desiwinds.window import get_basis_vectors, project_window

ELLS = (0, 2, 4)


@pytest.fixture(scope="module")
def cosmology():
    from cosmoprimo.fiducial import DESI

    return DESI(engine="camb")


@pytest.fixture(scope="module")
def theory():
    # usual window theory edges: the k = 0 mode alone, then regular bins up to the mesh Nyquist frequency
    edges = np.concatenate([[0.0, 2 * np.pi / 14000.0], np.arange(0.002, 0.158, 0.002)])
    k_edges = np.stack([edges[:-1], edges[1:]], axis=-1)
    k = k_edges.mean(axis=-1)
    return lss.Mesh2SpectrumPoles([lss.Mesh2SpectrumPole(k=k, k_edges=k_edges, num_raw=np.ones(k.size), ell=ell) for ell in ELLS], ells=list(ELLS))


def _blocks(vectors, theory):
    """Split the multipoles: ``(..., n_theory)`` -> ``(..., n_ells, n_k)``."""
    return vectors.reshape(*vectors.shape[:-1], len(theory.ells), -1)


@pytest.mark.parametrize("responses, names", [("kaiser", PNG_KAISER_BASIS_NAMES), ("all", PNG_ALL_BASIS_NAMES)])
def test_basis_layout(cosmology, theory, responses, names):
    vectors, out_names = get_png_basis_vectors(theory, cosmology=cosmology, responses=responses)
    assert out_names == names
    assert vectors.shape == (len(names), theory.value().size)
    blocks = _blocks(vectors, theory)
    for ivector, name in enumerate(names):
        ell = int(name.rsplit("_ell", 1)[1])
        for iell, other in enumerate(theory.ells):
            assert np.any(blocks[ivector, iell] != 0) == (other == ell)
        if name.startswith("g"):  # g shapes are set to 0 in the k = 0 bin
            assert blocks[ivector, theory.ells.index(ell), 0] == 0.0
        else:
            assert blocks[ivector, theory.ells.index(ell), 0] > 0.0


def test_center_and_modes_average(cosmology, theory):
    k_edges = np.asarray(theory.get(0).edges("k"))
    center, _ = get_png_basis_vectors(theory, cosmology=cosmology, k_average="center")
    modes, _ = get_png_basis_vectors(theory, cosmology=cosmology, k_average="modes")
    kc = k_edges[1:].mean(axis=-1)
    pk0, g0 = _png_shapes(cosmology, kc)
    for ivector, shape in enumerate([pk0, g0 * pk0, g0**2 * pk0]):
        assert np.allclose(_blocks(center, theory)[ivector, 0, 1:], shape)
    # "modes": k^2-weighted average over the bin, checked against a brute-force average in one low and one high bin
    for ibin in (2, 50):
        k = np.linspace(*k_edges[ibin], 20001)
        pk0, g0 = _png_shapes(cosmology, k)
        for ivector, shape in enumerate([pk0, g0 * pk0, g0**2 * pk0]):
            expected = np.trapezoid(k**2 * shape, k) / np.trapezoid(k**2, k)
            assert _blocks(modes, theory)[ivector, 0, ibin] == pytest.approx(expected, rel=1e-6)


def _direct_poles(cosmology, theory, z, b1, bphi_fnl, f):
    """Multipoles of (b1 + bphi fnl g(k, z) + f mu^2)^2 P_dd(k, z), computed directly at the bin centers (except the k = 0 bin)."""
    kc = np.asarray(theory.get(0).edges("k"))[1:].mean(axis=-1)
    pkz = cosmology.get_fourier().pk_interpolator(of="delta_cb").to_1d(z=z)(kc)
    pk_prim = cosmology.get_primordial(mode="scalar").pk_interpolator()(kc)
    gz = np.sqrt(9.0 / 25.0 * 2.0 * np.pi**2 / kc**3 * pk_prim / cosmology.h**3 / pkz)
    mu, weights = np.polynomial.legendre.leggauss(16)
    pkmu = (b1 + bphi_fnl * gz[:, None] + f * mu**2) ** 2 * pkz[:, None]
    return np.array([(2 * ell + 1) / 2 * np.sum(weights * np.polynomial.legendre.legval(mu, [0] * ell + [1]) * pkmu, axis=-1) for ell in theory.ells])


@pytest.mark.parametrize("z", [0.8, 2.5])
def test_kaiser_closure(cosmology, theory, z):
    """sum_i d_i u_i are the multipoles of the model: exactly with shapes at z, up to the neutrino scale-dependent growth with z = 0 shapes."""
    b1, bphi_fnl, f = 2.3, 30.0, 0.95
    expected = _direct_poles(cosmology, theory, z, b1, bphi_fnl, f)
    vectors_z, _ = get_png_basis_vectors(theory, cosmology=cosmology, k_average="center", z=z)
    exact = _blocks(np.asarray(get_png_kaiser_coefficients(b1, bphi_fnl, f, 1.0)) @ vectors_z, theory)[:, 1:]
    assert np.allclose(exact, expected, rtol=1e-10, atol=0)
    vectors_0, _ = get_png_basis_vectors(theory, cosmology=cosmology, k_average="center")
    r = get_growth_ratio(cosmology, z)
    separable = _blocks(np.asarray(get_png_kaiser_coefficients(b1, bphi_fnl, f, r)) @ vectors_0, theory)[:, 1:]
    relative = np.abs(separable - expected) / np.abs(expected).max(axis=-1, keepdims=True)
    assert relative.max() < 7e-3  # P_dd(z) / (r^2 P_0) - 1 reaches 0.6% at z = 2.5, k = 0.16 h/Mpc
    monopole = np.abs(separable[0] - expected[0]) / expected[0]
    assert monopole.max() < 7e-3


def test_coefficients_jax(cosmology):
    import jax
    import jax.numpy as jnp

    coefficients = jax.jit(get_png_kaiser_coefficients)(jnp.array(2.0), jnp.array(10.0), jnp.array(0.9), jnp.array(0.5))
    assert coefficients.shape == (len(PNG_KAISER_BASIS_NAMES),)
    assert float(coefficients[2]) == pytest.approx(100.0)  # (bphi fnl)^2, no growth factor


def test_basis_names_roundtrip(cosmology, theory, tmp_path):
    vectors, names = get_png_basis_vectors(theory, cosmology=cosmology)
    rng = np.random.default_rng(0)
    observable = lss.ObservableLeaf(value=np.zeros(7))
    window = lss.WindowMatrix(value=rng.normal(size=(7, theory.value().size)), observable=observable, theory=theory)
    projected = project_window(window, vectors, names=names)
    projected.write(tmp_path / "window.h5")
    read_vectors, read_names = get_basis_vectors(lss.read(tmp_path / "window.h5").theory, return_names=True)
    assert read_names == names
    assert np.allclose(read_vectors, vectors)
    assert get_basis_vectors(project_window(window, vectors).theory, return_names=True)[1] is None
    with pytest.raises(ValueError):
        project_window(window, vectors, names=names[:-1])
