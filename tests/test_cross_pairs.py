r"""
Basis windows of cross-spectra with the local line of sight and a theory :math:`A_{\ell_1 \ell_2}` (pairs of multipoles, :math:`A_{02} \neq A_{20}`).

Two tracers in overlapping balls on a ~30^3 mesh, as in ``test_window_basis.py``; requires a jaxpower whose ``generate_anisotropic_gaussian_mesh`` accepts pairs ``(ell1, ell2)``.
"""

import jax

jax.config.update("jax_enable_x64", True)

import lsstypes as lss
import numpy as np
import pytest
from jaxpower import BinMesh2SpectrumPoles, FKPField, ParticleField, compute_fkp2_normalization, get_mesh_attrs

pytest.importorskip("cosmoprimo")


def _jaxpower_has_pairs():
    import inspect

    import jaxpower.mock

    return "local_pairs" in inspect.getsource(jaxpower.mock)


if not _jaxpower_has_pairs():
    pytest.skip("this jaxpower does not generate meshes for pairs (ell1, ell2)", allow_module_level=True)

from desiwinds import post
from desiwinds.convenience import LOCAL_PAIRS, PNG_KAISER_PAIR_BASIS_NAMES, fiducial_theory, get_png_basis_vectors, get_png_kaiser_pair_coefficients
from desiwinds.forward import mock_survey_catalog
from desiwinds.window import get_basis_vectors, get_fiducial_theory, get_window_basis, get_window_spikes, project_window

N_PARTICLES = 10000


@pytest.fixture(scope="module")
def cross():
    rng = np.random.default_rng(7)

    def ball(n, center):
        x = rng.normal(size=(n, 3))
        x *= (150 * rng.uniform(size=n) ** (1 / 3) / np.linalg.norm(x, axis=1))[:, None]
        return x + np.array(center)

    positions = {name: (ball(N_PARTICLES, center), ball(N_PARTICLES, center)) for name, center in [("X", [800.0, 0.0, 0.0]), ("Y", [820.0, 20.0, 0.0])]}
    mattrs = get_mesh_attrs(*[p for pair in positions.values() for p in pair], cellsize=20.0, check=True)
    fkps = tuple(FKPField(ParticleField(d, np.ones(N_PARTICLES), attrs=mattrs), ParticleField(r, np.ones(N_PARTICLES), attrs=mattrs), attrs=mattrs) for d, r in positions.values())
    binner = BinMesh2SpectrumPoles(mattrs, edges={"step": 0.02}, ells=(0, 2, 4))
    norm = compute_fkp2_normalization(*fkps, bin=binner, cellsize=20.0)
    theory = fiducial_theory(np.arange(0.0, 0.16, 0.02), bA=2.0, bB=1.0)  # A02 != A20

    def mock_survey(theory, seed, return_data_shotnoise=False):
        return mock_survey_catalog(
            fkps,
            theory=theory,
            seed=seed,
            los="local",
            unitary_amplitude=True,
            ric_args=None,
            amr_args=None,
            nam_args=None,
            binner=binner,
            fkp_norms=[norm],
            estimator_weights=None,
            return_data_shotnoise=return_data_shotnoise,
        )

    vectors, names = get_png_basis_vectors(theory, k_average="center")
    spikes = get_window_spikes(mock_survey, theory, nreal=1, batch_size=4)
    basis = get_window_basis(mock_survey, theory, vectors, basis_names=names, nreal=1, batch_size=4)
    return dict(theory=theory, vectors=vectors, names=names, spikes=spikes, basis=basis)


def test_pair_basis_vectors(cross):
    theory, vectors, names = cross["theory"], cross["vectors"], cross["names"]
    assert list(theory.ells) == LOCAL_PAIRS
    assert names == PNG_KAISER_PAIR_BASIS_NAMES and vectors.shape == (8, theory.value().size)
    blocks = vectors.reshape(len(names), len(LOCAL_PAIRS), -1)
    for ivector, name in enumerate(names):
        pair = tuple(int(e) for e in name.rsplit("_ell", 1)[1])
        assert [bool(np.any(block != 0)) for block in blocks[ivector]] == [other == pair for other in LOCAL_PAIRS]


def test_pair_coefficients_closure():
    """sum_i d_i u_i = beta^X_{l1} beta^Y_{l2} P (shapes evaluated at the bin centers, no PNG in the first bin)."""
    from cosmoprimo.fiducial import DESI

    from desiwinds.convenience import _png_shapes

    cosmology = DESI(engine="camb")
    edges = np.arange(0.002, 0.1, 0.004)
    k_edges = np.stack([edges[:-1], edges[1:]], -1)
    k = k_edges.mean(axis=-1)
    theory = lss.ObservableTree([lss.ObservableLeaf(k=k, k_edges=k_edges, value=np.ones(k.size), coords=["k"]) for _ in LOCAL_PAIRS], ells=LOCAL_PAIRS)
    vectors, _ = get_png_basis_vectors(theory, cosmology=cosmology, k_average="center")
    b1, bphi_fnl, f = (2.0, 1.2), (30.0, -10.0), 0.9
    model = (np.asarray(get_png_kaiser_pair_coefficients(b1, bphi_fnl, f, 1.0)) @ vectors).reshape(len(LOCAL_PAIRS), -1)
    pk0, g0 = _png_shapes(cosmology, k)
    beta = [{0: b + bf * g0 + f / 3.0, 2: 2.0 * f / 3.0 * np.ones_like(k)} for b, bf in zip(b1, bphi_fnl)]
    expected = np.array([beta[0][l1] * beta[1][l2] * pk0 for l1, l2 in LOCAL_PAIRS])
    assert np.allclose(model, expected, rtol=1e-10, atol=0)


def test_cross_basis_equals_projected_spikes(cross):
    """The jvp is linear in the direction, also with two meshes: the pair basis window is the pair spike window times the vectors."""
    spike, basis = cross["spikes"][0][0], cross["basis"][0][0]
    expected = np.asarray(spike.value()) @ cross["vectors"].T
    assert np.allclose(np.asarray(basis.value()), expected, rtol=0, atol=1e-10 * np.abs(expected).max())
    # A02 and A20 are distinct theory directions
    nk = cross["theory"].get((0, 2)).size
    w = np.asarray(spike.value())
    assert not np.allclose(w[:, nk : 2 * nk], w[:, 2 * nk : 3 * nk])


def test_pair_basis_theory_roundtrip(cross, tmp_path):
    basis = cross["basis"][0][0]
    basis.write(tmp_path / "window.h5")
    read = lss.read(tmp_path / "window.h5")
    fiducial = get_fiducial_theory(read.theory)
    assert list(fiducial.ells) == LOCAL_PAIRS
    assert np.allclose(np.asarray(fiducial.value()), np.asarray(cross["theory"].value()))
    vectors, names = get_basis_vectors(read.theory, return_names=True)
    assert names == cross["names"] and np.allclose(vectors, cross["vectors"])
    projected = project_window(cross["spikes"][0][0], cross["vectors"], names=cross["names"], basis_theory=fiducial)
    assert np.allclose(np.asarray(projected.value()), np.asarray(basis.value()), rtol=0, atol=1e-10 * np.abs(basis.value()).max())


def test_selfpairs_cross(cross):
    """No self-pairs in the cross of two tracers: the analytic correction is the identity, the density variance is not defined for pairs."""
    basis = cross["basis"][0][0]
    assert post.correct_selfpairs_analytic(basis, None, 0.0, None) is basis
    with pytest.raises(NotImplementedError):
        post.get_density_variance_derivative(cross["theory"], None)
