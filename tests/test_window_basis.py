"""
Tests for :py:func:`desiwinds.window.get_window_basis`, the shot-noise Jacobians and :py:mod:`desiwinds.post`.

A small synthetic survey (uniform randoms in a ball, split 50/50 into data and randoms, global integral constraint) on a ~30^3 mesh,
so that everything runs on CPU in a few tens of seconds.
"""

import jax

jax.config.update("jax_enable_x64", True)

import lsstypes as lss
import numpy as np
import pytest
from jaxpower import BinMesh2SpectrumPoles, FKPField, ParticleField, compute_fkp2_normalization, get_mesh_attrs

from desiwinds import post
from desiwinds.forward import mock_survey_catalog
from desiwinds.window import get_basis_theory, get_basis_vectors, get_fiducial_theory, get_window_basis, get_window_spikes, project_window

N_PARTICLES = 20000
NREAL = 2


@pytest.fixture(scope="module")
def survey():
    rng = np.random.default_rng(42)

    def ball(n):
        x = rng.normal(size=(n, 3))
        x *= (150 * rng.uniform(size=n) ** (1 / 3) / np.linalg.norm(x, axis=1))[:, None]
        return x + np.array([800.0, 0.0, 0.0])

    data_positions, randoms_positions = ball(N_PARTICLES), ball(N_PARTICLES)
    mattrs = get_mesh_attrs(data_positions, randoms_positions, cellsize=20.0, check=True)
    data = ParticleField(data_positions, np.ones(N_PARTICLES), attrs=mattrs)
    randoms = ParticleField(randoms_positions, np.ones(N_PARTICLES), attrs=mattrs)
    fkp = FKPField(data, randoms, attrs=mattrs)
    binner = BinMesh2SpectrumPoles(mattrs, edges={"step": 0.02}, ells=(0, 2, 4))
    norm = compute_fkp2_normalization(fkp, bin=binner, cellsize=20.0)

    edges = np.arange(0.0, 0.16, 0.02)
    k = (edges[:-1] + edges[1:]) / 2
    p0 = 2e4 * np.exp(-k / 0.1)
    theory = lss.Mesh2SpectrumPoles(
        [
            lss.Mesh2SpectrumPole(k=k, k_edges=np.stack([edges[:-1], edges[1:]], -1), num_raw=factor * p0, ell=ell)
            for ell, factor in [(0, 1.0), (2, 0.3), (4, 0.05)]
        ],
        ells=[0, 2, 4],
    )

    def mock_survey(theory, seed, return_data_shotnoise=False):
        return mock_survey_catalog(
            fkp,
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

    basis_vectors = rng.normal(size=(5, theory.value().size))
    spikes = get_window_spikes(mock_survey, theory, nreal=NREAL, batch_size=4, return_shotnoise_jacobians=True)
    basis = get_window_basis(mock_survey, theory, basis_vectors, nreal=NREAL, batch_size=4, return_shotnoise_jacobians=True)
    unclustered_data_shotnoise = float(np.sum(np.ones(N_PARTICLES) ** 2) / np.ravel(norm)[0])
    return dict(
        mattrs=mattrs,
        theory=theory,
        mock_survey=mock_survey,
        basis_vectors=basis_vectors,
        spikes=spikes,
        basis=basis,
        unclustered_data_shotnoise=unclustered_data_shotnoise,
    )


def test_spike_window_is_jacobian(survey):
    """A spike window column is the jvp of the forward model along the theory bin, averaged over seeds."""
    theory, mock_survey = survey["theory"], survey["mock_survey"]
    window = survey["spikes"][0]["window"][0]
    fiducial = np.asarray(theory.value())

    def response(value, seed):
        (spectrum,) = mock_survey(theory=theory.clone(value=value), seed=jax.random.key(seed))
        return jax.numpy.concatenate(spectrum.value(concatenate=False)).real

    for column in (2, 10):
        tangent = np.zeros_like(fiducial)
        tangent[column] = 1.0
        expected = np.mean([jax.jvp(lambda value, seed=seed: response(value, seed), (fiducial,), (tangent,))[1] for seed in (3, 5)], axis=0)
        assert np.allclose(np.asarray(window.value())[:, column], expected, rtol=1e-8, atol=1e-8 * np.abs(expected).max())
    assert window.theory is theory or np.array_equal(np.asarray(window.theory.value()), fiducial)


def test_shotnoise_flag_does_not_change_window(survey):
    without = get_window_spikes(survey["mock_survey"], survey["theory"], nreal=NREAL, batch_size=4)[0][0]
    assert np.array_equal(np.asarray(without.value()), np.asarray(survey["spikes"][0]["window"][0].value()))


@pytest.mark.parametrize("product", ["window", "shotnoise_jacobian", "data_shotnoise_jacobian"])
def test_basis_equals_projected_spikes(survey, product):
    """The jvp is linear in the direction: realization by realization, a basis window is the spike window times the vectors."""
    vectors = survey["basis_vectors"]
    for spike, basis in zip(survey["spikes"][1][product], survey["basis"][1][product], strict=True):
        expected = np.asarray(spike[0].value()) @ vectors.T
        assert np.allclose(np.asarray(basis[0].value()), expected, rtol=0, atol=1e-10 * np.abs(expected).max())


def test_project_window(survey):
    projected = project_window(survey["spikes"][0]["window"][0], survey["basis_vectors"])
    basis = survey["basis"][0]["window"][0]
    assert np.allclose(np.asarray(projected.value()), np.asarray(basis.value()), rtol=0, atol=1e-10 * np.abs(basis.value()).max())
    assert np.allclose(get_basis_vectors(projected.theory), get_basis_vectors(basis.theory))


def test_basis_theory_roundtrip(survey, tmp_path):
    theory, vectors = survey["theory"], survey["basis_vectors"]
    window = survey["basis"][0]["window"][0]
    window.write(tmp_path / "window.h5")
    read = lss.read(tmp_path / "window.h5")
    assert read.theory.value().size == len(vectors) == read.value().shape[1]
    assert np.allclose(get_basis_vectors(read.theory), vectors)
    fiducial = get_fiducial_theory(read.theory)
    assert list(fiducial.ells) == list(theory.ells)
    assert np.allclose(np.asarray(fiducial.value()), np.asarray(theory.value()))
    assert np.allclose(np.asarray(fiducial.get(0).edges("k")), np.asarray(theory.get(0).edges("k")))
    # the stored coefficients are the least-squares projection of the fiducial on the vectors
    coefficients = np.linalg.lstsq(vectors.T, np.asarray(theory.value()), rcond=None)[0]
    assert np.allclose(np.asarray(read.theory.value()), coefficients)
    with pytest.raises(ValueError):
        get_basis_theory(theory, vectors[:, :-1])


def test_data_shotnoise_jacobian_matches_variance_derivative(survey):
    """The data self-pair noise is unclustered_data_shotnoise x var(delta): its jvp matches the analytic derivative of the variance."""
    theory, mock_survey = survey["theory"], survey["mock_survey"]
    windows = get_window_spikes(mock_survey, theory, nreal=12, batch_size=4, return_shotnoise_jacobians=True)[0]
    measured = np.ravel(windows["data_shotnoise_jacobian"][0].value())
    expected = survey["unclustered_data_shotnoise"] * post.get_density_variance_derivative(theory, survey["mattrs"])
    nk = theory.get(0).size
    assert np.all(expected[nk:] == 0.0)
    high = slice(3, nk)  # the lowest bins have few modes and noisy jvps
    assert np.sum(measured[high]) == pytest.approx(np.sum(expected[high]), rel=0.1)


def _white_noise_template(observable, rng):
    """Synthetic template on the binning of ``observable``: 1 + noise in the monopole, noise elsewhere."""
    return observable.map(lambda pole, label: pole.clone(value=float(label["ells"] == 0) + 0.1 * rng.normal(size=pole.size)), input_label=True)


def test_correct_selfpairs(survey):
    rng = np.random.default_rng(0)
    products = {product: windows[0] for product, windows in survey["spikes"][0].items()}
    window = products["window"]
    template = _white_noise_template(window.observable, rng)
    corrected = post.correct_selfpairs(window, products["shotnoise_jacobian"], products["data_shotnoise_jacobian"], template)
    flat_template = np.asarray(template.value())
    expected = (
        np.asarray(window.value())
        + np.asarray(products["shotnoise_jacobian"].value())
        - np.outer(flat_template, np.ravel(products["data_shotnoise_jacobian"].value()))
    )
    assert np.allclose(np.asarray(corrected.value()), expected)
    # the correction commutes with the change of basis
    basis = {product: windows[0] for product, windows in survey["basis"][0].items()}
    corrected_basis = post.correct_selfpairs(basis["window"], basis["shotnoise_jacobian"], basis["data_shotnoise_jacobian"], template)
    assert np.allclose(np.asarray(corrected_basis.value()), np.asarray(corrected.value()) @ survey["basis_vectors"].T)
    with pytest.raises(ValueError):
        post.correct_selfpairs(window, products["shotnoise_jacobian"], products["data_shotnoise_jacobian"], template.select(k=(0.0, 0.05)))


def test_correct_selfpairs_analytic(survey):
    rng = np.random.default_rng(1)
    window = survey["spikes"][0]["window"][0]
    template = _white_noise_template(window.observable, rng)
    unclustered = survey["unclustered_data_shotnoise"]
    corrected = post.correct_selfpairs_analytic(window, template, unclustered, survey["mattrs"])
    variance_derivative = post.get_density_variance_derivative(survey["theory"], survey["mattrs"])
    monopole = np.concatenate([np.full(window.observable.get(ell).size, float(ell == 0)) for ell in window.observable.ells])
    expected = np.asarray(window.value()) - np.outer(np.asarray(template.value()) - monopole, unclustered * variance_derivative)
    assert np.allclose(np.asarray(corrected.value()), expected)
    corrected_basis = post.correct_selfpairs_analytic(survey["basis"][0]["window"][0], template, unclustered, survey["mattrs"])
    assert np.allclose(np.asarray(corrected_basis.value()), np.asarray(corrected.value()) @ survey["basis_vectors"].T)


def test_apply_control_variate(survey):
    rng = np.random.default_rng(2)
    effect = survey["spikes"][0]["window"][0]
    geometry = effect.clone(value=np.asarray(effect.value()) + rng.normal(size=effect.value().shape))
    analytic = effect.clone(value=rng.normal(size=effect.value().shape))
    cv = post.apply_control_variate(effect, geometry, analytic)
    assert np.allclose(np.asarray(cv.value()), np.asarray(analytic.value()) + np.asarray(effect.value()) - np.asarray(geometry.value()))
    # basis windows: identity mapping with the same vectors, error otherwise
    vectors = survey["basis_vectors"]
    cv_basis = post.apply_control_variate(project_window(effect, vectors), project_window(geometry, vectors), project_window(analytic, vectors))
    assert np.allclose(np.asarray(cv_basis.value()), np.asarray(cv.value()) @ vectors.T)
    with pytest.raises(ValueError):
        post.apply_control_variate(project_window(effect, vectors), project_window(geometry, vectors), project_window(analytic, vectors[:2]))
    # finer analytic theory bins: spline mapping, or an error with "exact"
    edges = np.arange(0.0, 0.155, 0.01)
    fine = lss.Mesh2SpectrumPoles(
        [
            lss.Mesh2SpectrumPole(k=(edges[:-1] + edges[1:]) / 2, k_edges=np.stack([edges[:-1], edges[1:]], -1), num_raw=np.ones(edges.size - 1), ell=ell)
            for ell in (0, 2, 4)
        ],
        ells=[0, 2, 4],
    )
    analytic_fine = lss.WindowMatrix(value=rng.normal(size=(effect.value().shape[0], fine.value().size)), observable=effect.observable, theory=fine)
    cv_fine = post.apply_control_variate(effect, geometry, analytic_fine)
    assert cv_fine.value().shape == analytic_fine.value().shape

    # the mapped difference gives the same prediction for a smooth theory (a polynomial, reproduced by the cubic spline)
    def polynomial(theory):
        k = np.asarray(theory.get(0).coords("k"))
        return np.concatenate([1e4 * (1 + 3 * k - 20 * k**2), 2e3 * (1 - k), 5e2 * np.ones_like(k)])

    zeros = analytic_fine.clone(value=np.zeros(analytic_fine.value().shape))
    cv_difference = post.apply_control_variate(effect, geometry, zeros)
    expected = (np.asarray(effect.value()) - np.asarray(geometry.value())) @ polynomial(survey["theory"])
    assert np.allclose(np.asarray(cv_difference.value()) @ polynomial(fine), expected, rtol=1e-8, atol=1e-8 * np.abs(expected).max())
    with pytest.raises(ValueError):
        post.apply_control_variate(effect, geometry, analytic_fine, theory_interpolation="exact")


def test_average_windows(survey):
    realizations = [windows[0] for windows in survey["spikes"][1]["window"]]
    assert np.allclose(np.asarray(post.average_windows(realizations).value()), np.asarray(survey["spikes"][0]["window"][0].value()))
