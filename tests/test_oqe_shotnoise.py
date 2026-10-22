"""
Shot-noise Jacobians with OQE-like estimator weights: one spectrum per multipole, crossing the same particles with two weights.

As in the OQE windows of the validation scripts, the ell = 0 spectrum crosses ``w_tilde x w_0`` and the ell = 2 one
``w_tilde x w_2``, each on its own (binner with a single multipole), with ``fields=(0, 0)`` since both legs are the same particles.
The data-only shot noise of the self-pair correction is taken from the ell = 0 run; the ell = 2 run must still run, and its
estimated shot noise is zero (jaxpower assigns shot noise to the monopole only).
"""

import jax

jax.config.update("jax_enable_x64", True)

import lsstypes as lss
import numpy as np
import pytest
from jaxpower import BinMesh2SpectrumPoles, FKPField, ParticleField, compute_fkp2_normalization, get_mesh_attrs

from desiwinds import post
from desiwinds.forward import _get_data_shotnoise, mock_survey_catalog
from desiwinds.window import get_window_spikes

N_PARTICLES = 20000
WEIGHTS = ("w_tilde", "w_0", "w_2")


@pytest.fixture(scope="module")
def survey():
    rng = np.random.default_rng(7)

    def ball(n):
        x = rng.normal(size=(n, 3))
        x *= (150 * rng.uniform(size=n) ** (1 / 3) / np.linalg.norm(x, axis=1))[:, None]
        return x + np.array([800.0, 0.0, 0.0])

    data_positions, randoms_positions = ball(N_PARTICLES), ball(N_PARTICLES)
    mattrs = get_mesh_attrs(data_positions, randoms_positions, cellsize=20.0, check=True)
    # smooth, positive estimator weights depending on the distance, as OQE weights depend on redshift
    extra = {
        positions_name: {name: 0.5 + rng.uniform() + 0.5 * np.linalg.norm(positions, axis=1) / 1000.0 for name in WEIGHTS}
        for positions_name, positions in (("data", data_positions), ("randoms", randoms_positions))
    }
    data = ParticleField(data_positions, np.ones(N_PARTICLES), attrs=mattrs, extra=extra["data"])
    randoms = ParticleField(randoms_positions, np.ones(N_PARTICLES), attrs=mattrs, extra=extra["randoms"])
    fkp = FKPField(data, randoms, attrs=mattrs)

    def weighted(name):
        return fkp.clone(
            data=fkp.data.clone(weights=fkp.data.weights * fkp.data.extra[name]),
            randoms=fkp.randoms.clone(weights=fkp.randoms.weights * fkp.randoms.extra[name]),
        )

    binners, norms = {}, {}
    for ell in (0, 2):
        binners[ell] = BinMesh2SpectrumPoles(mattrs, edges={"step": 0.02}, ells=(ell,))
        norms[ell] = compute_fkp2_normalization(weighted("w_tilde"), weighted(f"w_{ell}"), bin=binners[ell], cellsize=20.0, fields=(0, 0))

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

    def mock_survey_ell(ell):
        def mock_survey(theory, seed, return_data_shotnoise=False):
            return mock_survey_catalog(
                (fkp, fkp),
                theory=theory,
                seed=seed,
                los="local",
                unitary_amplitude=True,
                ric_args=None,
                amr_args=None,
                nam_args=None,
                binner=binners[ell],
                fkp_norms=[norms[ell]],
                estimator_weights=("w_tilde", f"w_{ell}"),
                fields=(0, 0),
                return_data_shotnoise=return_data_shotnoise,
            )

        return mock_survey

    pair_sums = {ell: float(np.sum(extra["data"]["w_tilde"] * extra["data"][f"w_{ell}"])) for ell in (0, 2)}
    return dict(mattrs=mattrs, fkp=fkp, weighted=weighted, binners=binners, norms=norms, theory=theory, mock_survey_ell=mock_survey_ell, pair_sums=pair_sums)


@pytest.mark.parametrize("ell", [0, 2])
def test_data_shotnoise_without_monopole(survey, ell):
    """The data-only shot noise is the data self-pair sum of the pair weights over the normalization, with or without a monopole."""
    fields = (survey["weighted"]("w_tilde"), survey["weighted"](f"w_{ell}"))
    value = float(_get_data_shotnoise(*fields, fkp_norm=survey["norms"][ell], binner=survey["binners"][ell], fields=(0, 0)))
    assert value == pytest.approx(survey["pair_sums"][ell] / float(np.ravel(survey["norms"][ell])[0]), rel=1e-10)


def test_oqe_quadrupole_run(survey):
    """The ell = 2 run computes the Jacobians: zero estimated shot noise, a finite data-only shot noise."""
    windows = get_window_spikes(survey["mock_survey_ell"](2), survey["theory"], nreal=1, batch_size=4, return_shotnoise_jacobians=True)[0]
    assert np.all(np.asarray(windows["shotnoise_jacobian"][0].value()) == 0.0)
    data_shotnoise = np.ravel(windows["data_shotnoise_jacobian"][0].value())
    assert np.all(np.isfinite(data_shotnoise)) and np.any(data_shotnoise != 0.0)
    assert np.all(np.isfinite(np.asarray(windows["window"][0].value())))


def test_oqe_monopole_data_shotnoise_jacobian(survey):
    """With pair weights w_tilde x w_0, the jvp of the data self-pair noise matches sum(w_tilde w_0) / norm x d var(delta) / d theta."""
    theory = survey["theory"]
    windows = get_window_spikes(survey["mock_survey_ell"](0), theory, nreal=12, batch_size=4, return_shotnoise_jacobians=True)[0]
    measured = np.ravel(windows["data_shotnoise_jacobian"][0].value())
    unclustered = survey["pair_sums"][0] / float(np.ravel(survey["norms"][0])[0])
    expected = unclustered * post.get_density_variance_derivative(theory, survey["mattrs"])
    nk = theory.get(0).size
    high = slice(3, nk)  # the lowest bins have few modes and noisy jvps
    assert np.sum(measured[high]) == pytest.approx(np.sum(expected[high]), rel=0.1)
