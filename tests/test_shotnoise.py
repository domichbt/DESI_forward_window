"""
Tests for :py:mod:`desiwinds.shotnoise` on small synthetic surveys (:py:mod:`toy_survey`).

They check, in float64:

* the machinery: the estimator ``Q`` is the one used by the forward model, the JVPs are the derivatives of the pipeline (V3), linear pipelines have no curvature and no sigma dependence (V2), option A tends to option C when sigma -> 0,
* the analytic template (V1, V4): averaging option B over a *Hadamard* set of Rademacher draws (exactly orthogonal, so the Monte Carlo average is the exact expectation) reproduces it,
* the statistics helpers and the moved ``mock_whitenoise``.
"""

from functools import partial

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import hadamard
from toy_survey import make_toy

from desiwinds import forward, shotnoise
from desiwinds.shotnoise import (
    analytic_shotnoise_template,
    shotnoise_template_with_control_variate,
    combine_regions_weighted_by_normalization,
    conventional_shotnoise,
    shotnoise_template_curvature_shift,
    draw_relative_weight_noise,
    get_estimator_normalizations,
    get_region_particles,
    mock_whitenoise,
    apply_field_weights,
    field_weights_perturbation_gic_only,
    prepare_field_weights,
    jackknife_ratio_of_means,
    make_realization_keys,
    sample_shotnoise_template_antithetic,
    sample_shotnoise_template_linearized,
    sample_shotnoise_template_quadratic,
    bilinear_power_spectrum,
    select_realizations,
    shotnoise_template_from_samples,
    shotnoise_template_uncertainty,
    measure_power_spectrum_and_shotnoise,
)

EFFECTS = {
    "geometry": {"gic": False},
    "gic": {},
    "ric": {"ric": True},
    "ric+amr": {"ric": True, "amr": True},
    "all": {"ric": True, "amr": True, "nam": True},
}


@pytest.fixture(scope="module")
def toy():
    return make_toy(n_data=60, n_randoms=240, seed=1, ells=(0, 2, 4))


@pytest.fixture(scope="module")
def toy2():
    return make_toy(n_data=50, n_randoms=200, seed=2, ells=(0, 2), n_regions=2)


@pytest.fixture(scope="module")
def tiny():
    return make_toy(n_data=16, n_randoms=300, seed=4, ells=(0, 2, 4), meshsize=16)


def kwargs(toy, effects="gic"):
    """Keyword arguments of the ``sample_shotnoise_template_*`` functions."""
    return toy.kwargs(**EFFECTS[effects])


def parts(toy, effects="gic"):
    """``input_data_weights, apply_field_weights(u), field_weights_perturbation_gic_only(relative_noise)``, ``bilinear_power_spectrum(w, z=None), conventional_shotnoise(w, z=None)`` (arguments bound) and the full keyword arguments."""
    kw = kwargs(toy, effects)
    args = kw["field_weights_args"]
    norms = get_estimator_normalizations(kw["fkp_norms"])
    spectrum_of = partial(bilinear_power_spectrum, particles=get_region_particles(*toy.fkp_fields), binner=kw["binner"], norms=norms)
    shotnoise_of = partial(conventional_shotnoise, binner=kw["binner"], norms=norms)
    return (args.input_data_weights, partial(apply_field_weights, field_weights_args=args), partial(field_weights_perturbation_gic_only, field_weights_args=args)), (spectrum_of, shotnoise_of), kw


def norms_of(kw):
    return np.array([np.ravel(n)[0] for n in kw["fkp_norms"]])


def flat(w):
    return np.concatenate([np.asarray(x) for x in w])


KEY = jax.random.key(0)


# ---------------------------------------------------------------------------
# Machinery
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("effects", ["geometry", "gic", "ric", "ric+amr", "all"])
def test_Q_is_the_forward_model_estimator(toy, effects):  # noqa: N802
    """Q(noise_free_field_weights) is mock_whitenoise at sigma=0 with the shot noise added back; S(noise_free_field_weights) is that shot noise."""
    kw = kwargs(toy, effects)
    forward_kw = toy.forward_kwargs(**EFFECTS[effects])
    legacy = mock_whitenoise(*toy.fkp_fields, sigma=jnp.array([0.0]), seed=jax.random.key(0), los="local", **forward_kw)[0]
    q, s = measure_power_spectrum_and_shotnoise(*toy.fkp_fields, **kw)
    q, s = np.asarray(q)[0], float(s[0])
    for iell, ell in enumerate(kw["binner"].ells):
        expected = np.asarray(legacy.get(ell).value()) + (s if ell == 0 else 0.0)
        np.testing.assert_allclose(q[iell], expected, rtol=1e-8, atol=1e-8 * abs(s))


def test_Q_symmetric_and_bilinear(toy):  # noqa: N802
    (input_data_weights, phi_u, _), (spectrum_of, shotnoise_of), _ = parts(toy, "ric")
    noise_free_field_weights = phi_u(input_data_weights)
    rng = np.random.default_rng(0)
    a = tuple(jnp.asarray(rng.normal(size=w.shape)) for w in noise_free_field_weights)
    b = tuple(jnp.asarray(rng.normal(size=w.shape)) for w in noise_free_field_weights)
    qab, qba = spectrum_of(a, b), spectrum_of(b, a)
    np.testing.assert_allclose(qab, qba, rtol=1e-10, atol=1e-6)
    # Q(a + b) = Q(a) + 2 Q(a, b) + Q(b)
    ab = tuple(x + y for x, y in zip(a, b, strict=True))
    qa, qb = spectrum_of(a), spectrum_of(b)
    np.testing.assert_allclose(spectrum_of(ab), qa + 2 * qab + qb, rtol=1e-9, atol=1e-6 * float(jnp.abs(qa).max()))
    np.testing.assert_allclose(shotnoise_of(ab), shotnoise_of(a) + 2 * shotnoise_of(a, b) + shotnoise_of(b), rtol=1e-10)


@pytest.mark.parametrize("effects", ["geometry", "gic", "ric", "ric+amr", "all"])
def test_derivatives_match_finite_differences(toy, effects):
    """V3: v = J d and z = Hess[d, d] against central finite differences."""
    (input_data_weights, phi_u, _), _, _ = parts(toy, effects)
    d = input_data_weights * draw_relative_weight_noise(jax.random.key(3), input_data_weights.shape)
    v, z = jax.jvp(lambda x: jax.jvp(phi_u, (x,), (d,))[1], (input_data_weights,), (d,))
    _, v1 = jax.jvp(phi_u, (input_data_weights,), (d,))
    np.testing.assert_allclose(flat(v), flat(v1), rtol=1e-12, atol=1e-12)

    best_v, best_z = np.inf, np.inf
    noise_free_field_weights = flat(phi_u(input_data_weights))
    scale = np.abs(noise_free_field_weights).max()
    for h in (1e-3, 1e-4, 1e-5, 1e-6):
        wp, wm = flat(phi_u(input_data_weights + h * d)), flat(phi_u(input_data_weights - h * d))
        best_v = min(best_v, np.abs((wp - wm) / (2 * h) - flat(v)).max() / scale)
        best_z = min(best_z, np.abs((wp - 2 * noise_free_field_weights + wm) / h**2 - flat(z)).max() / scale)
    assert best_v < 1e-6
    assert best_z < 1e-3
    if effects == "geometry":
        np.testing.assert_array_equal(flat(z), 0.0)


def test_linear_pipeline_has_no_curvature_and_no_sigma_dependence(toy):
    """V2: gic frozen, no effects: z = 0, Y = X per realization, and option A is exactly sigma^2 times B (so the ratio is sigma independent)."""
    kw = kwargs(toy, "geometry")
    norms = norms_of(kw)
    c = sample_shotnoise_template_quadratic(*toy.fkp_fields, key=KEY, n_real=3, **kw)
    np.testing.assert_allclose(c["power_spectrum_response"], c["extras"]["linear_part"][0], rtol=1e-12)
    np.testing.assert_allclose(c["shotnoise_response"], c["extras"]["linear_part"][1], rtol=1e-12)
    np.testing.assert_allclose(c["extras"]["curvature_part"][0], 0.0, atol=1e-8 * np.abs(c["power_spectrum_response"]).max())

    b = sample_shotnoise_template_linearized(*toy.fkp_fields, key=KEY, n_real=3, **kw)
    for sigma in (0.3, 1.0):
        a = sample_shotnoise_template_antithetic(*toy.fkp_fields, key=KEY, n_real=3, sigma=sigma, **kw)
        scale = np.abs(b["power_spectrum_response"]).max()
        np.testing.assert_allclose(a["power_spectrum_response"], sigma**2 * b["power_spectrum_response"], rtol=1e-6, atol=1e-8 * scale)
        np.testing.assert_allclose(a["shotnoise_response"], sigma**2 * b["shotnoise_response"], rtol=1e-6)
        np.testing.assert_allclose(shotnoise_template_from_samples(a, norms)[0], shotnoise_template_from_samples(b, norms)[0], rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("effects", ["gic", "ric+amr"])
def test_option_a_tends_to_option_c_per_realization(toy, effects):
    """Per draw, (P(+) + P(-))/2 - P(0) = sigma^2 Y + O(sigma^4): option A at small sigma equals option C realization by realization."""
    kw = kwargs(toy, effects)
    sigma = 1e-3
    a = sample_shotnoise_template_antithetic(*toy.fkp_fields, key=KEY, n_real=2, sigma=sigma, **kw)
    c = sample_shotnoise_template_quadratic(*toy.fkp_fields, key=KEY, n_real=2, **kw)
    scale = np.abs(c["power_spectrum_response"]).max()
    np.testing.assert_allclose(a["power_spectrum_response"] / sigma**2, c["power_spectrum_response"], rtol=1e-3, atol=1e-3 * scale)
    np.testing.assert_allclose(a["shotnoise_response"] / sigma**2, c["shotnoise_response"], rtol=1e-3)


def test_option_a_sigma_scaling(toy):
    """V11: with curvature present, A differs from C by O(sigma^2)."""
    kw = kwargs(toy, "ric+amr")
    norms = norms_of(kw)
    c = shotnoise_template_from_samples(sample_shotnoise_template_quadratic(*toy.fkp_fields, key=KEY, n_real=4, **kw), norms)[0]
    errs = [np.abs(shotnoise_template_from_samples(sample_shotnoise_template_antithetic(*toy.fkp_fields, key=KEY, n_real=4, sigma=s, **kw), norms)[0] - c).max() for s in (0.5, 0.25)]
    assert errs[1] < errs[0]  # smaller sigma is closer to the exact quadratic result
    assert errs[1] < 0.6 * errs[0] or errs[1] < 1e-3


def test_option_c_decomposition_and_curvature(toy):
    kw = kwargs(toy, "ric+amr")
    norms = norms_of(kw)
    c = sample_shotnoise_template_quadratic(*toy.fkp_fields, key=KEY, n_real=3, **kw)
    lin_n, lin_d = c["extras"]["linear_part"]
    cur_n, cur_d = c["extras"]["curvature_part"]
    np.testing.assert_allclose(c["power_spectrum_response"], lin_n + cur_n, rtol=1e-12, atol=1e-6)
    np.testing.assert_allclose(c["shotnoise_response"], lin_d + cur_d, rtol=1e-12)
    b = sample_shotnoise_template_linearized(*toy.fkp_fields, key=KEY, n_real=3, **kw)  # common draws
    np.testing.assert_allclose(b["power_spectrum_response"], lin_n, rtol=1e-9, atol=1e-6 * np.abs(lin_n).max())
    delta, cov = shotnoise_template_curvature_shift(c, norms)
    assert delta.shape == (3, kw["binner"].xavg.shape[0]) and cov.shape == (delta.size, delta.size)
    assert np.abs(delta).max() > 0  # RIC+AMR have curvature
    with pytest.raises(ValueError):
        shotnoise_template_curvature_shift(b, norms)


def test_high_k_normalisation(toy):
    """V5, loosely: s_0 -> ~1 at the highest k even though this tiny catalogue has few modes."""
    kw = kwargs(toy, "ric+amr")
    s, _ = shotnoise_template_from_samples(sample_shotnoise_template_linearized(*toy.fkp_fields, key=KEY, n_real=8, **kw), norms_of(kw))
    assert abs(s[0, -1] - 1.0) < 0.3


def test_gaussian_and_rademacher_same_mean_for_B(toy):  # noqa: N802
    """V7 (small): both distributions are unbiased for option B."""
    kw = kwargs(toy, "gic")
    norms = norms_of(kw)
    sr = sample_shotnoise_template_linearized(*toy.fkp_fields, key=KEY, n_real=24, noise_distribution="rademacher", **kw)
    sg = sample_shotnoise_template_linearized(*toy.fkp_fields, key=KEY, n_real=24, noise_distribution="gaussian", **kw)
    s1, s2 = shotnoise_template_from_samples(sr, norms)[0][0], shotnoise_template_from_samples(sg, norms)[0][0]
    err = np.sqrt(shotnoise_template_uncertainty(sr, norms)[0] ** 2 + shotnoise_template_uncertainty(sg, norms)[0] ** 2)
    assert np.all(np.abs(s1 - s2) < 6 * err + 1e-3)


def test_batch_size_does_not_change_results(toy):
    kw = kwargs(toy, "ric")
    a = sample_shotnoise_template_linearized(*toy.fkp_fields, key=KEY, n_real=4, **kw)
    b = sample_shotnoise_template_linearized(*toy.fkp_fields, key=KEY, n_real=4, batch_size=2, **kw)
    np.testing.assert_allclose(a["power_spectrum_response"], b["power_spectrum_response"], rtol=1e-10)


def test_realizations_do_not_depend_on_n_real(toy):
    """Common random numbers: realization r is ``fold_in(key, r)``, whatever the number of realizations."""
    kw = kwargs(toy, "ric")
    a = sample_shotnoise_template_linearized(*toy.fkp_fields, key=KEY, n_real=3, **kw)
    b = sample_shotnoise_template_linearized(*toy.fkp_fields, key=KEY, n_real=5, **kw)
    np.testing.assert_allclose(a["power_spectrum_response"], b["power_spectrum_response"][:3], rtol=1e-12)


# ---------------------------------------------------------------------------
# Analytic template
# ---------------------------------------------------------------------------


def hadamard_mean(toy, effects, response):
    """Exact expectation over Rademacher draws: rows of a Hadamard matrix are orthogonal, so (1/n) sum_r eps_r eps_r^T = 1."""
    (input_data_weights, phi_u, phi_linear), (spectrum_of, shotnoise_of), _ = parts(toy, effects)
    n = len(input_data_weights)
    assert n & (n - 1) == 0, "Hadamard needs a power of two"
    num, den = [], []
    for row in hadamard(n):
        relative_noise = jnp.asarray(row, dtype=input_data_weights.dtype)
        v = phi_linear(relative_noise) if response == "linear_gic" else jax.jvp(phi_u, (input_data_weights,), (input_data_weights * relative_noise,))[1]
        num.append(spectrum_of(v))
        den.append(shotnoise_of(v))
    return np.mean(num, axis=0), np.mean(den, axis=0)


def test_analytic_geometry_term_matches_mode_sum(tiny):
    """The spherical-harmonic geometry term equals the explicit mean of L_ell(k.x) over the modes in each bin (no mass assignment involved)."""
    from jaxpower.utils import get_legendre

    kw = kwargs(tiny, "geometry")
    binner = kw["binner"]
    t, s, _ = analytic_shotnoise_template(*tiny.fkp_fields, **kw, include_gic=False)
    t, s = np.asarray(t), np.asarray(s)
    fkp = tiny.fkp_fields[0]
    squared_data_weights = np.asarray(fkp.data.weights * fkp.data.extra["weight_FKP"]) ** 2
    x = np.asarray(fkp.data.positions)
    xhat = x / np.linalg.norm(x, axis=1)[:, None]
    kvec = binner.mattrs.kcoords(sparse=False)
    kv = np.stack([np.asarray(kk) for kk in kvec], -1).reshape(-1, 3)
    knorm = np.linalg.norm(kv, axis=1)
    khat = np.where(knorm[:, None] > 0, kv / np.where(knorm == 0, 1, knorm)[:, None], 0.0)
    ibin = np.asarray(binner.ibin[0])
    nmodes = np.asarray(binner.nmodes)
    mu = khat @ xhat.T  # (modes, objects)
    norm = float(np.ravel(kw["fkp_norms"][0])[0])
    for iell, ell in enumerate(binner.ells):
        if ell == 0:
            continue
        per_mode = np.array((get_legendre(ell)(mu) * squared_data_weights).sum(1) * (2 * ell + 1))
        per_mode[knorm == 0] = 0.0
        binned = np.bincount(ibin, weights=per_mode, minlength=len(nmodes) + 2)[1 : len(nmodes) + 1]  # jaxpower's bins are offset by one (under/overflow)
        np.testing.assert_allclose(t[0, iell], binned / nmodes / norm, rtol=1e-6, atol=1e-8 * np.abs(t[0, 0]).max())
    np.testing.assert_allclose(s[0], squared_data_weights.sum() / norm)


@pytest.mark.parametrize(("include_gic", "effects"), [(False, "geometry"), (True, "gic")])
def test_option_b_expectation_matches_analytic(tiny, include_gic, effects):
    """V1/V4: exact Rademacher expectation of option B (Hadamard set) equals the analytic template, up to mass-assignment aliasing of the self-pair term."""
    kw = kwargs(tiny, effects)
    t, s, _ = analytic_shotnoise_template(*tiny.fkp_fields, **kw, include_gic=include_gic)
    t, s = np.asarray(t), np.asarray(s)
    num, den = hadamard_mean(tiny, effects, "jvp")
    scale = np.abs(t[0, 0]).max()
    low_k = slice(0, t.shape[-1] // 2)
    np.testing.assert_allclose(num[0][:, low_k], t[0][:, low_k], atol=0.05 * scale)
    np.testing.assert_allclose(den[0], s[0], rtol=1e-10)


def test_linear_gic_closed_form_matches_jvp(tiny):
    """For a pipeline without effects, J_A d is exactly the JVP of the pipeline."""
    (input_data_weights, phi_u, phi_linear), _, _ = parts(tiny, "gic")
    relative_noise = draw_relative_weight_noise(jax.random.key(9), input_data_weights.shape)
    _, v = jax.jvp(phi_u, (input_data_weights,), (input_data_weights * relative_noise,))
    np.testing.assert_allclose(flat(phi_linear(relative_noise)), flat(v), rtol=1e-10, atol=1e-10)


def test_control_variate_is_exact_when_pipeline_is_the_control(tiny):
    """With no effects the CV pair equals the estimator pair, so the CV-corrected template is the analytic one with zero variance."""
    kw = kwargs(tiny, "gic")
    norms = norms_of(kw)
    res = sample_shotnoise_template_linearized(*tiny.fkp_fields, key=KEY, n_real=6, compute_control_variate=True, **kw)
    np.testing.assert_allclose(res["extras"]["control_variate"][0], res["power_spectrum_response"], rtol=1e-9, atol=1e-6)
    np.testing.assert_allclose(res["extras"]["control_variate"][1], res["shotnoise_response"], rtol=1e-12)
    analytic_spectrum_response, analytic_shotnoise_response, analytic = analytic_shotnoise_template(*tiny.fkp_fields, **kw, include_gic=True)
    s, cov, diag = shotnoise_template_with_control_variate(res, norms, analytic_spectrum_response, analytic_shotnoise_response)
    np.testing.assert_allclose(s, np.asarray(analytic)[0], rtol=1e-8, atol=1e-8)
    assert np.allclose(np.diag(cov), 0, atol=1e-12)
    np.testing.assert_allclose(diag["variance_reduction"], 0.0, atol=1e-8)


def test_control_variate_with_effects_runs_and_fit(toy):
    kw = kwargs(toy, "ric+amr")
    norms = norms_of(kw)
    res = sample_shotnoise_template_linearized(*toy.fkp_fields, key=KEY, n_real=8, compute_control_variate=True, **kw)
    analytic_spectrum_response, analytic_shotnoise_response, _ = analytic_shotnoise_template(*toy.fkp_fields, **kw, include_gic=True)
    s1, c1, _ = shotnoise_template_with_control_variate(res, norms, analytic_spectrum_response, analytic_shotnoise_response, coefficient=1.0)
    s2, c2, diag = shotnoise_template_with_control_variate(res, norms, analytic_spectrum_response, analytic_shotnoise_response, coefficient="fit")
    plain, cp = shotnoise_template_from_samples(res, norms)
    assert s1.shape == s2.shape == plain.shape
    assert np.all(diag["variance_reduction"] <= 1 + 1e-9)
    # the fitted coefficient reduces the variance (in the least-squares sense) of the numerator
    assert np.trace(c2) <= np.trace(cp) * 1.05
    with pytest.raises(ValueError):
        shotnoise_template_with_control_variate(sample_shotnoise_template_linearized(*toy.fkp_fields, key=KEY, n_real=2, **kw), norms, analytic_spectrum_response, analytic_shotnoise_response)


def test_subset_keeps_extras(toy):
    kw = kwargs(toy, "ric")
    norms = norms_of(kw)
    res = sample_shotnoise_template_quadratic(*toy.fkp_fields, key=KEY, n_real=4, **kw)
    sub = select_realizations(res, [0, 2])
    assert len(sub["power_spectrum_response"]) == 2
    np.testing.assert_array_equal(sub["power_spectrum_response"], np.asarray(res["power_spectrum_response"])[[0, 2]])
    np.testing.assert_array_equal(sub["extras"]["linear_part"][1], np.asarray(res["extras"]["linear_part"][1])[[0, 2]])
    np.testing.assert_allclose(shotnoise_template_from_samples(select_realizations(res, np.arange(4)), norms)[0], shotnoise_template_from_samples(res, norms)[0])


def test_two_regions(toy2):
    """Several regions: per-region outputs, combination, and finite results."""
    kw = toy2.kwargs(ric=True, amr=True)
    norms = norms_of(kw)
    res = sample_shotnoise_template_quadratic(*toy2.fkp_fields, key=KEY, n_real=3, **kw)
    n_k = kw["binner"].xavg.shape[0]
    assert res["power_spectrum_response"].shape == (3, 2, 2, n_k)
    assert res["shotnoise_response"].shape == (3, 2)
    for regions in (None, [0], [1]):
        s, cov = shotnoise_template_from_samples(res, norms, regions)
        assert np.all(np.isfinite(s)) and np.all(np.isfinite(cov))
    # combining a single region is that region
    np.testing.assert_allclose(shotnoise_template_from_samples(res, norms, [1])[0], jackknife_ratio_of_means(res["power_spectrum_response"][:, 1], res["shotnoise_response"][:, 1])[0])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_ratio_jackknife_against_bruteforce():
    rng = np.random.default_rng(0)
    n = rng.normal(5, 1, (12, 2, 3))
    d = rng.normal(4, 0.5, 12)
    ratio, cov = jackknife_ratio_of_means(n, d)
    np.testing.assert_allclose(ratio, n.sum(0) / d.sum())
    loo = np.array([(n.sum(0) - n[i]) / (d.sum() - d[i]) for i in range(12)]).reshape(12, -1)
    expected = 11 / 12 * (loo - loo.mean(0)).T @ (loo - loo.mean(0))
    np.testing.assert_allclose(cov, expected, rtol=1e-12)
    # jackknife of a mean of iid samples is the usual error on the mean
    x = rng.normal(0, 2.0, (4000, 1))
    _, c = jackknife_ratio_of_means(x + 10, np.ones(4000))
    np.testing.assert_allclose(np.sqrt(c[0, 0]), 2.0 / np.sqrt(4000), rtol=0.05)
    with pytest.raises(ValueError):
        jackknife_ratio_of_means(n[:1], d[:1])


def test_ratio_is_ratio_of_means_not_mean_of_ratios():
    n, d = np.array([[1.0], [100.0]]), np.array([1.0, 100.0])
    ratio, _ = jackknife_ratio_of_means(n, d)
    assert ratio[0] == pytest.approx(1.0)


def test_combine_regions_weights_by_norm():
    n = np.arange(2 * 2 * 1 * 3, dtype=float).reshape(2, 2, 1, 3) + 1
    d = np.array([[1.0, 2.0], [3.0, 4.0]])
    norms = np.array([1.0, 3.0])
    cn, cd = combine_regions_weighted_by_normalization(n, d, norms)
    np.testing.assert_allclose(cn, n[:, 0] * 1 + n[:, 1] * 3)
    np.testing.assert_allclose(cd, d[:, 0] + 3 * d[:, 1])
    cn1, _ = combine_regions_weighted_by_normalization(n, d, norms, [1])
    np.testing.assert_allclose(cn1, 3 * n[:, 1])


def test_noise_draws_reproducible_and_unit_variance():
    keys = make_realization_keys(KEY, 5)
    np.testing.assert_array_equal(jax.random.key_data(keys[:3]), jax.random.key_data(make_realization_keys(KEY, 3)))
    np.testing.assert_array_equal(draw_relative_weight_noise(keys[4], (50,)), draw_relative_weight_noise(make_realization_keys(KEY, 5)[4], (50,)))
    assert not np.array_equal(draw_relative_weight_noise(keys[4], (50,)), draw_relative_weight_noise(keys[3], (50,)))
    r = np.asarray(draw_relative_weight_noise(keys[0], (20000,), "rademacher"))
    assert set(np.unique(r)) == {-1.0, 1.0}
    assert abs(r.mean()) < 0.03 and abs(r.var() - 1) < 1e-3
    g = np.asarray(draw_relative_weight_noise(keys[0], (20000,), "gaussian"))
    assert abs(g.mean()) < 0.05 and abs(g.var() - 1) < 0.05
    with pytest.raises(ValueError):
        draw_relative_weight_noise(keys[0], (3,), "uniform")


def test_mock_whitenoise_moved_and_reexported():
    assert forward.mock_whitenoise is shotnoise.mock_whitenoise
    with pytest.raises(AttributeError):
        forward.does_not_exist  # noqa: B018


def test_mock_whitenoise_runs_on_toy(toy):
    pk = mock_whitenoise(*toy.fkp_fields, sigma=jnp.array([1.0]), seed=jax.random.key(0), los="local", **toy.forward_kwargs(ric=True, amr=True))
    assert len(pk) == 1 and np.all(np.isfinite(np.asarray(pk[0].value())))


def test_cross_correlations_rejected(toy):
    with pytest.raises(NotImplementedError):
        prepare_field_weights((toy.fkp_fields[0], toy.fkp_fields[0]))


# ---------------------------------------------------------------------------
# Two estimator weightings (OQE leg pair)
# ---------------------------------------------------------------------------


def two_leg_kwargs(toy, estimator_weights, effects="gic"):
    """Like ``kwargs`` with ``estimator_weights`` forwarded to ``prepare_field_weights``."""
    effects_kw = dict(EFFECTS[effects])
    gic = effects_kw.pop("gic", True)
    kw = toy.forward_kwargs(**effects_kw, estimator_weights=estimator_weights)
    args = prepare_field_weights(*toy.fkp_fields, gic=gic, **{k: v for k, v in kw.items() if k not in ("binner", "fkp_norms")})
    return {"field_weights_args": args, "binner": toy.binner, "fkp_norms": toy.fkp_norms}


def with_second_weights(toy):
    """The toy with a second estimator weighting, ``weight_B``, on the data and randoms."""
    rng = np.random.default_rng(11)
    fkp_fields = tuple(
        f.clone(
            data=f.data.clone(extra=f.data.extra | {"weight_B": jnp.asarray(rng.uniform(0.4, 1.6, f.data.size))}),
            randoms=f.randoms.clone(extra=f.randoms.extra | {"weight_B": jnp.asarray(rng.uniform(0.4, 1.6, f.randoms.size))}),
        )
        for f in toy.fkp_fields
    )
    from dataclasses import replace

    return replace(toy, fkp_fields=fkp_fields)


def test_single_leg_structure(toy):
    """(a) A single string (or None) gives one leg, with the same structure as before."""
    for name in ("weight_FKP", None):
        args = prepare_field_weights(*toy.fkp_fields, estimator_weights=name)
        assert args.n_legs == 1 and args.other_data_estimator_weights is None
        w = apply_field_weights(args.input_data_weights, args)
        assert isinstance(w, tuple) and all(hasattr(x, "shape") for x in w)
        # frozen alpha is the ratio of the estimator-weighted sums
        f = toy.fkp_fields[0]
        d, r = (f.data.weights, f.randoms.weights) if name is None else (f.data.weights * f.data.extra[name], f.randoms.weights * f.randoms.extra[name])
        np.testing.assert_allclose(args.noise_free_data_to_randoms_ratio[0], d.sum() / r.sum(), rtol=1e-14)
    # the quantities of a single-leg run are those of the explicit formula d.w, -alpha r.w
    args = prepare_field_weights(*toy.fkp_fields, estimator_weights="weight_FKP")
    f = toy.fkp_fields[0]
    d, r = f.data.weights * f.data.extra["weight_FKP"], f.randoms.weights * f.randoms.extra["weight_FKP"]
    np.testing.assert_allclose(apply_field_weights(args.input_data_weights, args)[0], np.concatenate([d, -(d.sum() / r.sum()) * r]), rtol=1e-12)


def test_estimator_weights_tuple_validation(toy):
    with pytest.raises(ValueError):
        prepare_field_weights(*toy.fkp_fields, estimator_weights=("weight_FKP",) * 3)


@pytest.mark.parametrize("effects", ["geometry", "gic", "ric+amr"])
@pytest.mark.parametrize("sampler", [sample_shotnoise_template_linearized, sample_shotnoise_template_quadratic])
def test_two_legs_with_equal_weights_equal_single_leg(toy, effects, sampler):
    """(b) With w_A == w_B the two-leg run is the single-leg run."""
    single = sampler(*toy.fkp_fields, key=KEY, n_real=3, compute_control_variate=True, **kwargs(toy, effects))
    double = sampler(*toy.fkp_fields, key=KEY, n_real=3, compute_control_variate=True, **two_leg_kwargs(toy, ("weight_FKP", "weight_FKP"), effects))
    for x, y in zip(jax.tree.leaves(single), jax.tree.leaves(double), strict=True):
        np.testing.assert_allclose(y, x, rtol=1e-9, atol=1e-9 * np.abs(x).max())
    a1 = sample_shotnoise_template_antithetic(*toy.fkp_fields, key=KEY, n_real=2, sigma=0.5, **kwargs(toy, effects))
    a2 = sample_shotnoise_template_antithetic(*toy.fkp_fields, key=KEY, n_real=2, sigma=0.5, **two_leg_kwargs(toy, ("weight_FKP", "weight_FKP"), effects))
    np.testing.assert_allclose(a2["power_spectrum_response"], a1["power_spectrum_response"], rtol=1e-8, atol=1e-8 * np.abs(a1["power_spectrum_response"]).max())
    np.testing.assert_allclose(a2["shotnoise_response"], a1["shotnoise_response"], rtol=1e-9)
    # analytic template and measurement
    for include_gic in (False, True):
        for x, y in zip(
            analytic_shotnoise_template(*toy.fkp_fields, **kwargs(toy, "gic"), include_gic=include_gic),
            analytic_shotnoise_template(*toy.fkp_fields, **two_leg_kwargs(toy, ("weight_FKP", "weight_FKP")), include_gic=include_gic),
            strict=True,
        ):
            np.testing.assert_allclose(y, x, rtol=1e-9, atol=1e-9 * np.abs(x).max())
    q1, s1 = measure_power_spectrum_and_shotnoise(*toy.fkp_fields, **kwargs(toy, effects))
    q2, s2 = measure_power_spectrum_and_shotnoise(*toy.fkp_fields, **two_leg_kwargs(toy, ("weight_FKP", "weight_FKP"), effects))
    np.testing.assert_allclose(q2, q1, rtol=1e-9, atol=1e-9 * np.abs(q1).max())
    np.testing.assert_allclose(s2, s1, rtol=1e-12)


def test_two_none_legs_equal_single_leg_without_gic(toy):
    """With frozen alpha, two unweighted legs are the single unweighted leg (the frozen alpha of a leg is the ratio of its weighted sums)."""
    kw = toy.kwargs(gic=False)
    runs = [
        sample_shotnoise_template_linearized(
            *toy.fkp_fields, key=KEY, n_real=3, **kw | {"field_weights_args": prepare_field_weights(*toy.fkp_fields, estimator_weights=weights, gic=False)}
        )
        for weights in (None, (None, None))
    ]
    np.testing.assert_allclose(runs[1]["power_spectrum_response"], runs[0]["power_spectrum_response"], rtol=1e-9, atol=1e-9)


def test_two_legs_structure_and_frozen_alpha(toy):
    toy_b = with_second_weights(toy)
    args = prepare_field_weights(*toy_b.fkp_fields, estimator_weights=("weight_FKP", "weight_B"), gic=False)
    assert args.n_legs == 2
    w = apply_field_weights(args.input_data_weights, args)
    assert len(w) == 2 and len(w[0]) == 1
    f = toy_b.fkp_fields[0]
    for leg, name in enumerate(("weight_FKP", "weight_B")):
        d, r = f.data.weights * f.data.extra[name], f.randoms.weights * f.randoms.extra[name]
        np.testing.assert_allclose(w[leg][0], np.concatenate([d, -(d.sum() / r.sum()) * r]), rtol=1e-12)
    # the legs are different, and the cross spectrum is symmetric in them
    assert not np.allclose(w[0][0], w[1][0])
    _, (spectrum_of, shotnoise_of), _ = parts(toy_b, "gic")
    np.testing.assert_allclose(spectrum_of(w[0], w[1]), spectrum_of(w[1], w[0]), rtol=1e-10, atol=1e-6)


def test_two_legs_control_variate_is_exact_without_effects(tiny):
    """(c) With no effects the two-leg CV equals the response exactly, so the CV-corrected template is the two-leg analytic one."""
    tiny_b = with_second_weights(tiny)
    kw = two_leg_kwargs(tiny_b, ("weight_FKP", "weight_B"))
    norms = norms_of(kw)
    res = sample_shotnoise_template_linearized(*tiny_b.fkp_fields, key=KEY, n_real=6, compute_control_variate=True, **kw)
    np.testing.assert_allclose(res["extras"]["control_variate"][0], res["power_spectrum_response"], rtol=1e-9, atol=1e-6)
    np.testing.assert_allclose(res["extras"]["control_variate"][1], res["shotnoise_response"], rtol=1e-12)
    analytic_spectrum_response, analytic_shotnoise_response, analytic = analytic_shotnoise_template(*tiny_b.fkp_fields, **kw, include_gic=True)
    s, cov, diag = shotnoise_template_with_control_variate(res, norms, analytic_spectrum_response, analytic_shotnoise_response)
    np.testing.assert_allclose(s, np.asarray(analytic)[0], rtol=1e-8, atol=1e-8)
    assert np.allclose(np.diag(cov), 0, atol=1e-12)
    np.testing.assert_allclose(diag["variance_reduction"], 0.0, atol=1e-8)
    # the two-leg template is not the single-leg one
    single = analytic_shotnoise_template(*tiny_b.fkp_fields, **two_leg_kwargs(tiny_b, ("weight_FKP", "weight_FKP")), include_gic=True)[2]
    assert not np.allclose(np.asarray(single), np.asarray(analytic))


def test_two_legs_quadratic_curvature_decomposition(toy):
    toy_b = with_second_weights(toy)
    kw = two_leg_kwargs(toy_b, ("weight_FKP", "weight_B"), "ric+amr")
    c = sample_shotnoise_template_quadratic(*toy_b.fkp_fields, key=KEY, n_real=3, **kw)
    lin_n, lin_d = c["extras"]["linear_part"]
    cur_n, cur_d = c["extras"]["curvature_part"]
    np.testing.assert_allclose(c["power_spectrum_response"], lin_n + cur_n, rtol=1e-12, atol=1e-6)
    np.testing.assert_allclose(c["shotnoise_response"], lin_d + cur_d, rtol=1e-12)
    assert np.abs(cur_n).max() > 0


def test_two_legs_quadratic_matches_antithetic_small_sigma(toy):
    toy_b = with_second_weights(toy)
    kw = two_leg_kwargs(toy_b, ("weight_FKP", "weight_B"), "ric+amr")
    sigma = 1e-3
    a = sample_shotnoise_template_antithetic(*toy_b.fkp_fields, key=KEY, n_real=2, sigma=sigma, **kw)
    c = sample_shotnoise_template_quadratic(*toy_b.fkp_fields, key=KEY, n_real=2, **kw)
    scale = np.abs(c["power_spectrum_response"]).max()
    np.testing.assert_allclose(a["power_spectrum_response"] / sigma**2, c["power_spectrum_response"], rtol=1e-3, atol=1e-3 * scale)
    np.testing.assert_allclose(a["shotnoise_response"] / sigma**2, c["shotnoise_response"], rtol=1e-3)


def test_two_legs_monte_carlo_matches_analytic(tiny):
    """(d) The mean of the two-leg GIC-only response is the analytic template, within the jackknife errors (the exact Hadamard expectation also matches)."""
    tiny_b = with_second_weights(tiny)
    kw = two_leg_kwargs(tiny_b, ("weight_FKP", "weight_B"))
    norms = norms_of(kw)
    analytic_spectrum_response, analytic_shotnoise_response, analytic = analytic_shotnoise_template(*tiny_b.fkp_fields, **kw, include_gic=True)
    res = sample_shotnoise_template_linearized(*tiny_b.fkp_fields, key=KEY, n_real=256, compute_control_variate=True, **kw)
    # the response to the GIC-only pipeline is the control variate: compare the mean of its spectrum response to the analytic one, bin by bin
    cv_spectrum, cv_shotnoise = (np.asarray(x)[:, 0] for x in res["extras"]["control_variate"])
    mean, error = cv_spectrum.mean(0), cv_spectrum.std(0, ddof=1) / np.sqrt(len(cv_spectrum))
    low_k = slice(0, mean.shape[-1] // 2)  # the analytic self-pair term neglects mass-assignment aliasing at high k
    scale = np.abs(np.asarray(analytic_spectrum_response)[0, 0]).max()
    assert np.all(np.abs(mean - np.asarray(analytic_spectrum_response)[0])[:, low_k] < 5 * error[:, low_k] + 0.05 * scale)
    assert abs(cv_shotnoise.mean() - float(analytic_shotnoise_response[0])) < 5 * cv_shotnoise.std(ddof=1) / np.sqrt(len(cv_shotnoise))
    # exact expectation, from a Hadamard set
    (input_data_weights, phi_u, phi_linear), (spectrum_of, shotnoise_of), _ = parts_from(kw, tiny_b)
    num, den = [], []
    for row in hadamard(len(input_data_weights)):
        v = phi_linear(jnp.asarray(row, dtype=input_data_weights.dtype))
        num.append(spectrum_of(*v))
        den.append(shotnoise_of(*v))
    np.testing.assert_allclose(np.mean(den, axis=0), np.asarray(analytic_shotnoise_response), rtol=1e-10)
    np.testing.assert_allclose(np.mean(num, axis=0)[0][:, low_k], np.asarray(analytic_spectrum_response)[0][:, low_k], atol=0.05 * scale)


def parts_from(kw, toy):
    args = kw["field_weights_args"]
    norms = get_estimator_normalizations(kw["fkp_norms"])
    spectrum_of = partial(bilinear_power_spectrum, particles=get_region_particles(*toy.fkp_fields), binner=kw["binner"], norms=norms)
    shotnoise_of = partial(conventional_shotnoise, binner=kw["binner"], norms=norms)
    return (args.input_data_weights, partial(apply_field_weights, field_weights_args=args), partial(field_weights_perturbation_gic_only, field_weights_args=args)), (spectrum_of, shotnoise_of), kw


def test_two_legs_under_jit_with_unchanged_static_argnames(toy):
    """(e) The pipeline runs under jit with the same static arguments; FieldWeightsArgs with two legs is a pytree."""
    toy_b = with_second_weights(toy)
    kw = two_leg_kwargs(toy_b, ("weight_FKP", "weight_B"), "ric")
    leaves, treedef = jax.tree.flatten(kw["field_weights_args"])
    assert jax.tree.unflatten(treedef, leaves).n_legs == 2
    assert shotnoise._STATIC_ARGNAMES == ["n_real", "noise_distribution", "compute_control_variate", "batch_size", "los"]
    for sampler in (sample_shotnoise_template_antithetic, sample_shotnoise_template_linearized, sample_shotnoise_template_quadratic):
        res = sampler(*toy_b.fkp_fields, key=KEY, n_real=2, compute_control_variate=True, batch_size=2, los="local", **kw)
        assert res["power_spectrum_response"].shape == (2, 1, 3, kw["binner"].xavg.shape[0])
        assert res["shotnoise_response"].shape == (2, 1)
        assert np.all(np.isfinite(res["power_spectrum_response"]))
    q, s = jax.jit(measure_power_spectrum_and_shotnoise, static_argnames=["los"])(*toy_b.fkp_fields, **kw)
    assert np.all(np.isfinite(q)) and np.all(np.isfinite(s))
