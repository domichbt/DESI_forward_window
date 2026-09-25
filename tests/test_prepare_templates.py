"""Tests for :py:func:`desiwinds.utils.prepare_templates` (AMR template normalization / digitization) and its use in :py:func:`desiwinds.forward.apply_AMR`."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from desiwinds.forward import apply_AMR
from desiwinds.utils import prepare_templates

N_BINS = 10
OFFSET = N_BINS + 2


def run_prepare(data_templates, randoms_templates, data_regions, randoms_regions, randoms_is_real=None, tail=1.0, bin_margin=1e-7, n_bins=N_BINS):
    if randoms_is_real is None:
        randoms_is_real = np.ones(len(randoms_templates), dtype=bool)
    outputs = prepare_templates(
        data_templates=jnp.asarray(data_templates),
        randoms_templates=jnp.asarray(randoms_templates),
        data_regions=[jnp.asarray(m) for m in data_regions],
        randoms_regions=[jnp.asarray(m) for m in randoms_regions],
        randoms_is_real=jnp.asarray(randoms_is_real),
        tail=tail,
        n_bins=n_bins,
        bin_margin=bin_margin,
    )
    return [np.asarray(out) for out in outputs]


def make_regions(n, n_outside=0):
    """Two disjoint regions (first and second half), and ``n_outside`` trailing objects in no region."""
    idx = np.arange(n)
    n_in = n - n_outside
    return [idx < n_in // 2, (idx >= n_in // 2) & (idx < n_in)]


def expected_tails(randoms_templates, sel, tail):
    lower = np.percentile(randoms_templates[sel], tail / 2, axis=0, method="higher")
    upper = np.percentile(randoms_templates[sel], 100 - tail / 2, axis=0, method="lower")
    return lower, upper


# ---------------------------------------------------------------------------
# Bug 1: constant (row 0) digitization of the randoms
# ---------------------------------------------------------------------------


def test_data_and_randoms_digitized_identically():
    """Same values and masks for data and randoms must give the same outputs."""
    rng = np.random.default_rng(0)
    n = 6000
    templates = rng.normal(size=(n, 3))
    regions = make_regions(n, n_outside=200)
    dn, dd, rn, rd = run_prepare(templates, templates, regions, regions)
    np.testing.assert_array_equal(dn, rn)
    np.testing.assert_array_equal(dd, rd)


@pytest.mark.parametrize("which", ["data", "randoms"])
def test_constant_row_values(which):
    """Row 0 is ``n_bins - 1 + ireg * (n_bins + 2)`` for non-extreme objects, 0 for extreme ones and for objects in no region."""
    rng = np.random.default_rng(1)
    n, n_outside, tail = 10000, 300, 1.0
    randoms_templates = rng.normal(size=(n, 3))
    data_templates = rng.normal(size=(n, 3))
    regions = make_regions(n, n_outside=n_outside)
    dn, dd, rn, rd = run_prepare(data_templates, randoms_templates, regions, regions, tail=tail)
    templates, digitized = (data_templates, dd) if which == "data" else (randoms_templates, rd)

    expected = np.zeros(n, dtype=int)
    for ireg, sel in enumerate(regions):
        lower, upper = expected_tails(randoms_templates, sel, tail)
        non_extreme = np.all((templates >= lower) & (templates <= upper), axis=1)
        assert np.any(sel & ~non_extreme) and np.any(sel & non_extreme)  # both cases are actually tested
        expected[sel] = np.where(non_extreme[sel], N_BINS - 1 + ireg * OFFSET, 0)
    np.testing.assert_array_equal(digitized[0], expected)



def test_objects_in_no_region_are_discarded():
    """Objects in no region are in (discarded) bin 0 for every row, including the constant one."""
    rng = np.random.default_rng(5)
    n, n_outside = 6000, 500
    templates = rng.normal(size=(n, 3))
    regions = make_regions(n, n_outside=n_outside)
    dn, dd, rn, rd = run_prepare(templates, templates, regions, regions)
    np.testing.assert_array_equal(dd[:, -n_outside:], 0)
    np.testing.assert_array_equal(rd[:, -n_outside:], 0)


def test_constant_row_overlapping_regions():
    """For objects in several regions, row 0 follows the last region (like the other rows) instead of accumulating offsets."""
    rng = np.random.default_rng(6)
    n = 6000
    templates = rng.normal(size=(n, 3))
    idx = np.arange(n)
    regions = [idx < n // 3, idx >= n // 3, idx >= 2 * n // 3]  # region 2 overlaps region 1
    dn, dd, rn, rd = run_prepare(templates, templates, regions, regions, tail=0.0)  # tail=0: no extreme randoms
    both = regions[1] & regions[2]
    np.testing.assert_array_equal(rd[0, both], N_BINS - 1 + 2 * OFFSET)
    np.testing.assert_array_equal(rd[0, regions[1] & ~both], N_BINS - 1 + OFFSET)
    assert np.all(rd[1:, both] // OFFSET == 2)

# ---------------------------------------------------------------------------
# Bug 2: correction for objects on the upper bin edge
# ---------------------------------------------------------------------------


def test_upper_edge_goes_to_last_bin_without_margin():
    """With ``bin_margin=0``, the object sitting on the upper tail is in the last regular bin, not in the overflow bin."""
    rng = np.random.default_rng(2)
    n = 8000
    randoms_templates = rng.normal(size=(n, 3))
    data_templates = rng.normal(size=(n, 3))
    regions = make_regions(n)
    dn, dd, rn, rd = run_prepare(data_templates, randoms_templates, regions, regions, tail=1.0, bin_margin=0.0)

    for ireg, sel in enumerate(regions):
        lower, upper = expected_tails(randoms_templates, sel, 1.0)
        for normalized, digitized, templates in [(dn, dd, data_templates), (rn, rd, randoms_templates)]:
            non_extreme = sel & np.all((templates >= lower) & (templates <= upper), axis=1)
            bins = digitized[1:, non_extreme] - ireg * OFFSET
            assert bins.min() >= 1 and bins.max() <= N_BINS
        # the randoms exactly on the upper tail
        for isys in range(randoms_templates.shape[1]):
            on_edge = sel & (randoms_templates[:, isys] == upper[isys])
            assert on_edge.any()
            np.testing.assert_array_equal(rn[1 + isys, on_edge], 1.0)
            np.testing.assert_array_equal(rd[1 + isys, on_edge], N_BINS + ireg * OFFSET)


def test_no_spurious_edge_correction():
    """The edge correction must not fire when a normalized value happens to equal the raw upper bin edge."""
    n = 1001
    randoms_templates = np.linspace(-0.5, 0.5, n)[:, None]  # with tail=0, tails are exactly [-0.5, 0.5]
    data_templates = np.zeros((5, 1))  # normalized value 0.5, equal to the raw upper edge
    dn, dd, rn, rd = run_prepare(data_templates, randoms_templates, [np.ones(5, bool)], [np.ones(n, bool)], tail=0.0, bin_margin=0.0)
    np.testing.assert_array_equal(dn[1], 0.5)
    np.testing.assert_array_equal(dd[1], int(np.floor(0.5 * N_BINS)) + 1)


@pytest.mark.parametrize("bin_margin", [0.0, 1e-7, 0.1])
def test_digitization_matches_reference(bin_margin):
    """Digitization of non-extreme objects matches an independent ``searchsorted`` on the bin edges (last edge closed)."""
    rng = np.random.default_rng(3)
    n, tail = 8000, 1.0
    randoms_templates = rng.normal(size=(n, 3))
    data_templates = rng.normal(size=(n, 3))
    regions = make_regions(n)
    dn, dd, rn, rd = run_prepare(data_templates, randoms_templates, regions, regions, tail=tail, bin_margin=bin_margin)

    for ireg, sel in enumerate(regions):
        lower, upper = expected_tails(randoms_templates, sel, tail)
        for digitized, templates in [(dd, data_templates), (rd, randoms_templates)]:
            non_extreme = sel & np.all((templates >= lower) & (templates <= upper), axis=1)
            for isys in range(templates.shape[1]):
                edges = np.linspace(lower[isys] - bin_margin, upper[isys] + bin_margin, N_BINS + 1)
                x = templates[non_extreme, isys]
                expected = np.minimum(np.searchsorted(edges, x, side="right"), N_BINS)
                np.testing.assert_array_equal(digitized[1 + isys, non_extreme] - ireg * OFFSET, expected)


# ---------------------------------------------------------------------------
# Consequences for the AMR fit
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def amr_setup():
    rng = np.random.default_rng(4)
    nd, nr = 50000, 200000
    randoms_templates = rng.normal(size=(nr, 2))
    data_templates = rng.normal(size=(nd, 2))
    amplitudes = np.array([0.15, -0.1])
    data_weights = 1 + data_templates @ amplitudes  # injected linear systematic, stays positive (min is a > 5 sigma fluctuation)
    randoms_weights = np.ones(nr)
    data_regions, randoms_regions = make_regions(nd), make_regions(nr)
    dn, dd, rn, rd = run_prepare(data_templates, randoms_templates, data_regions, randoms_regions, tail=1.0, bin_margin=1e-7)
    return dict(
        data_templates=data_templates,
        amplitudes=amplitudes,
        data_weights=data_weights,
        randoms_weights=randoms_weights,
        data_regions=np.stack(data_regions),
        randoms_regions=np.stack(randoms_regions),
        dn=dn,
        dd=dd,
        rn=rn,
        rd=rd,
    )


def amr_weights(setup, dd, rd):
    dd, rd = jnp.asarray(dd), jnp.asarray(rd)
    return np.asarray(
        apply_AMR(
            jnp.asarray(setup["data_weights"]),
            jnp.asarray(setup["randoms_weights"]),
            jnp.asarray(setup["data_regions"]),
            jnp.asarray(setup["randoms_regions"]),
            dd,
            rd,
            jnp.asarray(setup["dn"]),
            jnp.asarray(setup["rn"]),
            jnp.argsort(dd, axis=1),
            jnp.argsort(rd, axis=1),
            n_bins=N_BINS,
            apply_to="data",
        )
    )


def test_constant_row_is_redundant_in_fit(amr_setup):
    """When all template rows bin the same objects, dropping the constant row (sending it to discarded bin 0) does not change the weights."""
    weights = amr_weights(amr_setup, amr_setup["dd"], amr_setup["rd"])
    dd_dropped, rd_dropped = amr_setup["dd"].copy(), amr_setup["rd"].copy()
    dd_dropped[0] = 0
    rd_dropped[0] = 0
    weights_dropped = amr_weights(amr_setup, dd_dropped, rd_dropped)
    np.testing.assert_allclose(weights_dropped, weights, rtol=1e-12)


def test_injected_systematic_is_removed(amr_setup):
    """Corrected data weights show no residual linear trend with the templates.

    The residual slope is limited by the finite number of data / randoms per bin: with this setup its scatter over seeds is ~0.005,
    so the threshold (0.025) is ~5 sigma, i.e. <= 25% of the injected amplitudes.
    """
    weights = amr_weights(amr_setup, amr_setup["dd"], amr_setup["rd"])
    corrected = amr_setup["data_weights"] * weights
    for isys, amplitude in enumerate(amr_setup["amplitudes"]):
        slope_before = np.polyfit(amr_setup["data_templates"][:, isys], amr_setup["data_weights"], 1)[0]
        slope_after = np.polyfit(amr_setup["data_templates"][:, isys], corrected, 1)[0]
        assert np.isclose(slope_before, amplitude, rtol=0.05)
        assert abs(slope_after) < 0.025
    assert np.isclose(corrected.mean(), 1.0, atol=1e-2)


def test_objects_in_no_region_do_not_affect_fit():
    """Adding objects in no region leaves the weights of in-region objects unchanged, and gives weight 1 to the others."""
    rng = np.random.default_rng(7)
    nd, nr, nd_out, nr_out = 40000, 160000, 4000, 30000
    randoms_templates = rng.normal(size=(nr + nr_out, 2))
    data_templates = rng.normal(size=(nd + nd_out, 2))
    data_weights = 1 + data_templates @ np.array([0.15, -0.1])
    randoms_weights = np.ones(nr + nr_out)

    def weights(n_d, n_r):
        # the first n_d / n_r objects are split in two regions, the rest is in no region
        data_regions, randoms_regions = make_regions(n_d, n_outside=n_d - nd), make_regions(n_r, n_outside=n_r - nr)
        dn, dd, rn, rd = run_prepare(data_templates[:n_d], randoms_templates[:n_r], data_regions, randoms_regions)
        setup = dict(
            data_weights=data_weights[:n_d],
            randoms_weights=randoms_weights[:n_r],
            data_regions=np.stack(data_regions),
            randoms_regions=np.stack(randoms_regions),
            dn=dn,
            rn=rn,
        )
        return amr_weights(setup, dd, rd)

    weights_without = weights(nd, nr)
    weights_with = weights(nd + nd_out, nr + nr_out)
    np.testing.assert_allclose(weights_with[:nd], weights_without, rtol=1e-12)
    np.testing.assert_array_equal(weights_with[nd:], 1.0)
