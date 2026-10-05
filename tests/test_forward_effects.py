"""
Tests for :py:func:`desiwinds.forward._apply_effects` and :py:func:`desiwinds.forward._split_weights`.

These helpers factor out the sequence RIC -> AMR (-> RIC) -> NAM -> data-to-randoms renormalization -> split that used to be written inline in
:py:func:`mock_survey_catalog` and :py:func:`mock_whitenoise`. The refactoring must be a pure code move, so the results are compared **bitwise**
to a verbatim copy of the previous inline code (``_legacy_*`` below), for auto and cross correlations and all combinations of effects.

The arguments are built by hand from random arrays, so that no data files, cosmology or particular ``jaxpower`` version are needed.
"""

import itertools
from types import SimpleNamespace

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from desiwinds.forward import AMR_args, NAM_args, RIC_args, _apply_effects, _split_weights, apply_AMR, apply_NAM, apply_RIC
from desiwinds.utils import local_argsort, local_concatenate, local_split, prepare_templates

N_DATA, N_RANDOMS = 60, 240
N_REGIONS = 2
RIC_BINS, AMR_BINS, NSIDE, N_SYS = 8, 4, 2, 3


# ---------------------------------------------------------------------------
# Verbatim copies of the code that used to live inside mock_survey_catalog / mock_whitenoise
# ---------------------------------------------------------------------------


def _legacy_apply_effects(data_weights, randoms_weights, ric_args, amr_args, nam_args, data_regions, randoms_regions):
    for idx, ric_arg in enumerate(ric_args):
        ric_weight = apply_RIC(
            data_weights=data_weights[idx],
            randoms_weights=randoms_weights[idx],
            data_regions=ric_arg.data_regions,
            randoms_regions=ric_arg.randoms_regions,
            data_distances_digitized=ric_arg.data_distances_digitized,
            randoms_distances_digitized=ric_arg.randoms_distances_digitized,
            n_bins=ric_arg.n_bins,
            apply_to=ric_arg.apply_to,
        )
        if ric_arg.apply_to == "data":
            data_weights[idx] = data_weights[idx] * ric_weight
        else:
            randoms_weights[idx] = randoms_weights[idx] * ric_weight

    for idx, amr_arg in enumerate(amr_args):
        amr_weights = apply_AMR(
            data_weights=data_weights[idx],
            randoms_weights=randoms_weights[idx],
            data_regions=amr_arg.data_regions,
            randoms_regions=amr_arg.randoms_regions,
            data_templates_digitized=amr_arg.data_templates_digitized,
            randoms_templates_digitized=amr_arg.randoms_templates_digitized,
            data_templates_normalized=amr_arg.data_templates_normalized,
            randoms_templates_normalized=amr_arg.randoms_templates_normalized,
            data_isort=amr_arg.data_isort,
            randoms_isort=amr_arg.randoms_isort,
            n_bins=amr_arg.n_bins,
            apply_to=amr_arg.apply_to,
        )
        if amr_arg.apply_to == "data":
            data_weights[idx] = data_weights[idx] * amr_weights
        else:
            randoms_weights[idx] = randoms_weights[idx] * amr_weights
        if ric_args:
            ric_weights = apply_RIC(
                data_weights=data_weights[idx],
                randoms_weights=randoms_weights[idx],
                data_regions=ric_args[idx].data_regions,
                randoms_regions=ric_args[idx].randoms_regions,
                data_distances_digitized=ric_args[idx].data_distances_digitized,
                randoms_distances_digitized=ric_args[idx].randoms_distances_digitized,
                n_bins=ric_args[idx].n_bins,
                apply_to=ric_args[idx].apply_to,
            )
            if ric_args[idx].apply_to == "data":
                data_weights[idx] = data_weights[idx] * ric_weights
            else:
                randoms_weights[idx] = randoms_weights[idx] * ric_weights

    for idx, nam_arg in enumerate(nam_args):
        nam_weights = apply_NAM(
            data_weights=data_weights[idx],
            randoms_weights=randoms_weights[idx],
            data_regions=nam_arg.data_regions,
            randoms_regions=nam_arg.randoms_regions,
            data_pixels=nam_arg.data_pixels,
            randoms_pixels=nam_arg.randoms_pixels,
            invsigma2=nam_arg.invsigma2,
            nside=nam_arg.nside,
            apply_to=nam_arg.apply_to,
        )
        if nam_arg.apply_to == "data":
            data_weights[idx] = data_weights[idx] * nam_weights
        else:
            randoms_weights[idx] = randoms_weights[idx] * nam_weights

    for idx, (_randoms_regions, _data_regions) in enumerate(zip(randoms_regions, data_regions, strict=True)):
        global_alpha = data_weights[idx].sum() / randoms_weights[idx].sum()
        alphas = (data_weights[idx] * _data_regions).sum(axis=-1) / (randoms_weights[idx] * _randoms_regions).sum(axis=-1)
        correction = (_randoms_regions * alphas[..., None] / global_alpha).sum(axis=0) + jnp.invert(_randoms_regions.any(axis=0))
        randoms_weights[idx] = randoms_weights[idx] * correction
    return data_weights, randoms_weights


def _legacy_split_weights(data_weights, randoms_weights, fkp_fields, sharding_mesh):
    split_indices_data = tuple(
        list(itertools.accumulate([fkp_field.data.weights.shape[0] for fkp_field in region_group]))[:-1] for region_group in zip(*fkp_fields, strict=True)
    )
    data_weights = tuple(
        zip(
            *(
                local_split(data_weight, split_idx, axis=0, sharding_mesh=sharding_mesh)
                for data_weight, split_idx in zip(data_weights, split_indices_data, strict=True)
            ),
            strict=True,
        )
    )
    split_indices_randoms = tuple(
        list(itertools.accumulate([fkp_field.randoms.weights.shape[0] for fkp_field in region_group]))[:-1] for region_group in zip(*fkp_fields, strict=True)
    )
    randoms_weights = tuple(
        zip(
            *(
                local_split(randoms_weight, split_idx, axis=0, sharding_mesh=sharding_mesh)
                for randoms_weight, split_idx in zip(randoms_weights, split_indices_randoms, strict=True)
            ),
            strict=True,
        )
    )
    return data_weights, randoms_weights


# ---------------------------------------------------------------------------
# Synthetic arguments
# ---------------------------------------------------------------------------


def _region_masks(rng, n):
    """Complementary region masks, shape (N_REGIONS, n); the last few particles are in no region at all."""
    labels = rng.integers(0, N_REGIONS, n)
    labels[-3:] = -1
    return np.stack([labels == r for r in range(N_REGIONS)])


def make_tracer(seed, apply_to=("randoms", "randoms", "randoms")):
    """Random weights and (RIC, AMR, NAM) arguments for one fake tracer."""
    rng = np.random.default_rng(seed)
    data_regions, randoms_regions = _region_masks(rng, N_DATA), _region_masks(rng, N_RANDOMS)
    data_weights = jnp.asarray(rng.uniform(0.5, 1.5, N_DATA))
    randoms_weights = jnp.asarray(rng.uniform(0.5, 1.5, N_RANDOMS))
    data_regions_j, randoms_regions_j = jnp.asarray(data_regions), jnp.asarray(randoms_regions)

    ric = RIC_args(
        data_distances_digitized=jnp.asarray(rng.integers(1, RIC_BINS + 1, N_DATA)),
        randoms_distances_digitized=jnp.asarray(rng.integers(1, RIC_BINS + 1, N_RANDOMS)),
        data_regions=data_regions_j,
        randoms_regions=randoms_regions_j,
        data_to_remove=jnp.zeros(N_DATA, dtype=bool),
        n_bins=RIC_BINS,
        apply_to=apply_to[0],
    )

    data_templates = rng.normal(size=(N_DATA, N_SYS))
    randoms_templates = rng.normal(size=(N_RANDOMS, N_SYS))
    d_norm, d_dig, r_norm, r_dig = prepare_templates(
        data_templates=jnp.asarray(data_templates),
        randoms_templates=jnp.asarray(randoms_templates),
        data_regions=list(data_regions_j),
        randoms_regions=list(randoms_regions_j),
        randoms_is_real=jnp.ones(N_RANDOMS, dtype=bool),
        tail=2.0,
        n_bins=AMR_BINS,
        bin_margin=1e-7,
    )
    amr = AMR_args(
        data_regions=data_regions_j,
        randoms_regions=randoms_regions_j,
        data_templates_digitized=d_dig,
        randoms_templates_digitized=r_dig,
        data_templates_normalized=d_norm,
        randoms_templates_normalized=r_norm,
        data_isort=local_argsort(d_dig, axis=1),
        randoms_isort=local_argsort(r_dig, axis=1),
        n_bins=AMR_BINS,
        apply_to=apply_to[1],
    )

    npix = 12 * NSIDE**2
    nam = NAM_args(
        data_pixels=jnp.asarray(rng.integers(0, npix, N_DATA)),
        randoms_pixels=jnp.asarray(rng.integers(0, npix, N_RANDOMS)),
        data_regions=data_regions_j,
        randoms_regions=randoms_regions_j,
        data_to_remove=jnp.zeros(N_DATA, dtype=bool),
        invsigma2=jnp.atleast_1d(1.0),
        nside=NSIDE,
        apply_to=apply_to[2],
    )
    return SimpleNamespace(data_weights=data_weights, randoms_weights=randoms_weights, ric=ric, amr=amr, nam=nam, data_regions=data_regions_j, randoms_regions=randoms_regions_j)


def _wrap(tracers, use_ric, use_amr, use_nam):
    """Build the keyword arguments exactly as mock_* does them: tuples of per-tracer args, empty if the effect is off."""
    return dict(
        ric_args=tuple(t.ric for t in tracers) if use_ric else (),
        amr_args=tuple(t.amr for t in tracers) if use_amr else (),
        nam_args=tuple(t.nam for t in tracers) if use_nam else (),
        data_regions=tuple(t.data_regions for t in tracers),
        randoms_regions=tuple(t.randoms_regions for t in tracers),
    )


def _assert_identical(got, expected):
    """Bitwise equality of two (data_weights, randoms_weights) results."""
    for g_list, e_list in zip(got, expected, strict=True):
        assert len(g_list) == len(e_list)
        for g, e in zip(g_list, e_list, strict=True):
            assert g.dtype == e.dtype
            assert g.shape == e.shape
            np.testing.assert_array_equal(np.asarray(g), np.asarray(e))


EFFECT_COMBINATIONS = [
    (False, False, False),  # renormalization only
    (True, False, False),
    (False, True, False),  # AMR without RIC: no RIC re-application
    (False, False, True),
    (True, True, False),  # standard DESI
    (True, False, True),
    (False, True, True),
    (True, True, True),
]
APPLY_TO = [
    ("randoms", "randoms", "randoms"),
    ("data", "data", "data"),
    ("randoms", "data", "randoms"),
    ("data", "randoms", "data"),
]


# ---------------------------------------------------------------------------
# _apply_effects
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("apply_to", APPLY_TO, ids=lambda a: "-".join(a))
@pytest.mark.parametrize("effects", EFFECT_COMBINATIONS, ids=lambda e: "".join(n for n, on in zip(("RIC", "AMR", "NAM"), e, strict=True) if on) or "none")
@pytest.mark.parametrize("n_tracers", [1, 2], ids=["auto", "cross"])
def test_apply_effects_bitwise_identical_to_inline(n_tracers, effects, apply_to):
    """Helper == inline code, bit for bit, for auto and cross correlations."""
    tracers = [make_tracer(seed, apply_to) for seed in range(n_tracers)]
    kwargs = _wrap(tracers, *effects)

    expected = _legacy_apply_effects([t.data_weights for t in tracers], [t.randoms_weights for t in tracers], **kwargs)
    got = _apply_effects([t.data_weights for t in tracers], [t.randoms_weights for t in tracers], **kwargs)
    _assert_identical(got, expected)


@pytest.mark.parametrize("effects", [(False, False, False), (True, True, False), (True, True, True)], ids=["none", "RIC+AMR", "all"])
def test_apply_effects_bitwise_identical_under_jit(effects):
    """Same graph once jitted (which is how the forward model is used)."""
    tracers = [make_tracer(seed) for seed in range(2)]
    kwargs = _wrap(tracers, *effects)
    dw, rw = [t.data_weights for t in tracers], [t.randoms_weights for t in tracers]

    expected = jax.jit(lambda d, r, kw: _legacy_apply_effects(list(d), list(r), **kw))(dw, rw, kwargs)
    got = jax.jit(lambda d, r, kw: _apply_effects(list(d), list(r), **kw))(dw, rw, kwargs)
    _assert_identical(got, expected)


def test_apply_effects_does_not_mutate_inputs():
    """Callers keep their lists: the helper must work on a copy."""
    tracers = [make_tracer(seed) for seed in range(2)]
    dw, rw = [t.data_weights for t in tracers], [t.randoms_weights for t in tracers]
    dw_ref, rw_ref = list(dw), list(rw)
    dw_before, rw_before = [np.asarray(x).copy() for x in dw], [np.asarray(x).copy() for x in rw]

    new_dw, new_rw = _apply_effects(dw, rw, **_wrap(tracers, True, True, True))

    assert all(a is b for a, b in zip(dw, dw_ref, strict=True))
    assert all(a is b for a, b in zip(rw, rw_ref, strict=True))
    for before, after in zip(dw_before + rw_before, dw + rw, strict=True):
        np.testing.assert_array_equal(before, np.asarray(after))
    assert new_dw is not dw and new_rw is not rw


def test_apply_effects_accepts_tuples_and_returns_lists():
    tracers = [make_tracer(0)]
    out_d, out_r = _apply_effects((tracers[0].data_weights,), (tracers[0].randoms_weights,), **_wrap(tracers, True, False, False))
    assert isinstance(out_d, list) and isinstance(out_r, list)
    assert len(out_d) == len(out_r) == 1


def test_apply_effects_renormalization_makes_regional_ratios_global():
    """Physical sanity check: after renormalization the data/randoms ratio is the same in every region."""
    t = make_tracer(3)
    (dw,), (rw,) = _apply_effects([t.data_weights], [t.randoms_weights], **_wrap([t], False, False, False))
    ratios = (dw * t.data_regions).sum(axis=-1) / (rw * t.randoms_regions).sum(axis=-1)
    # every region ends up at the global data/randoms ratio computed *before* the renormalization
    np.testing.assert_allclose(ratios, (t.data_weights.sum() / t.randoms_weights.sum()) * np.ones_like(ratios), rtol=1e-12)


def test_apply_effects_only_touches_requested_catalog():
    """With ``apply_to="randoms"`` for everything, data weights are untouched."""
    t = make_tracer(4, ("randoms",) * 3)
    (dw,), (rw,) = _apply_effects([t.data_weights], [t.randoms_weights], **_wrap([t], True, True, True))
    np.testing.assert_array_equal(np.asarray(dw), np.asarray(t.data_weights))
    assert not np.array_equal(np.asarray(rw), np.asarray(t.randoms_weights))


def test_apply_effects_effect_order_matters():
    """RIC then AMR then NAM: guards against someone reordering the steps (results differ if permuted)."""
    t = make_tracer(5)
    dw, rw = [t.data_weights], [t.randoms_weights]
    full = _apply_effects(dw, rw, **_wrap([t], True, True, True))
    no_nam = _apply_effects(dw, rw, **_wrap([t], True, True, False))
    # undoing NAM's presence changes the result; and the AMR-only output differs from the RIC+AMR one (RIC re-applied after AMR)
    assert not np.array_equal(np.asarray(full[1][0]), np.asarray(no_nam[1][0]))
    amr_only = _apply_effects(dw, rw, **_wrap([t], False, True, False))
    assert not np.array_equal(np.asarray(amr_only[1][0]), np.asarray(no_nam[1][0]))


# ---------------------------------------------------------------------------
# _split_weights
# ---------------------------------------------------------------------------


def _fake_fkp(n_data, n_randoms):
    return SimpleNamespace(data=SimpleNamespace(weights=jnp.zeros(n_data)), randoms=SimpleNamespace(weights=jnp.zeros(n_randoms)))


@pytest.mark.parametrize("n_tracers", [1, 2], ids=["auto", "cross"])
@pytest.mark.parametrize("sizes", [[(10, 40)], [(10, 40), (25, 70)], [(7, 13), (0, 5), (3, 22)]], ids=["1region", "2regions", "3regions-empty-data"])
def test_split_weights_bitwise_identical_to_inline(n_tracers, sizes):
    rng = np.random.default_rng(0)
    # one tuple of FKP fields per region, one element per tracer (as mock_* builds them)
    fkp_fields = tuple(tuple(_fake_fkp(nd, nr) for _ in range(n_tracers)) for nd, nr in sizes)
    n_d, n_r = sum(nd for nd, _ in sizes), sum(nr for _, nr in sizes)
    data_weights = [jnp.asarray(rng.uniform(size=n_d)) for _ in range(n_tracers)]
    randoms_weights = [jnp.asarray(rng.uniform(size=n_r)) for _ in range(n_tracers)]
    mesh = jax.sharding.get_abstract_mesh()  # empty (unsharded) mesh, as in a single-device run

    expected = _legacy_split_weights(data_weights, randoms_weights, fkp_fields, sharding_mesh=mesh)
    got = _split_weights(data_weights, randoms_weights, fkp_fields, sharding_mesh=mesh)

    for g_all, e_all in zip(got, expected, strict=True):
        assert len(g_all) == len(e_all) == len(sizes)  # one entry per region...
        for g_region, e_region in zip(g_all, e_all, strict=True):
            assert len(g_region) == len(e_region) == n_tracers  # ...each with one entry per tracer
            for g, e in zip(g_region, e_region, strict=True):
                np.testing.assert_array_equal(np.asarray(g), np.asarray(e))


def test_split_weights_is_inverse_of_concatenation():
    """Splitting then re-concatenating returns the input; and the split shapes follow the FKP fields."""
    sizes = [(10, 40), (25, 70)]
    rng = np.random.default_rng(1)
    fkp_fields = tuple((_fake_fkp(nd, nr),) for nd, nr in sizes)
    dw = [jnp.asarray(rng.uniform(size=35))]
    rw = [jnp.asarray(rng.uniform(size=110))]
    mesh = jax.sharding.get_abstract_mesh()
    split_d, split_r = _split_weights(dw, rw, fkp_fields, sharding_mesh=mesh)
    assert [x[0].shape[0] for x in split_d] == [10, 25]
    assert [x[0].shape[0] for x in split_r] == [40, 70]
    np.testing.assert_array_equal(np.asarray(local_concatenate([x[0] for x in split_d], axis=0, sharding_mesh=mesh)), np.asarray(dw[0]))
    np.testing.assert_array_equal(np.asarray(local_concatenate([x[0] for x in split_r], axis=0, sharding_mesh=mesh)), np.asarray(rw[0]))
