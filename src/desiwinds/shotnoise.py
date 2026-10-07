r"""
Shot noise of the power spectrum multipoles: white-noise forward model and scale-dependent shot-noise templates.

The shot noise contribution to the measured multipoles is the conventional (constant) shot noise times a dimensionless, scale-dependent template, which is distorted by the observational effects forward-modeled in :py:mod:`desiwinds.forward` (RIC, AMR, NAM, data to randoms renormalization, global integral constraint). The template is sampled by injecting independent noise on the weights of the "data" (a random half of the randoms) and propagating it through the pipeline. Three methods are available, an antithetic noise injection with normalization (:py:func:`sample_shotnoise_template_antithetic`), a linearization by forward-mode differentiation (:py:func:`sample_shotnoise_template_linearized`) and a nested one that keeps the curvature of the pipeline (:py:func:`sample_shotnoise_template_quadratic`); they are assembled by :py:func:`shotnoise_template_from_samples`. An analytic template for the geometry and the global integral constraint, :py:func:`analytic_shotnoise_template`, can serve as control variate.

The measured field is the sum over the objects of their weights times a Dirac delta, with the data first and then the randoms, whose weights absorb the sign and the ratio ``alpha`` of the data to randoms weights. Estimator weights (FKP, OQE...) and the estimator normalization are frozen at their noise-free values, and no shot noise is subtracted. Only a single tracer is supported, with one or two estimator weightings (*e.g.* the two legs of an OQE pair): single tracer, one or two estimator weightings; several regions (*e.g.* NGC and SGC) are handled as separate FKP fields, like in :py:func:`desiwinds.forward.mock_survey_catalog`.
"""

import itertools
from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from jaxpower import (
    BinMesh2SpectrumPoles,
    FKPField,
    ParticleField,
    compute_mesh2_spectrum,
)
from jaxpower.mesh import get_sharding_mesh
from jaxpower.utils import get_Ylm
from lsstypes import Mesh2SpectrumPoles

from .forward import (
    AMR_args,
    NAM_args,
    RIC_args,
    _apply_effects,
    _fill_with_constant,
    _get_pk,
    _get_pk_nogic,
    _split_weights,
    _update_fkp,
)
from .utils import local_concatenate, local_split, make_jax_dataclass


def mock_whitenoise(
    # Catalogs
    *fkp_fields: FKPField | tuple[FKPField, FKPField],
    # White noise generation
    sigma: jax.Array | tuple[jax.Array, jax.Array],
    seed: jax.Array,
    los: Literal["local", "x", "y", "z"],
    # Effects
    gic: bool = True,
    ric_args: RIC_args | tuple[RIC_args, RIC_args] | None,
    amr_args: AMR_args | tuple[AMR_args, AMR_args] | None,
    nam_args: NAM_args | tuple[NAM_args, NAM_args] | None,
    # Final P(k) estimation
    binner: BinMesh2SpectrumPoles,
    fkp_norms: list[jax.Array],
    estimator_weights: str | tuple[str, str] | None,
    # For region renormalization (need to be concatenated if multiple catalogs)
    data_regions: jax.Array | tuple[jax.Array, jax.Array] | None = None,
    randoms_regions: jax.Array | tuple[jax.Array, jax.Array] | None = None,
    # Field identifiers for the analytic shot noise
    fields: tuple[int, int] | None = None,
) -> list[Mesh2SpectrumPoles]:
    """
    Get the "windowed" power specturm of an input shot noise given a seed and a set of observational effects.

    Parameters
    ----------
    *fkp_fields : FKPField | tuple[FKPField, FKPField]
        FKP fields containing data and randoms information. The data shouldn't be clustered (*i.e.* the "data" should also be randoms), but the FKP field serves to designate data and randoms amongst the original randoms. One field per desired output power spectrum. Example: NGC and SGC can be provided as two separate FKP fields, to get two output spectra. Pass several **tuples** of FKP fields to compute cross-spectra, for example (LRG_NGC, ELG_NGC) and (LRG_SGC, ELG_SGC) to get the LRGxELG cross-spectra in NGC and SGC.
    sigma : jax.Array | tuple[jax.Array, jax.Array]
        Standard deviation for the Gaussian white noise generation. Different values can be provided for cross-correlations of independent fields.
    seed : jax.Array
        Random seed for the mock survey mesh generation.
    los : Literal["local", "x", "y", "z"]
        Line of sight definition for the mock generation.
    gic : bool
        Whether to apply the global integral constaint. Default is True. Setting any additional effects (RIC, AMR, NAM) forces GIC.
    ric_args : RIC_args | tuple[RIC_args, RIC_args] | None
        Fixed, precomputed arguments for RIC weights computation by :py:func:`desiwinds.forward.apply_RIC`. Obtain with :py:func:`desiwinds.forward.prepare_RIC`. One per tracer for cross correlation.
    amr_args : AMR_args | tuple[AMR_args, AMR_args] | None
        Fixed, precomputed arguments for AMR weights computation by :py:func:`desiwinds.forward.apply_AMR`. Obtain with :py:func:`desiwinds.forward.prepare_AMR`. One per tracer for cross correlation.
    nam_args : NAM_args | tuple[NAM_args, NAM_args] | None
        Fixed, precomputed arguments for NAM weights computation by :py:func:`desiwinds.forward.apply_NAM`. Obtain with :py:func:`desiwinds.forward.prepare_NAM`. One per tracer for cross correlation.
    binner : BinMesh2SpectrumPoles
        Binning operator for the power spectrum estimation.
    fkp_norms : list[jax.Array]
        Pre-computed power spectrum norms for the FKP fields ``fkp_fields``, disregarding any future changes in weights.
    estimator_weights : str | tuple[str, str] | None, optional
        Name of the weights stored in the FKP fields' particle fields ``extra_fields`` to use as extra weight at estimation time. For example, FKP or OQE weights should not be applied for RIC and AMR but should be added at the spectrum estimation time. Default is ``None`` (no extra weight). Can pass a pair of strings for cross correlation.
    data_regions : jax.Array | tuple[jax.Array, jax.Array] | None, optional
        Regions for the data to randoms renormalization. By default None. These can typically be provided as the ``data_regions`` attribute in ``ric_args``, ``amr_args`` or ``nam_args``.
    randoms_regions : jax.Array | tuple[jax.Array, jax.Array] | None, optional
        Regions for the data to randoms renormalization. By default None. These can typically be provided as the ``randoms_regions`` attribute in ``ric_args``, ``amr_args`` or ``nam_args``.
    fields : tuple[int, int] | None, optional
        Forwarded to :func:`jaxpower.compute_fkp2_shotnoise` for each cross-correlation pair passed in ``fkp_fields``.
        By default ``None``, i.e. jaxpower's default field-identifier inference, which treats the two elements of a
        pair as physically different, independent fields (giving zero cross shot noise) -- correct for a genuine
        cross-tracer correlation (e.g. LRGxELG). Pass ``(0, 0)`` when the pair instead represents the SAME
        underlying particles weighted differently, as with OQE weight variants of a single tracer (e.g.
        ``estimator_weights=("weight_optimal_1", "weight_optimal_2")`` applied to one tracer's field duplicated for
        the cross term): without this, the analytic shot noise for that pair is silently 0, since jaxpower has no
        other way to know the two fields share positions.

    Returns
    -------
    list[Mesh2SpectrumPoles]
        Power spectra of one realization of shot noise; one for each FKP field or pair of FKP fields.

    Notes
    -----
    * RIC is applied first, then AMR, then NAM.
    * The data to randoms renormalization is applied last, after all weights modifications.
    * To reproduce the DESI process, use RIC and AMR. NAM is not part of the standard pipeline.
        * RIC is applied to NGC and SGC together, ie to N/S/DES. No redshift ranges.
        * AMR is also applied to NGC and SGC together, with wide redshift bins.
        * NAM is not necessary but should be applied like AMR if needed.
        * The data to randoms renormalization is done to NGC and SGC together. Arguments ``data_regions`` and ``randoms_regions`` from ``ric_args`` are suitable.
    * Most of the time, it is preferable to apply RIC, AMR and NAM to the randoms; this is especially true when this function is used to generate window matrices.
    * Using the same seed guarantees reproducibility for a given number and ordering of devices only.
    * The spectra returned here have the conventional shot noise of the *noisy* weights subtracted. For a template free of this subtraction, see :py:func:`sample_shotnoise_template_antithetic`.

    Examples
    --------
    For one tracer (autocorrelation), with NGC and SGC as two separate FKP fields, and RIC and AMR applied to both:
    >>> fw_jit = jax.jit(mock_whitenoise, static_argnames=["los"])
    >>> pk_sgc, pk_ngc = fw_jit(
            fkp_sgc,
            fkp_ngc,
            sigma=jnp.array([1.]),
            seed=jax.random.key(42),
            los="local",
            ric_args=ric_args,
            amr_args=amr_args,
            nam_args=None,
            fkp_norms=fkp_norms,
            binner=binner,
            data_regions=ric_args.data_regions,
            randoms_regions=ric_args.randoms_regions,
        )

    For two tracers (cross-correlation), with NGC and SGC as two separate FKP fields, and RIC and AMR applied to both:
    >>> fw_jit = jax.jit(mock_whitenoise, static_argnames=["los"])
    >>> pk_sgc, pk_ngc = fw_jit(
            (fkp_sgc_tracer1, fkp_sgc_tracer2),
            (fkp_ngc_tracer1, fkp_ngc_tracer2),
            sigma=(jnp.array([1.]), jnp.array([1.])), # independent white noises
            seed=jax.random.key(42),
            los="local",
            ric_args=(ric_args_tracer1, ric_args_tracer2),
            amr_args=(amr_args_tracer1, amr_args_tracer2),
            nam_args=None,
            fkp_norms=fkp_norms,
            binner=binner,
            data_regions=tuple(ric_arg.data_regions for ric_arg in ric_args),
            randoms_regions=tuple(ric_arg.randoms_regions for ric_arg in ric_args),
        )

    For one tracer and a single region, with RIC and NAM applied:
    >>> fw_jit = jax.jit(mock_whitenoise, static_argnames=["los"])
    >>> pk = fw_jit(
            fkp_sgc_tracer1,
            sigma=jnp.array([1.]),
            seed=jax.random.key(42),
            los="local",
            ric_args=ric_args,
            amr_args=None,
            nam_args=nam_args,
            fkp_norms=fkp_norm,
            binner=binner,
            data_regions=ric_args.data_regions,
            randoms_regions=ric_args.randoms_regions,
        )

    For one tracer with OQE weights ("cross" correlation), with NGC and SGC as two separate FKP fields, and RIC and AMR applied to both:
    >>> fw_jit = jax.jit(mock_whitenoise, static_argnames=["los"])
    >>> pk_sgc, pk_ngc = fw_jit(
            (fkp_sgc_oqe1, fkp_sgc_oqe2),
            (fkp_ngc_oqe1, fkp_ngc_oqe2),
            sigma=jnp.array([1.]), # same white noise realisation for both
            seed=jax.random.key(42),
            los="local",
            ric_args=ric_args,
            amr_args=amr_args,
            nam_args=None,
            fkp_norms=fkp_norms,
            binner=binner,
            data_regions=ric_args.data_regions,
            randoms_regions=ric_args.randoms_regions,
        )
    """
    if (not gic) and any(x is not None for x in (ric_args, amr_args, nam_args, data_regions, randoms_regions)):
        gic = True

    sigma = sigma if isinstance(sigma, tuple) else (sigma,)
    ric_args = () if ric_args is None else (ric_args if isinstance(ric_args, tuple) else (ric_args,))
    amr_args = () if amr_args is None else (amr_args if isinstance(amr_args, tuple) else (amr_args,))
    nam_args = () if nam_args is None else (nam_args if isinstance(nam_args, tuple) else (nam_args,))
    data_regions = () if data_regions is None else (data_regions if isinstance(data_regions, tuple) else (data_regions,))
    randoms_regions = () if randoms_regions is None else (randoms_regions if isinstance(randoms_regions, tuple) else (randoms_regions,))

    sharding_mesh = get_sharding_mesh()
    # ensure all fields are tuples, for easier processing later
    # they will be unpacked for P(k) anyways
    fkp_fields = tuple(fkp_field if isinstance(fkp_field, tuple) else (fkp_field,) for fkp_field in fkp_fields)

    if gic:
        alphas_gic = None
    else:
        def frozen_alpha(fkp, name):
            data, randoms = fkp.data.weights, fkp.randoms.weights
            if name:
                data, randoms = data * fkp.data.extra[name], randoms * fkp.randoms.extra[name]
            return data.sum() / randoms.sum()

        alphas_gic = tuple(
            tuple(frozen_alpha(fkp, name) for fkp, name in zip(group, estimator_weights if isinstance(estimator_weights, tuple) else (estimator_weights,) * len(group), strict=True))
            for group in fkp_fields
        )

    # Length of list = 1 or 2 dependent on whether we are doing auto or cross spectra
    data_weights = [
        local_concatenate([fkp_field.data.weights for fkp_field in region_group], axis=0, sharding_mesh=sharding_mesh)
        for region_group in zip(*fkp_fields, strict=True)
    ]
    randoms_weights = [
        local_concatenate([fkp_field.randoms.weights for fkp_field in region_group], axis=0, sharding_mesh=sharding_mesh)
        for region_group in zip(*fkp_fields, strict=True)
    ]

    # add white noise to the data weights, one key per sigma
    keys = jax.random.split(seed, len(sigma))
    if (len(sigma) == 1) and (len(data_weights) == 2):  # force same noise on both fields
        data_weights = [weights * (jax.random.normal(keys[0], shape=weights.shape, dtype=float) * sigma[0] + 1.0) for weights in data_weights]
    else:  # independent noise on each field
        data_weights = [
            weights * (jax.random.normal(key, shape=weights.shape, dtype=float) * sig + 1.0)
            for weights, key, sig in zip(data_weights, keys, sigma, strict=True)
        ]

    data_weights, randoms_weights = _apply_effects(
        data_weights,
        randoms_weights,
        ric_args=ric_args,
        amr_args=amr_args,
        nam_args=nam_args,
        data_regions=data_regions,
        randoms_regions=randoms_regions,
    )
    data_weights, randoms_weights = _split_weights(data_weights, randoms_weights, fkp_fields, sharding_mesh=sharding_mesh)

    fkp_fields = jax.tree.map(_update_fkp, data_weights, randoms_weights, fkp_fields, _fill_with_constant(data_weights, estimator_weights))
    if gic:
        pks = [_get_pk(*fkp_field, fkp_norm=fkp_norm, binner=binner, los=los, fields=fields) for fkp_field, fkp_norm in zip(fkp_fields, fkp_norms, strict=True)]
    else:
        pks = [
            _get_pk_nogic(*fkp_field, alphas=alpha_gic, fkp_norm=fkp_norm, binner=binner, los=los, fields=fields)
            for fkp_field, alpha_gic, fkp_norm in zip(fkp_fields, alphas_gic, fkp_norms, strict=True)
        ]
    return pks


def make_realization_keys(key: jax.Array, n_real: int) -> jax.Array:
    """
    Get one random key per realization.

    Realization ``r`` uses ``fold_in(key, r)``, so that it does not depend on the number of realizations. Using the same ``key`` in all ``sample_shotnoise_template_*`` functions guarantees common random numbers.

    Parameters
    ----------
    key : jax.Array
        Random key.
    n_real : int
        Number of realizations.

    Returns
    -------
    jax.Array
        One key per realization.
    """
    return jax.vmap(jax.random.fold_in, in_axes=(None, 0))(key, jnp.arange(n_real))


def draw_relative_weight_noise(
    key: jax.Array, shape: tuple[int, ...], noise_distribution: Literal["rademacher", "gaussian"] = "rademacher", dtype=None
) -> jax.Array:
    r"""
    Draw independent, zero-mean and unit-variance noise, to be used as relative perturbation of weights.

    Parameters
    ----------
    key : jax.Array
        Random key.
    shape : tuple[int, ...]
        Shape of the noise.
    noise_distribution : Literal["rademacher", "gaussian"], optional
        Distribution of the noise: Rademacher (:math:`\pm 1`) or Gaussian, by default "rademacher"
    dtype : optional
        Type of the noise, by default the default JAX floating point type.

    Returns
    -------
    jax.Array
        The noise.

    Raises
    ------
    ValueError
        If ``noise_distribution`` is unknown.
    """
    dtype = dtype or jnp.result_type(float)
    if noise_distribution == "rademacher":
        return jax.random.rademacher(key, shape, dtype=dtype)
    if noise_distribution == "gaussian":
        return jax.random.normal(key, shape, dtype=dtype)
    raise ValueError(f"Unknown noise distribution {noise_distribution!r}; use 'rademacher' or 'gaussian'.")


def _split_regions(array: jax.Array, sizes: tuple[int, ...]) -> list[jax.Array]:
    """Split a concatenated array back into per-region arrays (sharding-aware)."""
    if len(sizes) == 1:
        return [array]
    split_idx = list(itertools.accumulate(sizes))[:-1]
    return list(local_split(array, split_idx, axis=0, sharding_mesh=get_sharding_mesh()))


def _concat(arrays: list[jax.Array]) -> jax.Array:
    return local_concatenate(list(arrays), axis=0, sharding_mesh=get_sharding_mesh()) if len(arrays) > 1 else arrays[0]


def _as_single(arg, name):
    """Normalize an optional per-tracer argument to a tuple of length 0 or 1 (single tracer)."""
    if arg is None:
        return ()
    if isinstance(arg, tuple):
        if len(arg) != 1:
            raise NotImplementedError(f"Only a single tracer is supported: {name} has {len(arg)} entries.")
        return arg
    return (arg,)


FieldWeightsArgs = make_jax_dataclass(
    class_name="FieldWeightsArgs",
    dynamic_fields=[
        "input_data_weights",
        "input_randoms_weights",
        "data_estimator_weights",
        "randoms_estimator_weights",
        "noise_free_data_to_randoms_ratio",
        "other_data_estimator_weights",
        "other_randoms_estimator_weights",
        "other_noise_free_data_to_randoms_ratio",
        "ric_args",
        "amr_args",
        "nam_args",
        "data_regions",
        "randoms_regions",
    ],
    aux_fields=["n_data_per_region", "n_randoms_per_region", "gic", "n_legs"],
    types_fields={
        "input_data_weights": jax.Array,
        "input_randoms_weights": jax.Array,
        "data_estimator_weights": jax.Array,
        "randoms_estimator_weights": jax.Array,
        "noise_free_data_to_randoms_ratio": list,
        "other_data_estimator_weights": jax.Array | None,
        "other_randoms_estimator_weights": jax.Array | None,
        "other_noise_free_data_to_randoms_ratio": list | None,
        "ric_args": tuple,
        "amr_args": tuple,
        "nam_args": tuple,
        "data_regions": tuple,
        "randoms_regions": tuple,
        "n_data_per_region": tuple,
        "n_randoms_per_region": tuple,
        "gic": bool,
        "n_legs": int,
    },
)


def prepare_field_weights(
    *fkp_fields: FKPField,
    estimator_weights: str | None | tuple[str | None, str | None] = None,
    ric_args: RIC_args | None = None,
    amr_args: AMR_args | None = None,
    nam_args: NAM_args | None = None,
    data_regions: jax.Array | None = None,
    randoms_regions: jax.Array | None = None,
    gic: bool = True,
) -> FieldWeightsArgs:
    r"""
    Prepare arguments necessary to computing the weights of the measured field in :py:func:`apply_field_weights`.

    Parameters
    ----------
    *fkp_fields : FKPField
        One FKP field per region (*e.g.* NGC and SGC). The "data" must be a random half playing the data (not clustered); its weights are the noise-free input data weights. The data to randoms split and the weights should look like those of the real measurement.
    estimator_weights : str | None | tuple[str | None, str | None], optional
        Name of the weights stored in the FKP fields' particle fields ``extra_fields`` to use as extra weight at estimation time (FKP or OQE weights for example). They are frozen at their noise-free values. By default ``None`` (no extra weight). A pair designates two estimator weightings (*legs*) applied to the same particles, *e.g.* an OQE pair (``W_TILDE``, ``W_ell``): the measured spectrum is then the symmetrized cross spectrum of the two legs, and the global integral constraint (and the frozen data to randoms ratio) is applied per leg, with the sums weighted by the weights of that leg.
    ric_args : RIC_args | None, optional
        Fixed, precomputed arguments for RIC weights computation by :py:func:`desiwinds.forward.apply_RIC`. Obtain with :py:func:`desiwinds.forward.prepare_RIC`. By default ``None``.
    amr_args : AMR_args | None, optional
        Fixed, precomputed arguments for AMR weights computation by :py:func:`desiwinds.forward.apply_AMR`. Obtain with :py:func:`desiwinds.forward.prepare_AMR`. By default ``None``.
    nam_args : NAM_args | None, optional
        Fixed, precomputed arguments for NAM weights computation by :py:func:`desiwinds.forward.apply_NAM`. Obtain with :py:func:`desiwinds.forward.prepare_NAM`. By default ``None``.
    data_regions : jax.Array | None, optional
        Regions for the data to randoms renormalization. By default ``None``. These can typically be provided as the ``data_regions`` attribute in ``ric_args``, ``amr_args`` or ``nam_args``.
    randoms_regions : jax.Array | None, optional
        Regions for the data to randoms renormalization. By default ``None``. These can typically be provided as the ``randoms_regions`` attribute in ``ric_args``, ``amr_args`` or ``nam_args``.
    gic : bool, optional
        Whether :math:`\alpha` (the ratio of the data to randoms weights) responds to the data weights, *i.e.* whether to apply the global integral constraint. Forced to ``True`` if any additional effect is set. By default ``True``.

    Returns
    -------
    FieldWeightsArgs
        Custom pytree class that contains all necessary information to compute the weights of the measured field:

        * ``input_data_weights``: concatenated noise-free input data weights,
        * ``input_randoms_weights``: concatenated input randoms weights,
        * ``data_estimator_weights``, ``randoms_estimator_weights``: frozen estimator weights of the data and the randoms (first leg),
        * ``noise_free_data_to_randoms_ratio``: noise-free ratio of the data to randoms weights of each region (first leg), used only if ``gic`` is ``False``. the sums are weighted by the estimator weights of the leg, like the ratio of the corresponding measurement,
        * ``other_data_estimator_weights``, ``other_randoms_estimator_weights``, ``other_noise_free_data_to_randoms_ratio``: the same for the second leg, ``None`` with a single leg,
        * ``ric_args``, ``amr_args``, ``nam_args``, ``data_regions``, ``randoms_regions``: effects and region masks, as tuples of length 0 or 1,
        * ``n_data_per_region``, ``n_randoms_per_region``, ``gic``, ``n_legs`` (auxiliary): number of data and randoms objects of each region, the effective ``gic`` and the number of estimator weightings (1 or 2).

    Raises
    ------
    NotImplementedError
        If cross-correlations of several tracers are requested (tuples of FKP fields): only one tracer is supported, with one or two estimator weightings.
    ValueError
        If ``estimator_weights`` is a tuple of length other than 2.
    """
    if any(isinstance(f, tuple) for f in fkp_fields):
        raise NotImplementedError("Only a single tracer is supported (no tuples of FKP fields); pass a tuple of estimator weights for two weightings.")
    if isinstance(estimator_weights, tuple) and len(estimator_weights) != 2:
        raise ValueError(f"estimator_weights must be a name, None or a pair, got {len(estimator_weights)} entries.")
    ric_args, amr_args, nam_args = (_as_single(a, n) for a, n in ((ric_args, "ric_args"), (amr_args, "amr_args"), (nam_args, "nam_args")))
    data_regions, randoms_regions = _as_single(data_regions, "data_regions"), _as_single(randoms_regions, "randoms_regions")
    if (not gic) and any((ric_args, amr_args, nam_args, data_regions, randoms_regions)):
        gic = True
    input_data_weights = _concat([f.data.weights for f in fkp_fields])
    input_randoms_weights = _concat([f.randoms.weights for f in fkp_fields])

    def leg_weights(name):
        if name is None:
            return jnp.ones_like(input_data_weights), jnp.ones_like(input_randoms_weights)
        return tuple(_concat([getattr(f, which).extra[name] for f in fkp_fields]) for which in ("data", "randoms"))

    def weighted_ratios(name):
        if name is None:
            return [f.data.weights.sum() / f.randoms.weights.sum() for f in fkp_fields]
        return [(f.data.weights * f.data.extra[name]).sum() / (f.randoms.weights * f.randoms.extra[name]).sum() for f in fkp_fields]

    if isinstance(estimator_weights, tuple):
        n_legs = 2
        (data_estimator_weights, randoms_estimator_weights), (other_data_estimator_weights, other_randoms_estimator_weights) = (leg_weights(n) for n in estimator_weights)
        ratio, other_ratio = (weighted_ratios(n) for n in estimator_weights)
    else:
        n_legs = 1
        data_estimator_weights, randoms_estimator_weights = leg_weights(estimator_weights)
        other_data_estimator_weights = other_randoms_estimator_weights = other_ratio = None
        ratio = weighted_ratios(estimator_weights)
    return FieldWeightsArgs(
        input_data_weights=input_data_weights,
        input_randoms_weights=input_randoms_weights,
        data_estimator_weights=data_estimator_weights,
        randoms_estimator_weights=randoms_estimator_weights,
        noise_free_data_to_randoms_ratio=ratio,
        other_data_estimator_weights=other_data_estimator_weights,
        other_randoms_estimator_weights=other_randoms_estimator_weights,
        other_noise_free_data_to_randoms_ratio=other_ratio,
        ric_args=ric_args,
        amr_args=amr_args,
        nam_args=nam_args,
        data_regions=data_regions,
        randoms_regions=randoms_regions,
        n_data_per_region=tuple(int(f.data.weights.shape[0]) for f in fkp_fields),
        n_randoms_per_region=tuple(int(f.randoms.weights.shape[0]) for f in fkp_fields),
        gic=gic,
        n_legs=n_legs,
    )


def _legs(args: FieldWeightsArgs) -> list[tuple]:
    """Estimator weights of each leg, as ``(data_estimator_weights, randoms_estimator_weights, noise_free_data_to_randoms_ratio)``."""
    legs = [(args.data_estimator_weights, args.randoms_estimator_weights, args.noise_free_data_to_randoms_ratio)]
    if args.n_legs == 2:
        legs.append((args.other_data_estimator_weights, args.other_randoms_estimator_weights, args.other_noise_free_data_to_randoms_ratio))
    return legs


def apply_field_weights(data_weights: jax.Array, field_weights_args: FieldWeightsArgs) -> tuple:
    r"""
    Compute the weights of the measured field for the given input data weights, one array per region (and per leg for two estimator weightings).

    RIC, AMR, NAM and the data to randoms renormalization are applied once, then, for each leg, the frozen estimator weights and the global integral constraint. Each returned array is the data block (``data weights * estimator weights``) followed by the randoms block (``-alpha * randoms weights * estimator weights``), where ``alpha`` is the ratio of the weighted sums of the leg (or its frozen noise-free value if ``gic`` is ``False``).

    Parameters
    ----------
    data_weights : jax.Array
        Input data weights, concatenated over the regions and of the same shape as ``field_weights_args.input_data_weights`` (which are the noise-free ones).
    field_weights_args : FieldWeightsArgs
        Fixed arguments, obtained with :py:func:`prepare_field_weights`.

    Returns
    -------
    tuple
        With a single estimator weighting, the weights of the field, one array per region. With two, a pair of such tuples, one per leg ``(leg_A, leg_B)``.

    Notes
    -----
    In the math, this is the pipeline :math:`\Phi(\mathbf u)` and ``data_weights`` is :math:`\mathbf u`.
    """
    args = field_weights_args
    data_weights_after_effects, randoms_weights_after_effects = _apply_effects(
        [data_weights],
        [args.input_randoms_weights],
        ric_args=args.ric_args,
        amr_args=args.amr_args,
        nam_args=args.nam_args,
        data_regions=args.data_regions,
        randoms_regions=args.randoms_regions,
    )
    per_leg = tuple(
        tuple(
            _concat([data_block, -(data_block.sum() / randoms_block.sum() if args.gic else ratio[iregion]) * randoms_block])
            for iregion, (data_block, randoms_block) in enumerate(
                zip(
                    _split_regions(data_weights_after_effects[0] * data_estimator_weights, args.n_data_per_region),
                    _split_regions(randoms_weights_after_effects[0] * randoms_estimator_weights, args.n_randoms_per_region),
                    strict=True,
                )
            )
        )
        for data_estimator_weights, randoms_estimator_weights, ratio in _legs(args)
    )
    return per_leg[0] if args.n_legs == 1 else per_leg


def field_weights_perturbation_gic_only(relative_noise: jax.Array, field_weights_args: FieldWeightsArgs) -> tuple:
    r"""
    Compute the first-order response of the field weights to relative noise on the data weights, when only the global integral constraint reacts.

    This is a closed form for the simplified pipeline without RIC, AMR, NAM nor region renormalization (those of ``field_weights_args`` are ignored): the data weights are perturbed by ``input_data_weights * relative_noise`` and the randoms weights by the induced change of ``alpha``. It is the control variate of :py:func:`shotnoise_template_with_control_variate`.

    Parameters
    ----------
    relative_noise : jax.Array
        Relative noise on the data weights, of the same shape as ``field_weights_args.input_data_weights``.
    field_weights_args : FieldWeightsArgs
        Fixed arguments, obtained with :py:func:`prepare_field_weights`.

    Returns
    -------
    tuple
        Perturbation of the weights of the field, one array per region (a pair of such tuples, one per leg, for two estimator weightings).

    Notes
    -----
    In the math, this is :math:`J_A \mathbf d` with :math:`\mathbf d = \mathbf u_0\boldsymbol\epsilon`. For leg :math:`X` with estimator weights :math:`w_X`, the perturbation is :math:`[u_0\epsilon w_X, -(\sum u_0\epsilon w_X / \sum r w_X)\, r w_X]`.
    """
    args = field_weights_args
    per_leg = tuple(
        tuple(
            _concat([data_block, -(data_block.sum() / randoms_block.sum() if args.gic else 0.0) * randoms_block])
            for data_block, randoms_block in zip(
                _split_regions(args.input_data_weights * relative_noise * data_estimator_weights, args.n_data_per_region),
                _split_regions(args.input_randoms_weights * randoms_estimator_weights, args.n_randoms_per_region),
                strict=True,
            )
        )
        for data_estimator_weights, randoms_estimator_weights, _ in _legs(args)
    )
    return per_leg[0] if args.n_legs == 1 else per_leg


def get_region_particles(*fkp_fields: FKPField) -> list[ParticleField]:
    """
    Get the particles of each region, data first and then randoms, to be used by :py:func:`bilinear_power_spectrum`.

    Parameters
    ----------
    *fkp_fields : FKPField
        One FKP field per region.

    Returns
    -------
    list[ParticleField]
        One particle field per region. Only the positions matter: the weights are placeholders, replaced when painting.
    """
    return [
        ParticleField(_concat([f.data.positions, f.randoms.positions]), weights=_concat([f.data.weights, f.randoms.weights]), attrs=f.attrs) for f in fkp_fields
    ]


def get_estimator_normalizations(fkp_norms: list) -> jax.Array:
    """
    Get the estimator normalization of each region.

    Parameters
    ----------
    fkp_norms : list
        Pre-computed power spectrum norms for the FKP fields, one per region, as returned by :py:func:`jaxpower.compute_fkp2_normalization`.

    Returns
    -------
    jax.Array
        Normalization of each region. The first element of each entry of ``fkp_norms`` is used: they must be uniform across multipoles and bins.
    """
    return jnp.stack([jnp.ravel(jnp.asarray(n))[0] for n in fkp_norms])


def _region_spectrum(mesh, other_mesh, binner, norm, los):
    """``(n_ells, n_k)`` spectrum of a mesh (or cross spectrum of two meshes) divided by the estimator normalization."""
    spectrum = compute_mesh2_spectrum(*((mesh,) if other_mesh is None else (mesh, other_mesh)), bin=binner, los=los)
    # lsstypes' ``value()`` concatenates with numpy and cannot be traced: rebuild num_raw / norm from the per-multipole leaves
    poles = [spectrum.get(ell).values() for ell in binner.ells]
    return jnp.stack([(pole["value"] * pole["norm"] + pole["num_shotnoise"]) / norm for pole in poles])


def bilinear_power_spectrum(
    field_weights: tuple[jax.Array, ...],
    other_field_weights: tuple[jax.Array, ...] | None = None,
    *,
    particles: list[ParticleField],
    binner: BinMesh2SpectrumPoles,
    norms: jax.Array,
    los: Literal["local", "x", "y", "z"] = "local",
) -> jax.Array:
    r"""
    Compute the power spectrum multipoles of fields with the given weights, which are bilinear in them.

    The normalization is frozen and no shot noise is subtracted. The fields are painted with TSC and interlacing 3 and compensated, as in :py:func:`desiwinds.forward.mock_survey_catalog`.

    Parameters
    ----------
    field_weights : tuple[jax.Array, ...]
        Weights of the field, one array per region (data block, then randoms block), *e.g.* an output of :py:func:`apply_field_weights`.
    other_field_weights : tuple[jax.Array, ...] | None, optional
        Weights of a second field, in which case the cross spectrum, symmetrized between the two fields, is returned. By default ``None`` (auto spectrum of ``field_weights``).
    particles : list[ParticleField]
        Particles of each region, data first and then randoms. Obtain with :py:func:`get_region_particles`.
    binner : BinMesh2SpectrumPoles
        Binning operator for the power spectrum estimation, same for all regions.
    norms : jax.Array
        Frozen estimator normalization of each region. Obtain with :py:func:`get_estimator_normalizations`.
    los : Literal["local", "x", "y", "z"], optional
        Line of sight, by default "local"

    Returns
    -------
    jax.Array
        Power spectrum multipoles, of shape ``(n_regions, n_ells, n_k)``.

    Notes
    -----
    In the math, this is :math:`Q_{\ell b}(\mathbf w, \mathbf z)`.
    """
    los = {"local": "firstpoint"}.get(los, los)

    def paint(weights):
        return [p.clone(weights=w).paint(resampler="tsc", interlacing=3, compensate=True, out="real") for p, w in zip(particles, weights, strict=True)]

    meshes = paint(field_weights)
    if other_field_weights is None:
        return jnp.stack([_region_spectrum(mesh, None, binner, norms[iregion], los) for iregion, mesh in enumerate(meshes)])
    other_meshes = paint(other_field_weights)
    return jnp.stack(
        [
            0.5 * (_region_spectrum(mesh, other_mesh, binner, norms[iregion], los) + _region_spectrum(other_mesh, mesh, binner, norms[iregion], los))
            for iregion, (mesh, other_mesh) in enumerate(zip(meshes, other_meshes, strict=True))
        ]
    )


def conventional_shotnoise(
    field_weights: tuple[jax.Array, ...], other_field_weights: tuple[jax.Array, ...] | None = None, *, binner: BinMesh2SpectrumPoles, norms: jax.Array
) -> jax.Array:
    r"""
    Compute the conventional constant shot noise of fields with the given weights, which is bilinear in them.

    This is the sum over the objects of the product of the weights, over the normalization. As in jaxpower's conventional shot noise, it is multiplied by the fraction of modes kept by the ``klimit`` of ``binner`` (1 if there is none).

    Parameters
    ----------
    field_weights : tuple[jax.Array, ...]
        Weights of the field, one array per region (data block, then randoms block).
    other_field_weights : tuple[jax.Array, ...] | None, optional
        Weights of a second field. By default ``None`` (same as ``field_weights``).
    binner : BinMesh2SpectrumPoles
        Binning operator for the power spectrum estimation.
    norms : jax.Array
        Frozen estimator normalization of each region. Obtain with :py:func:`get_estimator_normalizations`.

    Returns
    -------
    jax.Array
        Conventional shot noise of each region, of shape ``(n_regions,)``.

    Notes
    -----
    In the math, this is :math:`S(\mathbf w, \mathbf z) = \sum_a w_a z_a / I_0`.
    """
    other_field_weights = field_weights if other_field_weights is None else other_field_weights
    mode_scale = 1.0
    klimit = getattr(binner, "klimit", None)
    if klimit is not None:
        knorm = jnp.sqrt(sum(kk**2 for kk in binner.mattrs.kcoords(sparse=True)))
        mode_scale = jnp.sum((knorm >= klimit[0]) & (knorm <= klimit[-1])) / knorm.size
    return mode_scale * jnp.stack([jnp.sum(w * other_w) / norm for w, other_w, norm in zip(field_weights, other_field_weights, norms, strict=True)])


# The ``sample_shotnoise_template_*`` functions return a dict of arrays (a pytree, as required by ``jax.jit``): ``power_spectrum_response`` of shape
# (n_real, n_regions, n_ells, n_k), ``shotnoise_response`` of shape (n_real, n_regions) and ``extras``, a dict of (power_spectrum_response, shotnoise_response) pairs.
# These are the numerator and denominator of the template in the math.
_STATIC_ARGNAMES = ["n_real", "noise_distribution", "compute_control_variate", "batch_size", "los"]


def _of_legs(of, field_weights, n_legs):
    """Evaluate the bilinear ``of`` (spectrum or shot noise) on the field weights: auto of one leg, or symmetrized cross of the two legs."""
    return of(field_weights) if n_legs == 1 else of(*field_weights)


def _curvature_of(of, noise_free_field_weights, second_order, n_legs):
    """Curvature term of the second-order expansion of ``of`` around the noise-free field weights: ``Q(Phi_0, Phi_2)``, or ``(Q(Phi_0A, Phi_2B) + Q(Phi_2A, Phi_0B)) / 2`` for two legs."""
    if n_legs == 1:
        return of(noise_free_field_weights, second_order)
    return 0.5 * (of(noise_free_field_weights[0], second_order[1]) + of(second_order[0], noise_free_field_weights[1]))


def _responses(spectrum_of, shotnoise_of, field_weights, n_legs):
    return _of_legs(spectrum_of, field_weights, n_legs), _of_legs(shotnoise_of, field_weights, n_legs)


def _control_variate_responses(relative_noise, field_weights_args, spectrum_of, shotnoise_of):
    field_weights = field_weights_perturbation_gic_only(relative_noise, field_weights_args)
    return _responses(spectrum_of, shotnoise_of, field_weights, field_weights_args.n_legs)


def _antithetic_step(
    key, field_weights_args, spectrum_of, shotnoise_of, noise_free_spectrum, noise_free_shotnoise, sigma, noise_distribution, compute_control_variate
):
    data_weights = field_weights_args.input_data_weights
    relative_noise = draw_relative_weight_noise(key, data_weights.shape, noise_distribution, data_weights.dtype)
    field_weights_plus = apply_field_weights(data_weights * (1 + sigma * relative_noise), field_weights_args)
    field_weights_minus = apply_field_weights(data_weights * (1 - sigma * relative_noise), field_weights_args)
    extras = {"control_variate": _control_variate_responses(relative_noise, field_weights_args, spectrum_of, shotnoise_of)} if compute_control_variate else {}
    spectrum_plus, shotnoise_plus = _responses(spectrum_of, shotnoise_of, field_weights_plus, field_weights_args.n_legs)
    spectrum_minus, shotnoise_minus = _responses(spectrum_of, shotnoise_of, field_weights_minus, field_weights_args.n_legs)
    return {
        "power_spectrum_response": 0.5 * (spectrum_plus + spectrum_minus) - noise_free_spectrum,
        "shotnoise_response": 0.5 * (shotnoise_plus + shotnoise_minus) - noise_free_shotnoise,
        "extras": extras,
    }


def _linearized_step(key, field_weights_args, spectrum_of, shotnoise_of, noise_distribution, compute_control_variate):
    data_weights = field_weights_args.input_data_weights
    relative_noise = draw_relative_weight_noise(key, data_weights.shape, noise_distribution, data_weights.dtype)
    _, first_order_field_weights = jax.jvp(
        partial(apply_field_weights, field_weights_args=field_weights_args), (data_weights,), (data_weights * relative_noise,)
    )
    extras = {"control_variate": _control_variate_responses(relative_noise, field_weights_args, spectrum_of, shotnoise_of)} if compute_control_variate else {}
    spectrum_response, shotnoise_response = _responses(spectrum_of, shotnoise_of, first_order_field_weights, field_weights_args.n_legs)
    return {"power_spectrum_response": spectrum_response, "shotnoise_response": shotnoise_response, "extras": extras}


def _quadratic_step(key, field_weights_args, noise_free_field_weights, spectrum_of, shotnoise_of, noise_distribution, compute_control_variate):
    data_weights = field_weights_args.input_data_weights
    relative_noise = draw_relative_weight_noise(key, data_weights.shape, noise_distribution, data_weights.dtype)
    weight_perturbation = data_weights * relative_noise
    field_weights_of = partial(apply_field_weights, field_weights_args=field_weights_args)
    # forward-over-forward: first and second directional derivatives of the field weights along the perturbation
    first_order, second_order = jax.jvp(lambda x: jax.jvp(field_weights_of, (x,), (weight_perturbation,))[1], (data_weights,), (weight_perturbation,))
    n_legs = field_weights_args.n_legs
    linear_part = _responses(spectrum_of, shotnoise_of, first_order, n_legs)
    curvature_part = (_curvature_of(spectrum_of, noise_free_field_weights, second_order, n_legs), _curvature_of(shotnoise_of, noise_free_field_weights, second_order, n_legs))
    extras = {"linear_part": linear_part, "curvature_part": curvature_part}
    if compute_control_variate:
        extras["control_variate"] = _control_variate_responses(relative_noise, field_weights_args, spectrum_of, shotnoise_of)
    return {"power_spectrum_response": linear_part[0] + curvature_part[0], "shotnoise_response": linear_part[1] + curvature_part[1], "extras": extras}


@jax.jit(static_argnames=_STATIC_ARGNAMES)
def sample_shotnoise_template_antithetic(
    *fkp_fields: FKPField,
    field_weights_args: FieldWeightsArgs,
    key: jax.Array,
    n_real: int,
    binner: BinMesh2SpectrumPoles,
    fkp_norms: list,
    sigma: float = 1.0,
    noise_distribution: Literal["rademacher", "gaussian"] = "rademacher",
    compute_control_variate: bool = False,
    batch_size: int | None = None,
    los: Literal["local", "x", "y", "z"] = "local",
) -> dict:
    r"""
    Sample the shot-noise template by antithetic noise injection with normalization.

    For each realization, relative noise of amplitude ``sigma`` is injected on the input data weights with opposite signs, and the whole pipeline is run on both. The power spectrum response is the mean of the two unsubtracted spectra minus the noise-free one, and the shot noise response is the same combination of the conventional shot noises. The template is the ratio of their sums over the realizations (:py:func:`shotnoise_template_from_samples`). It is biased at order ``sigma**2``: keep ``sigma <= 1``. Rademacher noise with ``sigma=1`` gives weights 0 or 2. The whole run is jitted, with the realizations in a :py:func:`jax.lax.map`.

    Parameters
    ----------
    *fkp_fields : FKPField
        One FKP field per region, as given to :py:func:`prepare_field_weights`; only their positions are used here.
    field_weights_args : FieldWeightsArgs
        Fixed arguments, obtained with :py:func:`prepare_field_weights`.
    key : jax.Array
        Random key. Realization ``r`` uses ``fold_in(key, r)`` (see :py:func:`make_realization_keys`), so that all ``sample_shotnoise_template_*`` functions share their draws.
    n_real : int
        Number of realizations. It is static: changing it triggers a new compilation.
    binner : BinMesh2SpectrumPoles
        Binning operator for the power spectrum estimation, same for all regions.
    fkp_norms : list
        Pre-computed power spectrum norms for the FKP fields, one per region, disregarding any future changes in weights. See :py:func:`get_estimator_normalizations`.
    sigma : float, optional
        Amplitude of the relative noise, by default 1.0
    noise_distribution : Literal["rademacher", "gaussian"], optional
        Distribution of the relative noise on the data weights, by default "rademacher"
    compute_control_variate : bool, optional
        Whether to also store the responses of the simplified pipeline, used by :py:func:`shotnoise_template_with_control_variate` (``extras["control_variate"]``). By default ``False``.
    batch_size : int | None, optional
        Number of realizations evaluated in parallel by :py:func:`jax.lax.map`. By default ``None`` (one after the other). Memory grows with it.
    los : Literal["local", "x", "y", "z"], optional
        Line of sight, by default "local"

    Returns
    -------
    dict
        Samples, with a leading realization axis: ``power_spectrum_response`` of shape ``(n_real, n_regions, n_ells, n_k)``, ``shotnoise_response`` of shape ``(n_real, n_regions)`` and ``extras``, a dictionary of ``(power_spectrum_response, shotnoise_response)`` pairs. Pass it to :py:func:`shotnoise_template_from_samples`.

    Notes
    -----
    In the math, the responses are the numerator :math:`\frac12(P_+ + P_-) - P_0` and the denominator :math:`\frac12(S_+ + S_-) - S_0` of the template. Gaussian noise with large ``sigma`` can produce negative weights.
    """
    particles, norms = get_region_particles(*fkp_fields), get_estimator_normalizations(fkp_norms)
    spectrum_of = partial(bilinear_power_spectrum, particles=particles, binner=binner, norms=norms, los=los)
    shotnoise_of = partial(conventional_shotnoise, binner=binner, norms=norms)
    noise_free_field_weights = apply_field_weights(field_weights_args.input_data_weights, field_weights_args)
    step = partial(
        _antithetic_step,
        field_weights_args=field_weights_args,
        spectrum_of=spectrum_of,
        shotnoise_of=shotnoise_of,
        noise_free_spectrum=_of_legs(spectrum_of, noise_free_field_weights, field_weights_args.n_legs),
        noise_free_shotnoise=_of_legs(shotnoise_of, noise_free_field_weights, field_weights_args.n_legs),
        sigma=sigma,
        noise_distribution=noise_distribution,
        compute_control_variate=compute_control_variate,
    )
    return jax.lax.map(step, make_realization_keys(key, n_real), batch_size=batch_size)


@jax.jit(static_argnames=_STATIC_ARGNAMES)
def sample_shotnoise_template_linearized(
    *fkp_fields: FKPField,
    field_weights_args: FieldWeightsArgs,
    key: jax.Array,
    n_real: int,
    binner: BinMesh2SpectrumPoles,
    fkp_norms: list,
    noise_distribution: Literal["rademacher", "gaussian"] = "rademacher",
    compute_control_variate: bool = False,
    batch_size: int | None = None,
    los: Literal["local", "x", "y", "z"] = "local",
) -> dict:
    """
    Sample the linearized shot-noise template, by forward-mode differentiation of the pipeline.

    For each realization, the first-order response of the field weights to relative noise on the input data weights is obtained with :py:func:`jax.jvp`. The power spectrum response and the shot noise response are the unsubtracted spectrum and the conventional shot noise of that response. No noise amplitude is involved, and the curvature of the pipeline (RIC, AMR...) is dropped: use :py:func:`sample_shotnoise_template_quadratic` to keep it. The randoms block of the response (how RIC, AMR and alpha react to the noise) is kept in both.

    Parameters
    ----------
    *fkp_fields : FKPField
        One FKP field per region, as given to :py:func:`prepare_field_weights`; only their positions are used here.
    field_weights_args : FieldWeightsArgs
        Fixed arguments, obtained with :py:func:`prepare_field_weights`.
    key : jax.Array
        Random key. Realization ``r`` uses ``fold_in(key, r)`` (see :py:func:`make_realization_keys`), so that all ``sample_shotnoise_template_*`` functions share their draws.
    n_real : int
        Number of realizations. It is static: changing it triggers a new compilation.
    binner : BinMesh2SpectrumPoles
        Binning operator for the power spectrum estimation, same for all regions.
    fkp_norms : list
        Pre-computed power spectrum norms for the FKP fields, one per region, disregarding any future changes in weights. See :py:func:`get_estimator_normalizations`.
    noise_distribution : Literal["rademacher", "gaussian"], optional
        Distribution of the relative noise on the data weights, by default "rademacher"
    compute_control_variate : bool, optional
        Whether to also store the responses of the simplified pipeline, used by :py:func:`shotnoise_template_with_control_variate` (``extras["control_variate"]``). By default ``False``.
    batch_size : int | None, optional
        Number of realizations evaluated in parallel by :py:func:`jax.lax.map`. By default ``None`` (one after the other). Memory grows with it.
    los : Literal["local", "x", "y", "z"], optional
        Line of sight, by default "local"

    Returns
    -------
    dict
        Samples, with a leading realization axis: ``power_spectrum_response`` of shape ``(n_real, n_regions, n_ells, n_k)``, ``shotnoise_response`` of shape ``(n_real, n_regions)`` and ``extras``, a dictionary of ``(power_spectrum_response, shotnoise_response)`` pairs. Pass it to :py:func:`shotnoise_template_from_samples`.

    Examples
    --------
    >>> field_weights_args = prepare_field_weights(
            *fkp_fields,
            estimator_weights="weight_FKP",
            ric_args=ric_args,
            amr_args=amr_args,
            data_regions=ric_args.data_regions,
            randoms_regions=ric_args.randoms_regions,
        )
    >>> samples = sample_shotnoise_template_linearized(
            *fkp_fields,
            field_weights_args=field_weights_args,
            key=jax.random.key(42),
            n_real=100,
            binner=binner,
            fkp_norms=fkp_norms,
        )
    >>> template, covariance = shotnoise_template_from_samples(samples, norms)
    """
    particles, norms = get_region_particles(*fkp_fields), get_estimator_normalizations(fkp_norms)
    step = partial(
        _linearized_step,
        field_weights_args=field_weights_args,
        spectrum_of=partial(bilinear_power_spectrum, particles=particles, binner=binner, norms=norms, los=los),
        shotnoise_of=partial(conventional_shotnoise, binner=binner, norms=norms),
        noise_distribution=noise_distribution,
        compute_control_variate=compute_control_variate,
    )
    return jax.lax.map(step, make_realization_keys(key, n_real), batch_size=batch_size)


@jax.jit(static_argnames=_STATIC_ARGNAMES)
def sample_shotnoise_template_quadratic(
    *fkp_fields: FKPField,
    field_weights_args: FieldWeightsArgs,
    key: jax.Array,
    n_real: int,
    binner: BinMesh2SpectrumPoles,
    fkp_norms: list,
    noise_distribution: Literal["rademacher", "gaussian"] = "rademacher",
    compute_control_variate: bool = False,
    batch_size: int | None = None,
    los: Literal["local", "x", "y", "z"] = "local",
) -> dict:
    """
    Sample the shot-noise template including the curvature of the pipeline, by nested forward-mode differentiation.

    For each realization, a forward-over-forward pass gives the first- and second-order responses of the field weights to relative noise on the input data weights. The responses are the sum of a linear part (the spectrum and conventional shot noise of the first-order response) and of a curvature part (the cross spectrum and cross shot noise of the noise-free field weights with the second-order response). This is exact at quadratic order and involves no noise amplitude.

    ``extras`` contains, from the same draws, ``"linear_part"`` (what :py:func:`sample_shotnoise_template_linearized` gives) and ``"curvature_part"``, so that :py:func:`shotnoise_template_curvature_shift` is available. The curvature depends on the noise-free field weights: use a clustered mock as "data" to get the one relevant for the data.

    Parameters
    ----------
    *fkp_fields : FKPField
        One FKP field per region, as given to :py:func:`prepare_field_weights`; only their positions are used here.
    field_weights_args : FieldWeightsArgs
        Fixed arguments, obtained with :py:func:`prepare_field_weights`.
    key : jax.Array
        Random key. Realization ``r`` uses ``fold_in(key, r)`` (see :py:func:`make_realization_keys`), so that all ``sample_shotnoise_template_*`` functions share their draws.
    n_real : int
        Number of realizations. It is static: changing it triggers a new compilation.
    binner : BinMesh2SpectrumPoles
        Binning operator for the power spectrum estimation, same for all regions.
    fkp_norms : list
        Pre-computed power spectrum norms for the FKP fields, one per region, disregarding any future changes in weights. See :py:func:`get_estimator_normalizations`.
    noise_distribution : Literal["rademacher", "gaussian"], optional
        Distribution of the relative noise on the data weights, by default "rademacher"
    compute_control_variate : bool, optional
        Whether to also store the responses of the simplified pipeline, used by :py:func:`shotnoise_template_with_control_variate` (``extras["control_variate"]``). By default ``False``.
    batch_size : int | None, optional
        Number of realizations evaluated in parallel by :py:func:`jax.lax.map`. By default ``None`` (one after the other). Memory grows with it.
    los : Literal["local", "x", "y", "z"], optional
        Line of sight, by default "local"

    Returns
    -------
    dict
        Samples, with a leading realization axis: ``power_spectrum_response`` of shape ``(n_real, n_regions, n_ells, n_k)``, ``shotnoise_response`` of shape ``(n_real, n_regions)`` and ``extras``, a dictionary of ``(power_spectrum_response, shotnoise_response)`` pairs. Pass it to :py:func:`shotnoise_template_from_samples`.
    """
    particles, norms = get_region_particles(*fkp_fields), get_estimator_normalizations(fkp_norms)
    noise_free_field_weights = apply_field_weights(field_weights_args.input_data_weights, field_weights_args)
    step = partial(
        _quadratic_step,
        field_weights_args=field_weights_args,
        noise_free_field_weights=noise_free_field_weights,
        spectrum_of=partial(bilinear_power_spectrum, particles=particles, binner=binner, norms=norms, los=los),
        shotnoise_of=partial(conventional_shotnoise, binner=binner, norms=norms),
        noise_distribution=noise_distribution,
        compute_control_variate=compute_control_variate,
    )
    return jax.lax.map(step, make_realization_keys(key, n_real), batch_size=batch_size)


@jax.jit(static_argnames=["los"])
def measure_power_spectrum_and_shotnoise(
    *fkp_fields: FKPField,
    field_weights_args: FieldWeightsArgs,
    binner: BinMesh2SpectrumPoles,
    fkp_norms: list,
    los: Literal["local", "x", "y", "z"] = "local",
) -> tuple[jax.Array, jax.Array]:
    """
    Measure the unsubtracted power spectrum and the conventional shot noise of the noise-free pipeline (symmetrized cross of the two legs for two estimator weightings).

    On unclustered Poisson catalogs, the mean of the first output is the true shot noise contribution.

    Parameters
    ----------
    *fkp_fields : FKPField
        One FKP field per region, as given to :py:func:`prepare_field_weights`.
    field_weights_args : FieldWeightsArgs
        Fixed arguments, obtained with :py:func:`prepare_field_weights`.
    binner : BinMesh2SpectrumPoles
        Binning operator for the power spectrum estimation, same for all regions.
    fkp_norms : list
        Pre-computed power spectrum norms for the FKP fields, one per region. See :py:func:`get_estimator_normalizations`.
    los : Literal["local", "x", "y", "z"], optional
        Line of sight, by default "local"

    Returns
    -------
    tuple[jax.Array, jax.Array]
        Power spectrum multipoles without any shot noise subtraction, of shape ``(n_regions, n_ells, n_k)``, and conventional shot noise, of shape ``(n_regions,)``.
    """
    norms = get_estimator_normalizations(fkp_norms)
    noise_free_field_weights = apply_field_weights(field_weights_args.input_data_weights, field_weights_args)
    n_legs = field_weights_args.n_legs
    spectrum_of = partial(bilinear_power_spectrum, particles=get_region_particles(*fkp_fields), binner=binner, norms=norms, los=los)
    shotnoise_of = partial(conventional_shotnoise, binner=binner, norms=norms)
    return _of_legs(spectrum_of, noise_free_field_weights, n_legs), _of_legs(shotnoise_of, noise_free_field_weights, n_legs)


def combine_regions_weighted_by_normalization(
    numerator: np.ndarray, denominator: np.ndarray, norms: np.ndarray, regions: list[int] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Combine regions as the data measurement does: sum weighted by the estimator normalizations of the regions, over their sum.

    Parameters
    ----------
    numerator : np.ndarray
        Shape ``(n_real, n_regions, n_ells, n_k)``.
    denominator : np.ndarray
        Shape ``(n_real, n_regions)``.
    norms : np.ndarray
        Estimator normalization of each region, shape ``(n_regions,)``.
    regions : list[int] | None, optional
        Subset of regions to combine, by default ``None`` (all).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Combined numerator and denominator, of shapes ``(n_real, n_ells, n_k)`` and ``(n_real,)``. Both carry the same (arbitrary) overall factor, which cancels in their ratio.
    """
    selection = slice(None) if regions is None else list(regions)
    region_weights = np.asarray(norms)[selection]
    return np.einsum("r,irlk->ilk", region_weights, np.asarray(numerator)[:, selection]), np.einsum(
        "r,ir->i", region_weights, np.asarray(denominator)[:, selection]
    )


def jackknife_ratio_of_means(numerator: np.ndarray, denominator: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the ratio of the sums over realizations of ``numerator`` and ``denominator`` per bin, with its delete-one jackknife covariance.

    Parameters
    ----------
    numerator : np.ndarray
        Shape ``(n_real, ...)``.
    denominator : np.ndarray
        Shape ``(n_real,)``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The ratio, of shape ``(...)``, and its covariance over the flattened bins, of shape ``(size, size)``, *i.e.* including cross-multipole and cross-:math:`k` correlations. This is the covariance **of the ratio** (not of a single realization).

    Raises
    ------
    ValueError
        If there are less than two realizations.
    """
    numerator, denominator = np.asarray(numerator, dtype=float), np.asarray(denominator, dtype=float)
    n_real = numerator.shape[0]
    if n_real < 2:
        raise ValueError("At least two realizations are needed for a jackknife.")
    numerator_sum, denominator_sum = numerator.sum(0), denominator.sum()
    leave_one_out = (numerator_sum[None] - numerator) / (denominator_sum - denominator).reshape((n_real,) + (1,) * (numerator.ndim - 1))
    flat = leave_one_out.reshape(n_real, -1)
    deviations = flat - flat.mean(0)
    return numerator_sum / denominator_sum, (n_real - 1) / n_real * deviations.T @ deviations


def _select(result: dict, norms, regions, extra=None):
    numerator, denominator = (result["power_spectrum_response"], result["shotnoise_response"]) if extra is None else result["extras"][extra]
    return combine_regions_weighted_by_normalization(numerator, denominator, norms, regions)


def shotnoise_template_from_samples(
    result: dict, norms: np.ndarray, regions: list[int] | None = None, extra: str | None = None
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Compute the shot-noise template and its jackknife covariance from samples.

    The template is the ratio of the sums over the realizations of the power spectrum and shot noise responses, with the regions combined. Multiplied by the conventional shot noise of the data monopole, it gives the shot noise contribution to the measured multipoles.

    Parameters
    ----------
    result : dict
        Samples, output of one of the ``sample_shotnoise_template_*`` functions.
    norms : np.ndarray
        Estimator normalization of each region (the first element of each entry of ``fkp_norms``).
    regions : list[int] | None, optional
        Subset of regions to combine, by default ``None`` (all).
    extra : str | None, optional
        Entry of ``result["extras"]`` to use instead of the responses themselves, *e.g.* ``"linear_part"``. By default ``None``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Template of shape ``(n_ells, n_k)`` and covariance over the flattened bins (see :py:func:`jackknife_ratio_of_means`).

    Notes
    -----
    In the math, this is :math:`s_\ell(k)` and the responses are its numerator and denominator.
    """
    return jackknife_ratio_of_means(*_select(result, norms, regions, extra))


def shotnoise_template_uncertainty(result: dict, norms: np.ndarray, regions: list[int] | None = None, extra: str | None = None) -> np.ndarray:
    """
    Compute the jackknife standard deviation of :py:func:`shotnoise_template_from_samples`, of shape ``(n_ells, n_k)``.

    Parameters
    ----------
    result : dict
        Samples, output of one of the ``sample_shotnoise_template_*`` functions.
    norms : np.ndarray
        Estimator normalization of each region.
    regions : list[int] | None, optional
        Subset of regions to combine, by default ``None`` (all).
    extra : str | None, optional
        Entry of ``result["extras"]`` to use instead of the responses themselves. By default ``None``.

    Returns
    -------
    np.ndarray
        Standard deviation of the template.
    """
    template, covariance = shotnoise_template_from_samples(result, norms, regions, extra)
    return np.sqrt(np.diag(covariance)).reshape(template.shape)


def shotnoise_template_curvature_shift(result: dict, norms: np.ndarray, regions: list[int] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the difference between the template with and without the curvature of the pipeline, and its jackknife covariance.

    The two templates come from the same draws and the shift is jackknifed as a whole (not as a difference of independent errors).

    Parameters
    ----------
    result : dict
        Samples, output of :py:func:`sample_shotnoise_template_quadratic`.
    norms : np.ndarray
        Estimator normalization of each region.
    regions : list[int] | None, optional
        Subset of regions to combine, by default ``None`` (all).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Shift of shape ``(n_ells, n_k)`` and covariance over the flattened bins.

    Raises
    ------
    ValueError
        If ``result`` does not come from :py:func:`sample_shotnoise_template_quadratic`.
    """
    if "linear_part" not in result["extras"]:
        raise ValueError("shotnoise_template_curvature_shift requires the output of sample_shotnoise_template_quadratic (extras['linear_part']).")
    numerator, denominator = _select(result, norms, regions)
    linear_numerator, linear_denominator = _select(result, norms, regions, "linear_part")
    n_real = numerator.shape[0]
    shift = numerator.sum(0) / denominator.sum() - linear_numerator.sum(0) / linear_denominator.sum()
    leave_one_out = (numerator.sum(0)[None] - numerator) / (denominator.sum() - denominator).reshape((n_real, 1, 1)) - (
        linear_numerator.sum(0)[None] - linear_numerator
    ) / (linear_denominator.sum() - linear_denominator).reshape((n_real, 1, 1))
    flat = leave_one_out.reshape(n_real, -1)
    deviations = flat - flat.mean(0)
    return shift, (n_real - 1) / n_real * deviations.T @ deviations


def select_realizations(result: dict, indices) -> dict:
    """
    Select realizations, with their extras; *e.g.* for the jackknife resampling of derived statistics.

    Parameters
    ----------
    result : dict
        Samples, output of one of the ``sample_shotnoise_template_*`` functions.
    indices : array-like
        Indices of the realizations to keep, or boolean mask.

    Returns
    -------
    dict
        The selected samples, as numpy arrays.
    """
    return jax.tree.map(lambda x: np.asarray(x)[np.asarray(indices)], result)


@jax.jit(static_argnames=["include_gic", "los"])
def analytic_shotnoise_template(
    *fkp_fields: FKPField,
    field_weights_args: FieldWeightsArgs,
    binner: BinMesh2SpectrumPoles,
    fkp_norms: list,
    include_gic: bool = True,
    los: Literal["local", "x", "y", "z"] = "local",
) -> tuple[jax.Array, jax.Array, jax.Array]:
    r"""
    Compute the analytic shot-noise template for the survey geometry only, or for the geometry and the global integral constraint.

    The simplified pipeline has, for each estimator weighting :math:`X` (one or two *legs*, :math:`A` and :math:`B`), data weights ``input data weights * estimator weights`` (:math:`u_i w_{X,i}`) and randoms weights ``-alpha_X * randoms weights * estimator weights``, with ``alpha_X`` responding to the data weights if ``include_gic`` and frozen otherwise. With relative noise :math:`\epsilon_i` on the data weights (:math:`E[\epsilon_i\epsilon_j]=\delta_{ij}`), the first-order field of leg :math:`X` is :math:`\delta_X = \sum_i \epsilon_i u_i w_{X,i}(e_i - b_X)`, where :math:`e_i` is the delta of data object :math:`i` and :math:`b_X` the randoms weights of the leg normalized to a unit sum (:math:`b_{X,j}`). Taking the expectation of :math:`Q(\delta_A, \delta_B)` and writing :math:`p_i = u_i^2 w_{A,i} w_{B,i}` (:math:`w_i^2` for a single weighting), :math:`a` the data positions weighted by :math:`p_i` and :math:`Q(x, y)` the symmetrized bilinear power spectrum:

    * geometry only (:math:`\delta_X = \sum_i\epsilon_i u_i w_{X,i} e_i`): the spectrum response is :math:`\frac{2\ell+1}{I_0}\langle\sum_i p_i\mathcal L_\ell\rangle` (self pairs of the data) and the shot noise response is :math:`\sum_i p_i/I_0`.
    * with the global integral constraint: the spectrum response is the geometry one :math:`- Q(a, b_B) - Q(a, b_A) + (\sum_i p_i) Q(b_A, b_B)`, and the shot noise response is :math:`(\sum_i p_i / I_0)(1 + \sum_j b_{A,j} b_{B,j})` (data and randoms are disjoint objects, so :math:`e_i` has no overlap with :math:`b_X`). For :math:`A=B` these are :math:`-2Q(a,b) + (\sum_i w_i^2)Q(b,b)` and :math:`(\sum_i w_i^2/I_0)(1+\sum_j b_j^2)`.

    Parameters
    ----------
    *fkp_fields : FKPField
        One FKP field per region, as given to :py:func:`prepare_field_weights`.
    field_weights_args : FieldWeightsArgs
        Fixed arguments, obtained with :py:func:`prepare_field_weights`; only the weights are used.
    binner : BinMesh2SpectrumPoles
        Binning operator for the power spectrum estimation, same for all regions.
    fkp_norms : list
        Pre-computed power spectrum norms for the FKP fields, one per region. See :py:func:`get_estimator_normalizations`.
    include_gic : bool, optional
        Whether to include the global integral constraint, by default ``True``.
    los : Literal["local", "x", "y", "z"], optional
        Line of sight, by default "local"

    Returns
    -------
    tuple[jax.Array, jax.Array, jax.Array]
        Spectrum response of shape ``(n_regions, n_ells, n_k)``, shot noise response of shape ``(n_regions,)``, and the template (their ratio). Regions are separate: combine them with :py:func:`combine_regions_weighted_by_normalization`.

    Raises
    ------
    NotImplementedError
        If ``binner`` has odd multipoles.

    Notes
    -----
    With two estimator weightings in ``field_weights_args``, the template is the one of their symmetrized cross spectrum. Only the effects-free pipeline is described: RIC, AMR and NAM of ``field_weights_args`` are not accounted for (use :py:func:`sample_shotnoise_template_linearized`). In the math, the responses are :math:`T_\ell(k)` and :math:`S_2`, and the four terms of the global integral constraint case are those of its control variate.
    """
    if any(ell % 2 for ell in binner.ells):
        raise NotImplementedError("Analytic template only for even multipoles.")
    args = field_weights_args
    norms = get_estimator_normalizations(fkp_norms)
    spectrum_of = partial(bilinear_power_spectrum, particles=get_region_particles(*fkp_fields), binner=binner, norms=norms, los=los)
    # per leg: data weights and randoms weights normalized to a unit sum, one array per region
    data_legs, normalized_randoms_legs = [], []
    for data_estimator_weights, randoms_estimator_weights, _ in _legs(args):
        data_legs.append(_split_regions(args.input_data_weights * data_estimator_weights, args.n_data_per_region))
        randoms_block_weights = _split_regions(args.input_randoms_weights * randoms_estimator_weights, args.n_randoms_per_region)
        normalized_randoms_legs.append([weights / jnp.sum(weights) for weights in randoms_block_weights])
    # product of the data weights of the two legs (squared data weights for a single leg)
    data_weights_products = [weights * other_weights for weights, other_weights in zip(data_legs[0], data_legs[-1], strict=True)]
    # sum of the weights products over the normalization, from the data blocks alone
    def data_only(data_blocks):
        return [jnp.concatenate([data, jnp.zeros_like(randoms)]) for data, randoms in zip(data_blocks, normalized_randoms_legs[0], strict=True)]

    shotnoise_response = conventional_shotnoise(data_only(data_legs[0]), data_only(data_legs[-1]), binner=binner, norms=norms)
    spectrum_response = jnp.stack(
        [
            _geometric_self_pair_term(binner, norms[i], fkp_field.data.positions, data_weights_products[i], shotnoise_response[i])
            for i, fkp_field in enumerate(fkp_fields)
        ]
    )
    if include_gic:
        products_field_weights = [jnp.concatenate([products, jnp.zeros_like(randoms)]) for products, randoms in zip(data_weights_products, normalized_randoms_legs[0], strict=True)]

        def randoms_field_weights(normalized_randoms):
            return [jnp.concatenate([jnp.zeros_like(products), randoms]) for products, randoms in zip(data_weights_products, normalized_randoms, strict=True)]

        randoms_field_weights_legs = [randoms_field_weights(normalized_randoms) for normalized_randoms in normalized_randoms_legs]
        sum_data_weights_products = jnp.stack([jnp.sum(products) for products in data_weights_products])
        if args.n_legs == 1:
            data_randoms_cross = 2.0 * spectrum_of(products_field_weights, randoms_field_weights_legs[0])
        else:
            data_randoms_cross = spectrum_of(products_field_weights, randoms_field_weights_legs[1]) + spectrum_of(products_field_weights, randoms_field_weights_legs[0])
        spectrum_response = (
            spectrum_response
            - data_randoms_cross
            + sum_data_weights_products[:, None, None] * spectrum_of(randoms_field_weights_legs[0], None if args.n_legs == 1 else randoms_field_weights_legs[1])
        )
        shotnoise_response = shotnoise_response * (
            1.0 + jnp.stack([jnp.sum(randoms * other_randoms) for randoms, other_randoms in zip(normalized_randoms_legs[0], normalized_randoms_legs[-1], strict=True)])
        )
    return spectrum_response, shotnoise_response, spectrum_response / shotnoise_response[:, None, None]


def _geometric_self_pair_term(
    binner: BinMesh2SpectrumPoles, norm: jax.Array, positions: jax.Array, squared_data_weights: jax.Array, monopole_self_pair_term: jax.Array
) -> jax.Array:
    r"""
    Geometric self-pair term of one region, shape ``(n_ells, n_k)``.

    This is :math:`\frac{2\ell+1}{I_0}\langle \sum_i w_i^2 \mathcal L_\ell(\hat k\cdot\hat x_i)\rangle_{\hat k\in b}`, using :math:`\mathcal L_\ell(\hat k\cdot\hat x) = \frac{4\pi}{2\ell+1}\sum_m Y_{\ell m}(\hat k) Y_{\ell m}(\hat x)`: the mode average only involves the binned harmonics of the *discrete mesh modes* (exact on the actual :math:`k` grid, including its anisotropy). Even multipoles only; ``monopole_self_pair_term`` is the monopole :math:`\sum_i w_i^2 / I_0`.
    """
    kvec = binner.mattrs.kcoords(sparse=True)
    n_k = len(binner.xavg)
    rows = []
    for ell in binner.ells:
        if ell == 0:
            rows.append(monopole_self_pair_term * jnp.ones(n_k))
            continue
        acc = jnp.zeros(n_k)
        for m in range(-ell, ell + 1):
            ylm = get_Ylm(ell, m, reduced=False, real=True)
            ybar = binner(jnp.broadcast_to(ylm(*kvec), tuple(binner.mattrs.meshsize)), remove_zero=True)
            acc = acc + ybar * jnp.sum(squared_data_weights * ylm(*positions.T))
        rows.append(4.0 * jnp.pi * acc / norm)
    return jnp.stack(rows)


def shotnoise_template_with_control_variate(
    result: dict,
    norms: np.ndarray,
    analytic_spectrum_response: np.ndarray,
    analytic_shotnoise_response: np.ndarray,
    coefficient: float | Literal["fit"] = 1.0,
    regions: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    r"""
    Compute the variance-reduced shot-noise template, using the analytic template with the global integral constraint as control variate.

    This requires ``compute_control_variate=True`` in the run, so that the responses of the simplified pipeline were computed on the **same** draws as the main ones. Per region, multipole and bin, the control variate (response of the simplified pipeline minus its analytic expectation) is subtracted from the spectrum response, with ``coefficient``; the same is done for the shot noise response. The template is the ratio of the sums of the adjusted responses, with a jackknife covariance (the adjusted realizations are jackknifed).

    Parameters
    ----------
    result : dict
        Samples, output of one of the ``sample_shotnoise_template_*`` functions.
    norms : np.ndarray
        Estimator normalization of each region.
    analytic_spectrum_response : np.ndarray
        Analytic expectation of the spectrum response of the simplified pipeline: first output of :py:func:`analytic_shotnoise_template` with ``include_gic=True`` and the same catalogs.
    analytic_shotnoise_response : np.ndarray
        Analytic expectation of the shot noise response of the simplified pipeline: second output of :py:func:`analytic_shotnoise_template`.
    coefficient : float | Literal["fit"], optional
        Coefficient of the control variate. ``"fit"`` uses the covariance of the response with the control variate over its variance, per region, multipole and bin (and its shot noise analogue per region). By default 1.0.
    regions : list[int] | None, optional
        Regions to combine, by default ``None`` (all).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, dict]
        Template of shape ``(n_ells, n_k)``, its covariance, and diagnostics: the coefficients ``spectrum_coefficient`` and ``shotnoise_coefficient``, and the variance reduction factor of the spectrum response ``variance_reduction`` (one minus its squared correlation with the control variate).

    Raises
    ------
    ValueError
        If ``result`` has no control variate, or ``coefficient`` is not a float or ``"fit"``.

    Notes
    -----
    In the math, the control variate is built from :math:`X^A_r, D^A_r` and the coefficients are :math:`\beta, \beta_S`.
    """
    if "control_variate" not in result["extras"]:
        raise ValueError("Run with compute_control_variate=True to use a control variate.")
    control_variate_spectrum, control_variate_shotnoise = (np.asarray(x) for x in result["extras"]["control_variate"])
    analytic_spectrum_response, analytic_shotnoise_response = np.asarray(analytic_spectrum_response), np.asarray(analytic_shotnoise_response)
    spectrum_response, shotnoise_response = np.asarray(result["power_spectrum_response"]), np.asarray(result["shotnoise_response"])

    def fit(response, control):
        response_deviation, control_deviation = response - response.mean(0), control - control.mean(0)
        return (response_deviation * control_deviation).sum(0) / np.maximum((control_deviation * control_deviation).sum(0), np.finfo(float).tiny)

    if isinstance(coefficient, str):
        if coefficient != "fit":
            raise ValueError("coefficient must be a float or 'fit'")
        spectrum_coefficient, shotnoise_coefficient = fit(spectrum_response, control_variate_spectrum), fit(shotnoise_response, control_variate_shotnoise)
    else:
        spectrum_coefficient, shotnoise_coefficient = coefficient, coefficient
    # per-realization adjusted quantities; their means are the control-variate estimators, and the jackknife is done on them
    adjusted_spectrum_response = spectrum_response - spectrum_coefficient * (control_variate_spectrum - analytic_spectrum_response[None])
    adjusted_shotnoise_response = shotnoise_response - shotnoise_coefficient * (control_variate_shotnoise - analytic_shotnoise_response[None])
    template, covariance = jackknife_ratio_of_means(
        *combine_regions_weighted_by_normalization(adjusted_spectrum_response, adjusted_shotnoise_response, norms, regions)
    )
    control_deviation, response_deviation = control_variate_spectrum - control_variate_spectrum.mean(0), spectrum_response - spectrum_response.mean(0)
    squared_correlation = ((control_deviation * response_deviation).sum(0) ** 2) / np.maximum(
        (control_deviation**2).sum(0) * (response_deviation**2).sum(0), np.finfo(float).tiny
    )
    return (
        template,
        covariance,
        {"spectrum_coefficient": spectrum_coefficient, "shotnoise_coefficient": shotnoise_coefficient, "variance_reduction": 1.0 - squared_correlation},
    )
