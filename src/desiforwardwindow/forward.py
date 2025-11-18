"""Forward modeling of observational effects."""

from collections.abc import Callable
from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
from jaxpower import (
    BinMesh2SpectrumPoles,
    ParticleField,
    RealMeshField,
    compute_mesh2_spectrum,
    compute_normalization,
    generate_anisotropic_gaussian_mesh,
)
from lsstypes import Mesh2SpectrumPoles, ObservableTree, tree_map

from .utils import bincount_2d


def _prepare_AIC(
    data: ParticleField,
    randoms: ParticleField,
    # AIC specific data
    template_values: jnp.ndarray,
    mask_is_data: jnp.ndarray,
    # AIC specific parameters
    tail: float = 0.5,
    bin_margin: float = 1e-7,
    n_bins: int = 10,
) -> dict[str, jnp.ndarray]:
    mask_is_randoms = jnp.invert(mask_is_data)
    templates_lower_tails = jnp.percentile(template_values[mask_is_randoms].T, tail / 2, axis=1)
    templates_upper_tails = jnp.percentile(template_values[mask_is_randoms].T, 100 - tail / 2, axis=1)

    mask_extremes = jnp.invert(
        jnp.any(
            (template_values < templates_lower_tails).T | (template_values > templates_upper_tails).T,
            axis=0,
        )
    )

    bin_edges = jnp.linspace(
        start=template_values[mask_extremes, :].min(axis=0) - bin_margin,
        stop=template_values[mask_extremes, :].max(axis=0) + bin_margin,
        num=n_bins + 1,
    )

    templates_normalized = (template_values - bin_edges[0, :]) / (bin_edges[-1, :] - bin_edges[0, :])  # For binning, have to mask

    mask_extremes_in_randoms = mask_extremes[mask_is_randoms]
    mask_extremes_in_data = mask_extremes[mask_is_data]

    templates_digitized = jnp.clip(
        jnp.floor(templates_normalized * n_bins).astype(int) - (template_values == bin_edges[-1, :]) + 1,
        min=0,
        max=n_bins + 1,
    )

    # Binned weights and jacobian are used in the solution
    randoms_weights_binned = bincount_2d(
        templates_digitized[mask_is_randoms & mask_extremes].T,
        weights=randoms.weights[mask_extremes_in_randoms],
        length=n_bins + 1,
    )[:, 1:, ...]

    jacobian = bincount_2d(
        templates_digitized[mask_is_randoms & mask_extremes].T,
        weights=(
            randoms.weights[mask_extremes_in_randoms]
            * jnp.stack(
                [
                    jnp.ones_like(randoms.weights[mask_extremes_in_randoms]),
                    *templates_normalized[mask_is_randoms & mask_extremes, :].T,
                ]
            )
        ).T,
        length=n_bins + 1,
    )[:, 1:, ...]

    normalization = (
        data[mask_extremes_in_data].sum() / randoms[mask_extremes_in_randoms].sum()
    )  # without extremes, without the actualized data weights: approximate

    # Ravel everything to take advantage of matrix operations
    jacobian = jacobian.reshape((-1, jacobian.shape[-1]))
    randoms_weights_binned = randoms_weights_binned.reshape((-1,))
    # Precompute a matrix that will be reused
    transpose_jw = normalization * jacobian.T * randoms_weights_binned  # matrix product with diag matrix is just numpy product with the diag vector
    factor = jnp.linalg.inv(transpose_jw.dot(jacobian)).dot(transpose_jw)
    constant = normalization * factor.dot(randoms_weights_binned)

    # pre-masked templates
    data_templates_digitized_no_extremes = templates_digitized[mask_is_data * mask_extremes]
    data_templates_normalized_with_extremes = templates_normalized[mask_is_data]

    # for refactor later: only need [data_weights, data_templates_digitized_no_extremes, mask_extremes_in_data, n_bins, data_templates_normalized_with_extremes, factor, constant]
    return {
        "data_templates_digitized_no_extremes": data_templates_digitized_no_extremes,
        "mask_extremes_in_data": mask_extremes_in_data,
        "n_bins": n_bins,
        "data_templates_normalized_with_extremes": data_templates_normalized_with_extremes,
        "factor": factor,
        "constant": constant,
    }


def _get_AIC_weights(
    data_weights: jnp.ndarray,
    data_templates_digitized_no_extremes: jnp.ndarray,
    mask_extremes_in_data: jnp.ndarray,
    n_bins: int,
    data_templates_normalized_with_extremes: jnp.ndarray,
    factor: jnp.ndarray,
    constant: jnp.ndarray,
):
    data_weights_binned = bincount_2d(
        data_templates_digitized_no_extremes.T,
        weights=data_weights[mask_extremes_in_data],
        length=n_bins + 1,
    )[:, 1:, ...]
    p_opt = factor.dot(data_weights_binned.reshape((-1,))) - constant
    return 1 / (1 + p_opt[0] + data_templates_normalized_with_extremes.dot(p_opt[1:]))


def get_AIC_foward_model(
    data: ParticleField,
    randoms: ParticleField,
    # AIC specific data
    template_values: jnp.ndarray,
    mask_is_data: jnp.ndarray,
    # AIC specific parameters
    tail: float = 0.5,
    bin_margin: float = 1e-7,
    n_bins: int = 10,
) -> Callable:
    """
    Build a jittable, differentiable ``get_AIC_weights`` function that takes in `data.weights` and returns AIC (photometric) weights.

    Parameters
    ----------
    data : ParticleField
        Data particles positions and weights.
    randoms : ParticleField
        Randoms particles positions and weights.
    template_values : jnp.ndarray
        Values of the templates for the data and the randoms.
    mask_is_data : jnp.ndarray
        Mask indicating where data is in ``template_values``.
    tail : float, optional
        Percentile of the (random's) template value distribution to remove, by default 0.5
    bin_margin : float, optional
        Margin on the side of edges, by default 1e-7
    n_bins : int, optional
        Number of bins for each template in the regression, by default 10

    Returns
    -------
    Callable
        Jitted, differentiable ``get_AIC_weights`` function that takes in ``data.weights`` and returns AIC (photometric) weights.
    """
    fixed_args = _prepare_AIC(
        data=data,
        randoms=randoms,
        # AIC specific data
        template_values=template_values,
        mask_is_data=mask_is_data,
        # AIC specific parameters
        tail=tail,
        bin_margin=bin_margin,
        n_bins=n_bins,
    )
    get_AIC_weights = jax.jit(partial(_get_AIC_weights, **fixed_args))  # Can fix everything but data_weights
    return get_AIC_weights


def _prepare_RIC(
    data: ParticleField,
    randoms: ParticleField,
    # RIC specific parameters
    n_bins_RIC: int,
):
    mattrs = randoms.attrs

    dmin = jnp.min(mattrs.boxcenter - mattrs.boxsize / 2.0)
    dmax = (1.0 + 1e-9) * jnp.sqrt(jnp.sum((mattrs.boxcenter + mattrs.boxsize / 2.0) ** 2))
    distance_edges = jnp.linspace(dmin, dmax, n_bins_RIC)
    randoms_distances = jnp.sqrt(jnp.power(randoms.positions, 2).sum(axis=-1))
    randoms_distances_digitized = jnp.digitize(randoms_distances, bins=distance_edges)  # could be made faster with jnp.floor
    randoms_distances_binned = jnp.bincount(randoms_distances_digitized, weights=randoms.weights, length=n_bins_RIC + 1)[1:]

    data_distances = jnp.sqrt(jnp.power(data.positions, 2).sum(axis=-1))
    data_distances_digitized = jnp.digitize(data_distances, bins=distance_edges)

    randoms_sum = randoms.sum()

    return {
        "data_distances_digitized": data_distances_digitized,
        "n_bins_RIC": n_bins_RIC,
        "randoms_distances_binned": randoms_distances_binned,
        "randoms_sum": randoms_sum,
    }


def _get_RIC_weights(
    data_weights,
    data_distances_digitized,
    n_bins_RIC,
    randoms_distances_binned,
    randoms_sum,
):
    data_distances_binned = jnp.bincount(data_distances_digitized, weights=data_weights, length=n_bins_RIC + 1)[1:]
    return (
        data_weights.sum()
        / randoms_sum
        * jnp.where(
            data_distances_binned == 0,
            1.0,
            (randoms_distances_binned / data_distances_binned),
        )[data_distances_digitized - 1]
    )


def get_RIC_forward_model(
    data: ParticleField,
    randoms: ParticleField,
    # RIC specific parameters
    n_bins_RIC: int,
) -> Callable:
    """
    Build a jittable, differentiable ``get_RIC_weights`` function that takes in `data.weights` and returns RIC (forward model) weights.

    Parameters
    ----------
    data : ParticleField
        _description_
    randoms : ParticleField
        _description_
    n_bins_RIC : int
        _description_

    Returns
    -------
    Callable
        Jitted, differentiable ``get_RIC_weights`` function that takes in ``data.weights`` and returns RIC weights.
    """
    fixed_args = _prepare_RIC(data=data, randoms=randoms, n_bins_RIC=n_bins_RIC)

    get_RIC_weights = jax.jit(partial(_get_RIC_weights, **fixed_args))  # Can fix everything but data_weights
    return get_RIC_weights


def mock_survey(
    # Gaussian mock generation
    theory: ObservableTree,
    seed: jnp.ndarray,
    los: Literal["local", "x", "y", "z"],
    unitary_amplitude: bool,
    # Data catalog and effects
    data: ParticleField,
    get_RIC_weights: Callable | None,
    get_AIC_weights: Callable | None,
    # Final P(k) estimation
    binner: BinMesh2SpectrumPoles,
    randoms_mesh: RealMeshField,
    randoms_shotnoise: float,
) -> Mesh2SpectrumPoles:
    """
    Apply observation forward modeling to a theoretical power spectrum.

    Parameters
    ----------
    theory : ObservableTree
        Theory power spectrum to "observe".
    seed : jnp.ndarray
        Jax random key for the mesh generation.
    los : Literal["local", "x", "y", "z"]
        Line of sight for the mock generation.
    unitary_amplitude : bool
        If ``True``, normalize the mock's amplitude to be unitary.
    data : ParticleField
        "Data" particles (randomly distributed positions), on which to paint the mock's fluctuation. This is where the geometry is contained.
    get_RIC_weights : Callable | None, Optional
        Optional function taking in ``data.weights``-like arguments and returning weights to apply the radial integral constraint. See :py:func:`get_RIC_forward_model`.
    get_AIC_weights : Callable | None
        Optional function taking in ``data.weights``-like arguments and returning weights to apply the angular integral constraint. See :py:func:`get_AIC_forward_model`.
    binner : BinMesh2SpectrumPoles
        Binning operator to compute the output power spectrum.
    randoms_mesh : RealMeshField
        Pre-painted randoms catalog.
    randoms_shotnoise : float
        Sum of the squared weights of the randoms.

    Returns
    -------
    Mesh2SpectrumPoles
        Realization of an observation of the theory power spectrum.
    """
    # Generate a gaussian mesh mock with exact required theory P(k)
    mattrs = data.attrs
    mesh = generate_anisotropic_gaussian_mesh(
        mattrs,
        poles=theory,
        seed=seed,
        los=los,
        unitary_amplitude=unitary_amplitude,
    )
    # Paint it on the portion of randoms designated as "data" -> data catalog w/ geometry
    data_field = data.clone(weights=data.weights * (mesh.read(data.positions, resampler="cic", compensate=True) + 1))
    # Apply RIC if necessary
    if get_RIC_weights is not None:
        RIC_weights = get_RIC_weights(data_field.weights)
        data_field = data_field.clone(weights=data_field.weights * RIC_weights)
    # Apply AIC if necessary
    if get_AIC_weights is not None:
        AIC_weights = get_AIC_weights(data_field.weights)
        data_field = data_field.clone(weights=data_field.weights * AIC_weights)
    # Paint to mesh for P(k) computation and build FKP mesh
    data_mesh = data_field.paint(resampler="tsc", interlacing=3, compensate=True, out="real")
    alpha = data_mesh.sum() / randoms_mesh.sum()
    fkp_mesh = data_mesh - alpha * randoms_mesh
    norm = alpha * compute_normalization(data_mesh, randoms_mesh)
    shotnoise = jnp.sum(data.weights**2) + alpha**2 * randoms_shotnoise
    pk = compute_mesh2_spectrum(fkp_mesh, bin=binner, los={"local": "firstpoint"}.get(los, los))
    return pk.clone(
        norm=[norm] * len(binner.ells),
        num_shotnoise=[shotnoise * (ell == 0) * jnp.ones_like(binner.edges[..., 0]) for ell in binner.ells],
    )


def mock_survey_diff(
    # Gaussian mock generation
    theory: ObservableTree,
    seed: jnp.ndarray,
    los: Literal["local", "x", "y", "z"],
    unitary_amplitude: bool,
    # Data catalog and effects
    data: ParticleField,
    get_RIC_weights: Callable | None,
    get_AIC_weights: Callable | None,
    # Final P(k) estimation
    binner: BinMesh2SpectrumPoles,
    randoms_mesh: RealMeshField,
    randoms_shotnoise: float,
) -> Mesh2SpectrumPoles:
    """
    Apply observation forward modeling to a theoretical power spectrum and return the difference of the power spectrum with a geometry-only observation (no integral constraints).

    This is intended for use of the RIC and AIC as a control variate.

    Parameters
    ----------
    theory : ObservableTree
        Theory power spectrum to "observe".
    seed : jnp.ndarray
        Jax random key for the mesh generation.
    los : Literal["local", "x", "y", "z"]
        Line of sight for the mock generation.
    unitary_amplitude : bool
        If ``True``, normalize the mock's amplitude to be unitary.
    data : ParticleField
        "Data" particles (randomly distributed positions), on which to paint the mock's fluctuation. This is where the geometry is contained.
    get_RIC_weights : Callable | None, Optional
        Optional function taking in ``data.weights``-like arguments and returning weights to apply the radial integral constraint. See :py:func:`get_RIC_forward_model`.
    get_AIC_weights : Callable | None
        Optional function taking in ``data.weights``-like arguments and returning weights to apply the angular integral constraint. See :py:func:`get_AIC_forward_model`.
    binner : BinMesh2SpectrumPoles
        Binning operator to compute the output power spectrum.
    randoms_mesh : RealMeshField
        Pre-painted randoms catalog.
    randoms_shotnoise : float
        Sum of the squared weights of the randoms.

    Returns
    -------
    Mesh2SpectrumPoles
        Difference of two observations (with and without RIC/AIC) of a realization of the theory power spectrum.
    """
    if (get_RIC_weights is None) and (get_AIC_weights is None):
        raise ValueError(
            "When calculating a difference, should add at least one other observational effect than the geometry. Did you mean to call `mock_survey` ?"
        )
    # Generate a gaussian mesh mock with exact required theory P(k)
    mattrs = data.attrs
    mesh = generate_anisotropic_gaussian_mesh(
        mattrs,
        poles=theory,
        seed=seed,
        los=los,
        unitary_amplitude=unitary_amplitude,
    )
    # For the geometry-only observation, no need to go to catalog: compute directly on the mesh
    mesh_geo = mesh * randoms_mesh
    norm = compute_normalization(randoms_mesh, randoms_mesh, bin=binner)
    pk_geo = compute_mesh2_spectrum(mesh_geo, bin=binner, los=los).clone(norm=norm)
    # Paint data mesh on the portion of randoms designated as "data" -> data catalog w/ geometry
    data_field = data.clone(weights=data.weights * (mesh.read(data.positions, resampler="cic", compensate=True) + 1))
    # Apply RIC if necessary
    if get_RIC_weights is not None:
        RIC_weights = get_RIC_weights(data_field.weights)
        data_field = data_field.clone(weights=data_field.weights * RIC_weights)
    # Apply AIC if necessary
    if get_AIC_weights is not None:
        AIC_weights = get_AIC_weights(data_field.weights)
        data_field = data_field.clone(weights=data_field.weights * AIC_weights)
    # Paint to mesh for P(k) computation and build FKP mesh
    data_mesh = data_field.paint(resampler="tsc", interlacing=3, compensate=True, out="real")
    alpha = data_mesh.sum() / randoms_mesh.sum()
    fkp_mesh = data_mesh - alpha * randoms_mesh
    norm = alpha * compute_normalization(data_mesh, randoms_mesh)
    shotnoise = jnp.sum(data.weights**2) + alpha**2 * randoms_shotnoise
    pk_IC = compute_mesh2_spectrum(fkp_mesh, bin=binner, los={"local": "firstpoint"}.get(los, los))
    pk_IC = pk_IC.clone(
        norm=[norm] * len(binner.ells),
        num_shotnoise=[shotnoise * (ell == 0) * jnp.ones_like(binner.edges[..., 0]) for ell in binner.ells],
    )
    # Return the difference of the spectra
    spectra = [pk_IC, pk_geo]
    return tree_map(lambda poles: poles[0].clone(value=poles[0].value() - poles[1].value()), spectra)
