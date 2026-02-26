"""Forward modeling of observational effects."""

import itertools
from typing import Literal
from warnings import warn

import jax
import jax.numpy as jnp
from jaxpower import (
    BinMesh2SpectrumPoles,
    FKPField,
    MeshAttrs,
    ParticleField,
    RealMeshField,
    compute_fkp2_shotnoise,
    compute_mesh2_spectrum,
    generate_anisotropic_gaussian_mesh,
)
from jaxpower.mesh import get_sharding_mesh, make_array_from_process_local_data
from lsstypes import Mesh2SpectrumPoles, ObservableTree

from .utils import bincount, bincount_sorted, local_argsort, local_concatenate, local_split, make_jax_dataclass, select_region

RIC_args = make_jax_dataclass(
    class_name="RIC_args",
    dynamic_fields=[
        "data_distances_digitized",
        "randoms_distances_digitized",
        "data_regions",
        "randoms_regions",
        "data_to_remove",
    ],
    aux_fields=["n_bins", "apply_to"],
    types_fields={
        "data_distances_digitized": jax.Array,
        "randoms_distances_digitized": jax.Array,
        "data_regions": jax.Array,
        "randoms_regions": jax.Array,
        "data_to_remove": jax.Array,
        "n_bins": int,
        "apply_to": Literal["data", "randoms"],
    },
)


def prepare_RIC(
    data: list[ParticleField],
    randoms: list[ParticleField],
    regions: list[str],
    # RIC specific parameters
    n_bins: int,
    apply_to: Literal["data", "randoms"],
) -> RIC_args:
    """
    Prepare arguments necessary to applying RIC in :py:func:`mock_survey_catalog`.

    Parameters
    ----------
    data : list[ParticleField]
        Fields containing positions and weights of the data particles.
    randoms : list[ParticleField]
        Fields containing positions and weights of the randoms particles.
    regions : list[str]
        Regions to split data in.
    n_bins : int
        Number of distance bins to use.
    apply_to : Literal["data", "randoms"]
        Whether to produce weights to apply to randoms or data.

    Returns
    -------
    RIC_args
        Custom pytree class that contains all necessary information to applying RIC in :py:func:`mock_survey_catalog`.
    """
    # locally concatenate all data and randoms
    sharding_mesh = get_sharding_mesh()
    data_positions = local_concatenate([d.positions for d in data], axis=0, sharding_mesh=sharding_mesh)
    randoms_positions = local_concatenate([r.positions for r in randoms], axis=0, sharding_mesh=sharding_mesh)

    # Get minimal and maximal distance for binning
    dmin, dmax = jnp.inf, 0.0
    for ddata in data:
        corners = jnp.stack([ddata.attrs.boxcenter - ddata.attrs.boxsize / 2.0, ddata.attrs.boxcenter + ddata.attrs.boxsize / 2.0])
        cube_mins = jnp.min(corners, axis=0)
        cube_maxs = jnp.max(corners, axis=0)
        if (cube_mins < 0).all() and (cube_maxs > 0).all():
            dmin = 0.0
        else:
            projection = jnp.clip(jnp.zeros(3), cube_mins, cube_maxs)
            dmin = min(dmin, jnp.linalg.norm(projection))
        dmax = max(dmax, (1.0 + 1e-9) * jnp.linalg.norm(jnp.max(jnp.abs(corners), axis=0)))

    distance_edges = jnp.linspace(dmin, dmax, n_bins)

    data_distances = jnp.sqrt(jnp.power(data_positions, 2).sum(axis=-1))
    randoms_distances = jnp.sqrt(jnp.power(randoms_positions, 2).sum(axis=-1))
    data_distances_digitized = jnp.digitize(data_distances, bins=distance_edges)  # could be made faster with jnp.floor
    randoms_distances_digitized = jnp.digitize(randoms_distances, bins=distance_edges)

    # region selection
    data_ra = (jnp.arctan2(data_positions[..., 1], data_positions[..., 0]) % (2 * jnp.pi)) * 180 / jnp.pi
    randoms_ra = (jnp.arctan2(randoms_positions[..., 1], randoms_positions[..., 0]) % (2 * jnp.pi)) * 180 / jnp.pi
    data_dec = jnp.arcsin(data_positions[..., 2] / data_distances) * 180 / jnp.pi
    randoms_dec = jnp.arcsin(randoms_positions[..., 2] / randoms_distances) * 180 / jnp.pi

    data_regions = []
    randoms_regions = []

    for region in regions:
        data_mask = select_region(ra=data_ra, dec=data_dec, region=region, sharding_mesh=sharding_mesh)
        randoms_mask = select_region(ra=randoms_ra, dec=randoms_dec, region=region, sharding_mesh=sharding_mesh)
        if not data_mask.any():
            raise ValueError("No data in region %s. Cannot proceed.", region)
        if not randoms_mask.any():
            raise ValueError("No randoms in region %s. Cannot proceed.", region)
        data_regions.append(data_mask)
        randoms_regions.append(randoms_mask)

    data_distances_counts = jnp.bincount(data_distances_digitized, weights=None, length=n_bins + 1)[1:]
    randoms_distances_counts = jnp.bincount(randoms_distances_digitized, weights=None, length=n_bins + 1)[1:]
    data_to_remove = ((data_distances_counts != 0) * (randoms_distances_counts == 0))[data_distances_digitized]

    data_regions = jnp.stack(data_regions)
    randoms_regions = jnp.stack(randoms_regions)

    data_coverage = data_regions.sum(axis=0)
    randoms_coverage = randoms_regions.sum(axis=0)

    if (data_coverage >= 2).any():
        warn(f"Some ({(data_coverage >= 2).sum()}/{data_coverage.size}) data particles are in several regions at once.", RuntimeWarning, stacklevel=2)
    if (randoms_coverage >= 2).any():
        warn(f"Some ({(randoms_coverage >= 2).sum()}/{randoms_coverage.size}) randoms particles are in several regions at once.", RuntimeWarning, stacklevel=2)
    if (data_coverage < 1).any():
        warn(f"Some ({(data_coverage == 0).sum()}/{data_coverage.size}) data particles are in no region at all.", RuntimeWarning, stacklevel=2)
    if (randoms_coverage < 1).any():
        warn(f"Some ({(randoms_coverage == 0).sum()}/{randoms_coverage.size}) randoms particles are in no region at all.", RuntimeWarning, stacklevel=2)

    return RIC_args(
        data_distances_digitized=data_distances_digitized,
        randoms_distances_digitized=randoms_distances_digitized,
        data_regions=data_regions,
        randoms_regions=randoms_regions,
        data_to_remove=data_to_remove,
        n_bins=n_bins,
        apply_to=apply_to,
    )


@jax.jit(static_argnames=["n_bins", "apply_to"])
def apply_RIC(
    data_weights: jax.Array,
    randoms_weights: jax.Array,
    data_regions: jax.Array,
    randoms_regions: jax.Array,
    data_distances_digitized: jax.Array,
    randoms_distances_digitized: jax.Array,
    n_bins: int,
    apply_to: Literal["data", "randoms"] = "data",
) -> jax.Array:
    """
    Compute weights to apply the radial integral constraint at the catalog level.

    Parameters
    ----------
    data_weights : jax.Array
        Input data weights, shape (n_d,).
    randoms_weights : jax.Array
        Input randoms weights, shape (n_r,).
    data_regions : jax.Array
        Input masks for each region for the data, shape (r, n_d,).
    randoms_regions : jax.Array
        Input masks for each region for the data, shape (r, n_r,).
    data_distances_digitized : jax.Array
        Digitized radial distances of the data, shape (n_d,).
    randoms_distances_digitized : jax.Array
        Digitized radial distances of the randoms, shape (n_r,).
    n_bins : int
        Number of bins used in the digitization.
    apply_to : Literal["data", "randoms"], optional
        Whether to compute weights for the data or the randoms, by default "data"

    Returns
    -------
    jax.Array
        Weights to enforce RIC, to apply multiplicatively to the original data or randoms weights, shape (n_d,) or (n_r,).

    Raises
    ------
    ValueError
        If ``apply_to`` is an unsupported value.

    Notes
    -----
    * ``n_bins`` cannot be inferred dynamically, otherwise the function would not be compatible with ``jax.jit``.
    * Region masks must be perfectly complementary with complete coverage of the particles.
        * For uncovered particles, the returned weight will be 1.
        * For doubly covered particles, the returned weight will be the sum of the weights in each region
    """
    if apply_to == "data":
        pass
    elif apply_to == "randoms":
        data_weights, randoms_weights = randoms_weights, data_weights
        data_regions, randoms_regions = randoms_regions, data_regions
        data_distances_digitized, randoms_distances_digitized = randoms_distances_digitized, data_distances_digitized
    else:
        raise ValueError('Can only apply to randoms or data, not "%s"!', apply_to)

    # alphas of shape (r,): sum over last axis = n
    alphas = (data_weights * data_regions).sum(axis=-1) / (randoms_weights * randoms_regions).sum(axis=-1)
    data_weights_binned = bincount(data_distances_digitized, weights=data_weights * data_regions, length=n_bins + 1)[..., 1:]
    randoms_weights_binned = bincount(randoms_distances_digitized, weights=randoms_weights * randoms_regions, length=n_bins + 1)[..., 1:]
    ric_weights_binned = jnp.where(
        data_weights_binned == 0,
        0.0,  # don't care, will never be applied
        (alphas[..., None] * randoms_weights_binned / data_weights_binned),
    )  # NOTE: this is not reverse-differentiation compatible
    ric_weights = jnp.where(data_regions, ric_weights_binned[..., data_distances_digitized - 1], 0.0)
    return ric_weights.sum(axis=range(ric_weights.ndim - 1)) + jnp.invert(data_regions.any(axis=0))


AMR_args = make_jax_dataclass(
    class_name="AMR_args",
    dynamic_fields=[
        "data_regions",
        "randoms_regions",
        "data_extremes",
        "randoms_extremes",
        "data_templates_digitized",
        "randoms_templates_digitized",
        "data_templates_normalized",
        "randoms_templates_normalized",
        "data_isort",
        "randoms_isort",
    ],
    aux_fields=["n_bins", "apply_to"],
    types_fields={
        "data_regions": jax.Array,
        "randoms_regions": jax.Array,
        "data_extremes": jax.Array,
        "randoms_extremes": jax.Array,
        "data_templates_digitized": jax.Array,
        "randoms_templates_digitized": jax.Array,
        "data_templates_normalized": jax.Array,
        "randoms_templates_normalized": jax.Array,
        "data_isort": jax.Array,
        "randoms_isort": jax.Array,
        "n_bins": int,
        "apply_to": Literal["data", "randoms"],
    },
)


def prepare_AMR(
    data: list[ParticleField],
    randoms: list[ParticleField],
    regions_zranges: list[tuple[str, tuple[float, float]]],
    apply_to: Literal["data", "randoms"],
    # AMR specific parameters
    tail: float = 0.5,
    bin_margin: float = 1e-7,
    n_bins: int = 10,
) -> AMR_args:
    """
    Precompute all arguments necessary to get Angular Mode Removal in :py:func:`mock_survey_catalog`.

    Parameters
    ----------
    data : list[ParticleField]
        Fields containing positions and weights of the data particles. Must have extra fields ``"Z"`` and ``"template_values"`` available.
    randoms : list[ParticleField]
        Fields containing positions and weights of the randoms particles. Must have extra fields ``"Z"`` and ``"template_values"`` available.
    tail : float, optional
        Percentile of the (random's) template value distribution to remove, by default 0.5
    bin_margin : float, optional
        Margin on the side of edges, by default 1e-7
    n_bins : int, optional
        Number of bins for each template in the regression, by default 10

    Returns
    -------
    AMR_args
        Dataclass containing all information needed by :py:func:`mock_survey_catalog` to apply AMR.

    Notes
    -----
    The digitized templates are offset by (n_bins+2)*n_regions, so that the region can be inferred directly from the digitized value. Since we expect digitized values to span [0, ``n_bins``+1], the offset is ``n_bins``+2.
    """
    sharding_mesh = get_sharding_mesh()

    # Locally concatenate, preserving the sharding
    data_positions = local_concatenate([d.positions for d in data], axis=0, sharding_mesh=sharding_mesh)
    randoms_positions = local_concatenate([r.positions for r in randoms], axis=0, sharding_mesh=sharding_mesh)
    data_redshifts = local_concatenate([d.extra["Z"] for d in data], axis=0, sharding_mesh=sharding_mesh)
    randoms_redshifts = local_concatenate([r.extra["Z"] for r in randoms], axis=0, sharding_mesh=sharding_mesh)
    template_values_data = local_concatenate([d.extra["template_values"] for d in data], axis=0, sharding_mesh=sharding_mesh)
    template_values_randoms = local_concatenate([r.extra["template_values"] for r in randoms], axis=0, sharding_mesh=sharding_mesh)

    # Compute the 0.5th and 99.5th percentiles of the templates in the randoms
    # Need to work around fake particles
    is_real = local_concatenate([(_randoms.weights != 0) for _randoms in randoms], axis=0, sharding_mesh=sharding_mesh)
    templates_lower_tails = jnp.percentile(template_values_randoms[is_real], tail / 2, axis=0, method="higher")
    templates_upper_tails = jnp.percentile(template_values_randoms[is_real], 100 - tail / 2, axis=0, method="lower")
    del is_real

    # Now proceed as usual

    # Select the regions
    data_distances = jnp.sqrt(jnp.power(data_positions, 2).sum(axis=-1))
    randoms_distances = jnp.sqrt(jnp.power(randoms_positions, 2).sum(axis=-1))
    data_ra = (jnp.arctan2(data_positions[..., 1], data_positions[..., 0]) % (2 * jnp.pi)) * 180 / jnp.pi
    randoms_ra = (jnp.arctan2(randoms_positions[..., 1], randoms_positions[..., 0]) % (2 * jnp.pi)) * 180 / jnp.pi
    data_dec = jnp.arcsin(data_positions[..., 2] / data_distances) * 180 / jnp.pi
    randoms_dec = jnp.arcsin(randoms_positions[..., 2] / randoms_distances) * 180 / jnp.pi

    mask_extremes_d = jnp.all((template_values_data >= templates_lower_tails) & (template_values_data <= templates_upper_tails), axis=1)
    mask_extremes_r = jnp.all((template_values_randoms >= templates_lower_tails) & (template_values_randoms <= templates_upper_tails), axis=1)

    bin_edges = jnp.linspace(
        start=templates_lower_tails - bin_margin,
        stop=templates_upper_tails + bin_margin,
        num=n_bins + 1,
    )

    templates_normalized_r = (template_values_randoms - bin_edges[0, :]) / (bin_edges[-1, :] - bin_edges[0, :])
    templates_normalized_d = (template_values_data - bin_edges[0, :]) / (bin_edges[-1, :] - bin_edges[0, :])

    templates_digitized_r = jnp.clip(
        jnp.floor(templates_normalized_r * n_bins).astype(int) - (templates_normalized_r == bin_edges[-1, :]) + 1,
        min=0,
        max=n_bins + 1,
    )

    templates_digitized_d = jnp.clip(
        jnp.floor(templates_normalized_d * n_bins).astype(int) - (templates_normalized_d == bin_edges[-1, :]) + 1,
        min=0,
        max=n_bins + 1,
    )

    data_regions = []
    randoms_regions = []

    for region, (zmin, zmax) in regions_zranges:
        data_mask = select_region(ra=data_ra, dec=data_dec, region=region, sharding_mesh=sharding_mesh)
        data_mask &= (zmin <= data_redshifts) & (data_redshifts <= zmax)

        randoms_mask = select_region(ra=randoms_ra, dec=randoms_dec, region=region, sharding_mesh=sharding_mesh)
        randoms_mask &= (zmin <= randoms_redshifts) & (randoms_redshifts <= zmax)

        data_regions.append(data_mask)
        randoms_regions.append(randoms_mask)

        if not data_mask.any():
            raise ValueError("No data in region %s, redshift range %.1f - %.1f. Cannot proceed.", region, zmin, zmax)
        if not randoms_mask.any():
            raise ValueError("No randoms in region %s, redshift range %.1f - %.1f. Cannot proceed.", region, zmin, zmax)

    data_regions = jnp.stack(data_regions)
    randoms_regions = jnp.stack(randoms_regions)

    # pre-computed templates
    data_templates_digitized = jnp.vstack(
        [jnp.full(shape=data_positions.shape[0], dtype=templates_digitized_d.dtype, fill_value=n_bins - 1), templates_digitized_d.T]
    )
    randoms_templates_digitized = jnp.vstack(
        [jnp.full(shape=randoms_positions.shape[0], dtype=templates_digitized_r.dtype, fill_value=n_bins - 1), templates_digitized_r.T]
    )

    # Offset the digitized templates depending on the region
    data_templates_digitized = data_templates_digitized + (n_bins + 2) * (jnp.arange(data_regions.shape[0])[:, None] * data_regions).sum(axis=0)
    randoms_templates_digitized = randoms_templates_digitized + (n_bins + 2) * (jnp.arange(randoms_regions.shape[0])[:, None] * randoms_regions).sum(axis=0)

    if apply_to == "data":
        data_templates_normalized = jnp.vstack([jnp.ones_like(data_positions[:, 0]), templates_normalized_d.T])
    else:
        data_templates_normalized = None
    randoms_templates_normalized = jnp.vstack([jnp.ones_like(randoms_positions[:, 0]), templates_normalized_r.T])

    data_coverage = data_regions.sum(axis=0)
    randoms_coverage = randoms_regions.sum(axis=0)

    if (data_coverage >= 2).any():
        warn(f"Some ({(data_coverage >= 2).sum()}/{data_coverage.size}) data particles are in several regions at once.", RuntimeWarning, stacklevel=2)
    if (randoms_coverage >= 2).any():
        warn(f"Some ({(randoms_coverage >= 2).sum()}/{randoms_coverage.size}) randoms particles are in several regions at once.", RuntimeWarning, stacklevel=2)
    if (data_coverage < 1).any():
        warn(f"Some ({(data_coverage == 0).sum()}/{data_coverage.size}) data particles are in no region at all.", RuntimeWarning, stacklevel=2)
    if (randoms_coverage < 1).any():
        warn(f"Some ({(randoms_coverage == 0).sum()}/{randoms_coverage.size}) randoms particles are in no region at all.", RuntimeWarning, stacklevel=2)

    data_isort = local_argsort(data_templates_digitized, axis=1, sharding_mesh=sharding_mesh)
    randoms_isort = local_argsort(randoms_templates_digitized, axis=1, sharding_mesh=sharding_mesh)

    return AMR_args(
        data_regions=data_regions,
        randoms_regions=randoms_regions,
        data_extremes=mask_extremes_d,
        randoms_extremes=mask_extremes_r,
        data_templates_digitized=data_templates_digitized,
        randoms_templates_digitized=randoms_templates_digitized,
        data_templates_normalized=data_templates_normalized,
        randoms_templates_normalized=randoms_templates_normalized,
        data_isort=data_isort,
        randoms_isort=randoms_isort,
        n_bins=n_bins,
        apply_to=apply_to,
    )


bincount_sorted_vmapped = jax.vmap(bincount_sorted, in_axes=(0, None, 0, None, None))


@jax.jit(static_argnames=["n_bins", "apply_to"])
def apply_AMR(
    data_weights: jax.Array,
    randoms_weights: jax.Array,
    data_regions: jax.Array,
    randoms_regions: jax.Array,
    data_extremes: jax.Array,
    randoms_extremes: jax.Array,
    data_templates_digitized: jax.Array,
    randoms_templates_digitized: jax.Array,
    data_templates_normalized: jax.Array | None,
    randoms_templates_normalized: jax.Array,
    data_isort: jax.Array,
    randoms_isort: jax.Array,
    n_bins: int = 10,
    apply_to: Literal["data", "randoms"] = "data",
) -> jax.Array:
    """
    Return imaging systematics correction weights from linearized _à la eBOSS_ style weights.

    Parameters
    ----------
    data_weights : jax.Array
        Input data weights, shape (n_d,).
    randoms_weights : jax.Array
        Input randoms weights, shape (n_r,).
    data_regions : jax.Array
        Input masks for each region for the data, shape (r, n_d,).
    randoms_regions : jax.Array
        Input masks for each region for the data, shape (r, n_r,).
    data_extremes : jax.Array
        Mask indicating extreme template values to discard in the data.
    randoms_extremes : jax.Array
        Mask indicating extreme template values to discard in the randoms.
    data_templates_digitized : jax.Array
        Digitized values of the templates for the data, shape (n_sys + 1, n_d). First line should be all ``n_bins - 1`` for the constant term.
    randoms_templates_digitized : jax.Array
        Digitized values of the templates for the randoms, shape (n_sys + 1, n_r). First line should be all ``n_bins - 1`` for the constant term.
    data_templates_normalized : jax.Array | None
        Normalized values of the templates for the data, shape (n_sys + 1, n_d). First line should be all ones for the constant term. This only needs to be provided if ``apply_to`` is set to ``"data"``.
    randoms_templates_normalized : jax.Array
        Normalized values of the templates for the randoms, shape (n_sys + 1, n_d). First line should be all ones for the constant term.
    n_bins : int, optional
        Number of bins used for the templates, by default 10.
    apply_to : Literal["data", "randoms"], optional
        Whether to get weights to apply to the data or the randoms, by default "data".

    Returns
    -------
    jax.Array
        Weights to multiplicatively apply to either the data or the randoms.

    Raises
    ------
    ValueError
        If ``to_apply`` is unrecognized.

    Notes
    -----
     * The regions mask should not contain the extremes, ohterwise the output weights for the extremes will be wrong.
     * Regions should not overlap, otherwise weights for multi-region particles will be wrong.
    """
    data_regions = jnp.atleast_2d(data_regions)
    randoms_regions = jnp.atleast_2d(randoms_regions)

    data_weights = data_weights * data_extremes
    randoms_weights = randoms_weights * randoms_extremes
    # shapes: (regions, N_sys + 1, N_bins)
    data_binned_exploded = bincount_sorted_vmapped(
        data_templates_digitized, data_weights, data_isort, (n_bins + 2) * data_regions.shape[0], get_sharding_mesh()
    )
    data_binned = jnp.stack(jnp.split(data_binned_exploded, data_regions.shape[0], axis=-1))[..., 1:-1]

    randoms_binned_exploded = bincount_sorted_vmapped(
        randoms_templates_digitized, randoms_weights, randoms_isort, (n_bins + 2) * randoms_regions.shape[0], get_sharding_mesh()
    )
    randoms_binned = jnp.stack(jnp.split(randoms_binned_exploded, randoms_regions.shape[0], axis=-1))[..., 1:-1]

    # shape: (regions, N_sys + 1, N_bins,  N_sys + 1)
    # The last dimension is for the matrix product with the coefficients vector
    # The rightmost (N_sys + 1, N_bins) correspond, in spirit, to one big axis
    randoms_templates_binned_exploded = bincount_sorted_vmapped(
        randoms_templates_digitized, randoms_weights * randoms_templates_normalized, randoms_isort, (n_bins + 2) * randoms_regions.shape[0], get_sharding_mesh()
    )
    randoms_templates_binned = jnp.stack(jnp.split(randoms_templates_binned_exploded, randoms_regions.shape[0], axis=1))[..., 1:-1, :]

    normalization = (randoms_regions * randoms_weights).sum(axis=1) / (data_regions * data_weights).sum(axis=1)
    prefactor = jnp.where(randoms_binned != 0, jnp.sqrt(normalization[:, None, None] / randoms_binned), 0.0)

    X = prefactor[..., None] * randoms_templates_binned
    y = prefactor * (data_binned - randoms_binned / normalization[..., None, None])

    Xty = jnp.einsum("rijp, rij -> rp ", X, y)
    XtX = jnp.einsum("rijq, rijp -> rpq", X, X)

    # Batch over first dimension (the regions): shape (regions, N_sys + 1)
    coefficients = jnp.linalg.solve(XtX, Xty[..., None]).squeeze(-1)

    if apply_to == "data":
        return (data_regions / (1 + jnp.einsum("in, ri->rn", data_templates_normalized, coefficients))).sum(axis=0) + jnp.invert(
            data_regions.any(axis=0)
        )  # these are data weights
    elif apply_to == "randoms":
        # TODO: check that I'm not saying n'importe nawak for randoms
        return (randoms_regions * (1 + jnp.einsum("in, ri->rn", randoms_templates_normalized, coefficients))).sum(axis=0) + jnp.invert(
            randoms_regions.any(axis=0)
        )  # these are randoms weights
    else:
        raise ValueError('Can only apply to randoms or data, not "%s"!', apply_to)


NAM_args = make_jax_dataclass(
    class_name="NAM_args",
    dynamic_fields=[
        "data_pixels",
        "randoms_pixels",
        "data_regions",
        "randoms_regions",
        "data_to_remove",
    ],
    aux_fields=["nside", "apply_to"],
    types_fields={
        "data_pixels": jax.Array,
        "randoms_pixels": jax.Array,
        "data_regions": jax.Array,
        "randoms_regions": jax.Array,
        "data_to_remove": jax.Array,
        "nside": int,
        "apply_to": Literal["data", "randoms"],
    },
)


def prepare_NAM(
    data: list[ParticleField],
    randoms: list[ParticleField],
    regions_zranges: list[tuple[str, tuple[float, float]]],
    # NAM specific parameters
    nside: int,
    apply_to: Literal["data", "randoms"],
) -> NAM_args:
    """
    Prepare arguments necessary to applying NAM/AIC in :py:func:`mock_survey_catalog`.

    Parameters
    ----------
    data : list[ParticleField]
        Fields containing positions and weights of the data particles. Must have extra field ``"Z"`` (redshift) available.
    randoms : list[ParticleField]
        Fields containing positions and weights of the randoms particles. Must have extra field ``"Z"`` (redshift) available.
    regions_zranges: list[tuple[str, tuple[float, float]]]
        Regions and redshift ranges to split data in.
    nside : int
        NSIDE to use for healpixelization.
    apply_to : Literal["data", "randoms"]
        Whether to produce weights to apply to randoms or data.

    Returns
    -------
    RIC_args
        Custom pytree class that contains all necessary information to applying NAM/AIC in :py:func:`mock_survey_catalog`.

    Notes
    -----
    Healpix manipulation is always done in ``RING`` scheme.
    """
    # Locally concatenate, preserving the sharding
    sharding_mesh = get_sharding_mesh()
    data_positions = local_concatenate([d.positions for d in data], axis=0, sharding_mesh=sharding_mesh)
    randoms_positions = local_concatenate([r.positions for r in randoms], axis=0, sharding_mesh=sharding_mesh)
    data_redshifts = local_concatenate([d.extra["Z"] for d in data], axis=0, sharding_mesh=sharding_mesh)
    randoms_redshifts = local_concatenate([r.extra["Z"] for r in randoms], axis=0, sharding_mesh=sharding_mesh)

    def _vec2pix(positions):
        import healpy as hp

        return hp.vec2pix(nside, *positions.T)

    def vec2pix(positions):
        return jax.pure_callback(_vec2pix, jax.ShapeDtypeStruct(positions.shape[:1], jnp.int64), positions)

    from jax.sharding import PartitionSpec as P

    sharding_mesh = get_sharding_mesh()

    if sharding_mesh.axis_names:
        vec2pix = jax.shard_map(vec2pix, mesh=sharding_mesh, in_specs=P(sharding_mesh.axis_names), out_specs=P(sharding_mesh.axis_names))

    # Select the regions
    data_distances = jnp.sqrt(jnp.power(data_positions, 2).sum(axis=-1))
    randoms_distances = jnp.sqrt(jnp.power(randoms_positions, 2).sum(axis=-1))
    data_ra = (jnp.arctan2(data_positions[..., 1], data_positions[..., 0]) % (2 * jnp.pi)) * 180 / jnp.pi
    randoms_ra = (jnp.arctan2(randoms_positions[..., 1], randoms_positions[..., 0]) % (2 * jnp.pi)) * 180 / jnp.pi
    data_dec = jnp.arcsin(data_positions[..., 2] / data_distances) * 180 / jnp.pi
    randoms_dec = jnp.arcsin(randoms_positions[..., 2] / randoms_distances) * 180 / jnp.pi

    data_pixels = vec2pix(data_positions)
    randoms_pixels = vec2pix(randoms_positions)

    data_regions = []
    randoms_regions = []

    for region, (zmin, zmax) in regions_zranges:
        data_mask = select_region(ra=data_ra, dec=data_dec, region=region, sharding_mesh=sharding_mesh)
        data_mask &= (zmin <= data_redshifts) & (data_redshifts <= zmax)
        randoms_mask = select_region(ra=randoms_ra, dec=randoms_dec, region=region, sharding_mesh=sharding_mesh)
        randoms_mask &= (zmin <= randoms_redshifts) & (randoms_redshifts <= zmax)

        if not data_mask.any():
            raise ValueError("No data in region %s, redshift range %.1f - %.1f. Cannot proceed.", region, zmin, zmax)
        if not randoms_mask.any():
            raise ValueError("No randoms in region %s, redshift range %.1f - %.1f. Cannot proceed.", region, zmin, zmax)

        data_regions.append(data_mask)
        randoms_regions.append(randoms_mask)

    data_pixels_counts = jnp.bincount(data_pixels, weights=None, length=12 * nside**2)
    randoms_pixels_counts = jnp.bincount(randoms_pixels, weights=None, length=12 * nside**2)
    data_but_no_randoms = ((data_pixels_counts != 0) * (randoms_pixels_counts == 0))[data_pixels]

    data_regions = jnp.stack(data_regions)
    randoms_regions = jnp.stack(randoms_regions)

    data_coverage = data_regions.sum(axis=0)
    randoms_coverage = randoms_regions.sum(axis=0)

    if (data_coverage >= 2).any():
        warn(f"Some ({(data_coverage >= 2).sum()}/{data_coverage.size}) data particles are in several regions at once.", RuntimeWarning, stacklevel=2)
    if (randoms_coverage >= 2).any():
        warn(f"Some ({(randoms_coverage >= 2).sum()}/{randoms_coverage.size}) randoms particles are in several regions at once.", RuntimeWarning, stacklevel=2)
    if (data_coverage < 1).any():
        warn(f"Some ({(data_coverage == 0).sum()}/{data_coverage.size}) data particles are in no region at all.", RuntimeWarning, stacklevel=2)
    if (randoms_coverage < 1).any():
        warn(f"Some ({(randoms_coverage == 0).sum()}/{randoms_coverage.size}) randoms particles are in no region at all.", RuntimeWarning, stacklevel=2)

    return NAM_args(
        data_pixels=data_pixels,
        randoms_pixels=randoms_pixels,
        data_regions=data_regions,
        randoms_regions=randoms_regions,
        data_to_remove=data_but_no_randoms,
        nside=nside,
        apply_to=apply_to,
    )


@jax.jit(static_argnames=["apply_to", "nside"])
def apply_NAM(
    data_weights: jax.Array,
    randoms_weights: jax.Array,
    data_regions: jax.Array,
    randoms_regions: jax.Array,
    data_pixels: jax.Array,
    randoms_pixels: jax.Array,
    nside: int = 256,
    apply_to: Literal["data", "randoms"] = "randoms",
) -> jax.Array:
    """
    Compute weights to apply the angular integral constraint  / nulled angular modes at the catalog level.

    Parameters
    ----------
    data_weights : jax.Array
        Input data weights, shape (n_d,).
    randoms_weights : jax.Array
        Input randoms weights, shape (n_r,).
    data_regions : jax.Array
        Input masks for each region for the data, shape (r, n_d,).
    randoms_regions : jax.Array
        Input masks for each region for the data, shape (r, n_r,).
    data_pixels : jax.Array
        HEALPix of the data, shape (n_d,).
    randoms_pixel : jax.Array
        HEALPix of the randoms, shape (n_r,).
    nside : int, optional
        Which NSIDE was used in the pixelization, by default 256.
    apply_to : Literal["data", "randoms"], optional
        Whether to compute weights for the data or the randoms, by default "randoms"

    Returns
    -------
    jax.Array
        Weights to enforce AIC/NAM, to apply multiplicatively to the original data or randoms weights, shape (n_d,) or (n_r,).

    Raises
    ------
    ValueError
        If ``apply_to`` is an unsupported value.

    Notes
    -----
    * ``nside`` cannot be inferred dynamically, otherwise the function would not be compatible with ``jax.jit``.
    * Region masks must be perfectly complementary with complete coverage of the particles.
        * For uncovered particles, the returned weight will be 1.
        * For doubly covered particles, the returned weight will be the sum of the weights in each region
    """
    if apply_to == "data":
        data_weights, randoms_weights = randoms_weights, data_weights
        data_regions, randoms_regions = randoms_regions, data_regions
        data_pixels, randoms_pixels = randoms_pixels, data_pixels
    elif apply_to == "randoms":
        pass
    else:
        raise ValueError('Can only apply to randoms or data, not "%s"!', apply_to)

    alphas = (data_weights * data_regions).sum(axis=-1) / (randoms_weights * randoms_regions).sum(axis=-1)
    data_weights_binned = bincount(data_pixels, weights=data_regions * data_weights, length=12 * nside**2)
    randoms_weights_binned = bincount(randoms_pixels, weights=randoms_regions * randoms_weights, length=12 * nside**2)
    nam_weights_binned = jnp.where(
        randoms_weights_binned == 0,
        0.0,  # don't care, will never be applied
        data_weights_binned / (alphas[..., None] * randoms_weights_binned),
    )  # NOTE: this is not reverse-differentiation compatible
    nam_weights = jnp.where(randoms_regions, nam_weights_binned[..., randoms_pixels], 0.0)
    return nam_weights.sum(axis=range(nam_weights.ndim - 1)) + jnp.invert(randoms_regions.any(axis=0))  # sum over regions and add 1 where no region


def _read_mesh_to_fkp(
    fkp_field: FKPField | tuple[FKPField, ...],
    mesh: RealMeshField,
    resampler: str = "cic",
    compensate: bool = True,
) -> FKPField | tuple[FKPField, ...]:
    if isinstance(fkp_field, tuple):
        return tuple(_read_mesh_to_fkp(fkp_field=_fkp_field, mesh=mesh, resampler=resampler, compensate=compensate) for _fkp_field in fkp_field)
    else:
        data = fkp_field.data.clone(weights=fkp_field.data.weights * (1 + mesh.read(fkp_field.data.positions, resampler=resampler, compensate=compensate)))
    return fkp_field.clone(data=data)


def _read_data(
    fkp_field: FKPField | tuple[FKPField, ...],
    theory: ObservableTree,
    seed: jax.Array,
    los: Literal["local", "x", "y", "z"],
    unitary_amplitude: bool,
) -> FKPField | tuple[FKPField, ...]:
    mattrs = fkp_field[0].attrs if isinstance(fkp_field, tuple) else fkp_field.attrs
    mesh = generate_anisotropic_gaussian_mesh(
        mattrs,
        poles=theory,
        seed=seed,
        los=los,
        unitary_amplitude=unitary_amplitude,
    )
    fkp_field = _read_mesh_to_fkp(fkp_field, mesh)
    del mesh
    return fkp_field


def _get_pk(*fkp_fields, fkp_norm, binner, los):
    num_shotnoise = compute_fkp2_shotnoise(*fkp_fields, bin=binner)
    fkp_meshs = [fkp_field.paint(resampler="tsc", interlacing=3, compensate=True, out="real") for fkp_field in fkp_fields]
    del fkp_fields
    pk = compute_mesh2_spectrum(*fkp_meshs, bin=binner, los={"local": "firstpoint"}.get(los, los))
    return pk.clone(
        norm=fkp_norm,
        num_shotnoise=num_shotnoise,
    )


def _update_fkp(data_weights, randoms_weights, fkp_field, estimator_weights):
    return fkp_field.clone(
        data=fkp_field.data.clone(
            weights=data_weights * getattr(fkp_field.data, estimator_weights, 1.0),
        ),
        randoms=fkp_field.randoms.clone(
            weights=randoms_weights * getattr(fkp_field.randoms, estimator_weights, 1.0),
        ),
    )


def mock_survey_catalog(
    # Catalogs
    *fkp_fields: FKPField | tuple[FKPField, FKPField],
    # Mock generation
    theory: ObservableTree,
    seed: jax.Array,
    los: Literal["local", "x", "y", "z"],
    unitary_amplitude: bool,
    # Effects
    ric_args: RIC_args | tuple[RIC_args, RIC_args] | None,
    amr_args: AMR_args | tuple[AMR_args, AMR_args] | None,
    nam_args: NAM_args | tuple[NAM_args, NAM_args] | None,
    # Final P(k) estimation
    binner: BinMesh2SpectrumPoles,
    fkp_norms: list[jax.Array],
    estimator_weights: str | None,
    # For region renormalization (need to be concatenated if multiple catalogs)
    data_regions: jax.Array | tuple[jax.Array, jax.Array] | None = None,
    randoms_regions: jax.Array | tuple[jax.Array, jax.Array] | None = None,
    # Mesh generation
    meshattrs: MeshAttrs | None = None,
) -> list[Mesh2SpectrumPoles]:
    """
    Get the power spectrum of a mock survey given an input theory, a seed and a set of observational effects.

    Parameters
    ----------
    *fkp_fields : FKPField | tuple[FKPField, FKPField]
        FKP fields containing data and randoms information. The data shouldn't be clustered (*i.e.* the "data" should also be randoms), but the FKP field serves to designate data and randoms amongst the original randoms. One field per desired output power spectrum. Example: NGC and SGC can be provided as two separate FKP fields, to get two output spectra. Pass several **tuples** of FKP fields to compute cross-spectra, for example (LRG_NGC, ELG_NGC) and (LRG_SGC, ELG_SGC) to get the LRGxELG cross-spectra in NGC and SGC.
    theory : ObservableTree
        Fiducial theoretical power spectrum for the mock survey.
    seed : jax.Array
        Random seed for the mock survey mesh generation.
    los : Literal["local", "x", "y", "z"]
        Line of sight definition for the mock generation.
    unitary_amplitude : bool
        Whether to use unitary amplitude for the mock survey mesh generation.
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
    estimator_weights : str | None, optional
        Name of the weights stored in the FKP fields' particle fields ``extra_fields`` to use as extra weight at estimation time. For example, FKP or OQE weights should not be applied for RIC and AMR but should be added at the spectrum estimation time. Default is ``None`` (no extra weight).
    data_regions : jax.Array | tuple[jax.Array, jax.Array] | None, optional
        Regions for the data to randoms renormalization. By default None. These can typically be provided as the ``data_regions`` attribute in ``ric_args``, ``amr_args`` or ``nam_args``.
    randoms_regions : jax.Array | tuple[jax.Array, jax.Array] | None, optional
        Regions for the data to randoms renormalization. By default None. These can typically be provided as the ``randoms_regions`` attribute in ``ric_args``, ``amr_args`` or ``nam_args``.
    meshattrs: MeshAttrs | None = None,
        If not None, one mock mesh will be generated with these attributes instead of one mock mesh per FKP field with the FKP field's attributes. This mesh should cover all particles in all FKP fields. Default is None.

    Returns
    -------
    list[Mesh2SpectrumPoles]
        Power spectra of one realization of the mock survey; one for each FKP field.

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

    Examples
    --------
    For one tracer (autocorrelation), with NGC and SGC as two separate FKP fields, and RIC and AMR applied to both:
    >>> fw_jit = jax.jit(mock_survey_catalog, static_argnames=["los", "unitary_amplitude"])
    >>> pk_sgc, pk_ngc = fw_jit(
            fkp_sgc,
            fkp_ngc,
            theory=theory,
            seed=jax.random.key(42),
            los="local",
            unitary_amplitude=True,
            ric_args=ric_args,
            amr_args=amr_args,
            nam_args=None,
            fkp_norms=fkp_norms,
            binner=binner,
            data_regions=ric_args.data_regions,
            randoms_regions=ric_args.randoms_regions,
        )

    For two tracers (cross-correlation), with NGC and SGC as two separate FKP fields, and RIC and AMR applied to both:
    >>> fw_jit = jax.jit(mock_survey_catalog, static_argnames=["los", "unitary_amplitude"])
    >>> pk_sgc, pk_ngc = fw_jit(
            (fkp_sgc_tracer1, fkp_sgc_tracer2),
            (fkp_ngc_tracer1, fkp_ngc_tracer2),
            theory=theory,
            seed=jax.random.key(42),
            los="local",
            unitary_amplitude=True,
            ric_args=(ric_args_tracer1, ric_args_tracer2),
            amr_args=(amr_args_tracer1, amr_args_tracer2),
            nam_args=None,
            fkp_norms=fkp_norms,
            binner=binner,
            data_regions=tuple(ric_arg.data_regions for ric_arg in ric_args),
            randoms_regions=tuple(ric_arg.randoms_regions for ric_arg in ric_args),
        )

    For one tracer and a single region, with RIC and NAM applied:
    >>> fw_jit = jax.jit(mock_survey_catalog, static_argnames=["los", "unitary_amplitude"])
    >>> pk = fw_jit(
            theory=theory,
            seed=jax.random.key(42),
            los="local",
            unitary_amplitude=True,
            ric_args=ric_args,
            amr_args=None,
            nam_args=nam_args,
            fkp_norms=fkp_norms,
            binner=binner,
            data_regions=ric_args.data_regions,
            randoms_regions=ric_args.randoms_regions,
        )
    """
    ric_args = () if ric_args is None else (ric_args if isinstance(ric_args, tuple) else (ric_args,))
    amr_args = () if amr_args is None else (amr_args if isinstance(amr_args, tuple) else (amr_args,))
    nam_args = () if nam_args is None else (nam_args if isinstance(nam_args, tuple) else (nam_args,))
    data_regions = () if data_regions is None else (data_regions if isinstance(data_regions, tuple) else (data_regions,))
    randoms_regions = () if randoms_regions is None else (randoms_regions if isinstance(randoms_regions, tuple) else (randoms_regions,))
    estimator_weights = estimator_weights or ""

    sharding_mesh = get_sharding_mesh()
    if meshattrs is None:
        keys = jax.random.split(seed, len(fkp_fields))
        fkp_fields = [_read_data(fkp_field, theory, key, los, unitary_amplitude) for fkp_field, key in zip(fkp_fields, keys, strict=True)]
        # fkp_field can be a tuple, but the same key will be used for all fields in the tuple, which is what we want since they should be read from the same mesh
    else:
        mesh = generate_anisotropic_gaussian_mesh(
            meshattrs,
            poles=theory,
            seed=seed,
            los=los,
            unitary_amplitude=unitary_amplitude,
        )
        fkp_fields = [_read_mesh_to_fkp(fkp_field, mesh) for fkp_field in fkp_fields]
        del mesh

    # ensure all fields are tuples, for easier processing later
    # they will be unpacked for P(k) anyways
    fkp_fields = tuple(fkp_field if isinstance(fkp_field, tuple) else (fkp_field,) for fkp_field in fkp_fields)

    # Length of list = 1 or 2 dependent on whether we are doing auto or cross spectra
    data_weights = [
        local_concatenate([fkp_field.data.weights for fkp_field in region_group], axis=0, sharding_mesh=sharding_mesh)
        for region_group in zip(*fkp_fields, strict=True)
    ]
    randoms_weights = [
        local_concatenate([fkp_field.randoms.weights for fkp_field in region_group], axis=0, sharding_mesh=sharding_mesh)
        for region_group in zip(*fkp_fields, strict=True)
    ]

    for idx, ric_arg in enumerate(ric_args):
        # if ric_args was set to None in call, ric_args is now an empty tuple, so the loop will be skipped
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
            data_extremes=amr_arg.data_extremes,
            randoms_extremes=amr_arg.randoms_extremes,
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
        # Need to re-enforce RIC after AMR. Corresponds to adding w_sys to randoms by joining on TARGETID_DATA in the DESI pipeline
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
            nside=nam_arg.nside,
            apply_to=nam_arg.apply_to,
        )
        if nam_arg.apply_to == "data":
            data_weights[idx] = data_weights[idx] * nam_weights
        else:
            randoms_weights[idx] = randoms_weights[idx] * nam_weights

    # global randoms renormalization per region
    for idx, (_randoms_regions, _data_regions) in enumerate(zip(randoms_regions, data_regions, strict=True)):
        global_alpha = data_weights[idx].sum() / randoms_weights[idx].sum()
        alphas = (data_weights[idx] * _data_regions).sum(axis=-1) / (randoms_weights[idx] * _randoms_regions).sum(axis=-1)
        correction = (_randoms_regions * alphas[..., None] / global_alpha).sum(axis=0) + jnp.invert(
            _randoms_regions.any(axis=0)
        )  # apply alpha/global_alpha inside regions, 1 outside
        randoms_weights[idx] = randoms_weights[idx] * correction

    # Rebuild FKP fields
    # Split back the weights
    split_indices_data = tuple(
        list(itertools.accumulate([fkp_field.data.weights.shape[0] for fkp_field in region_group]))[:-1]
        for region_group in zip(
            *fkp_fields,
            strict=True,
        )
    )
    jax.block_until_ready(split_indices_data)
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
        list(itertools.accumulate([fkp_field.randoms.weights.shape[0] for fkp_field in region_group]))[:-1]
        for region_group in zip(
            *fkp_fields,
            strict=True,
        )
    )
    jax.block_until_ready(split_indices_randoms)
    randoms_weights = tuple(
        zip(
            *(
                local_split(randoms_weight, split_idx, axis=0, sharding_mesh=sharding_mesh)
                for randoms_weight, split_idx in zip(randoms_weights, split_indices_randoms, strict=True)
            ),
            strict=True,
        )
    )

    fkp_fields = jax.tree.map(_update_fkp, data_weights, randoms_weights, fkp_fields, jax.tree.map(lambda _: estimator_weights, data_weights))
    pks = [_get_pk(*fkp_field, fkp_norm=fkp_norm, binner=binner, los=los) for fkp_field, fkp_norm in zip(fkp_fields, fkp_norms, strict=True)]
    return pks


def mock_survey_mesh(
    # Gaussian mock generation
    theory: ObservableTree,
    seed: jnp.ndarray,
    los: Literal["local", "x", "y", "z"],
    unitary_amplitude: bool,
    # Final P(k) estimation
    binner: BinMesh2SpectrumPoles,
    norm: jnp.array,
    # selections
    selection1: RealMeshField,
    selection2: RealMeshField,
    ric: bool,
    nbins: int = 1000,
    # regions
    ric_regions: list[jax.Array] | None = None,
) -> Mesh2SpectrumPoles:
    """
    Apply mesh-based geometry forward modeling to a theoretical power spectrum.

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
    binner : BinMesh2SpectrumPoles
        Binning operator to compute the output power spectrum.
    norm : jnp.array
        Pre-computed normalization.
    selection1 : RealMeshField
        Survey selection function, i.e. pre-painted randoms catalogs.
    selection2 : RealMeshField
        Survey selection function, i.e. pre-painted randoms catalogs. Should be an independent set of randoms with regard to ``selection1`` to avoid shot noise.
    ric : bool
        Whether to apply radial integral constraint.
    nbins : int
        Number of radial bins for the RIC.
    ric_regions : list[jax.Array] | None
        Optional list of region masks (in the shape of the mesh) where RIC should be performed independently.

    Returns
    -------
    Mesh2SpectrumPoles
        Realization of an observation of the theory power spectrum.

    Notes
    -----
    Unlike catalog based approaches, there is no data-to-random ratio renormalization per region since there are no randoms.
    """
    ric_regions = ric_regions or [jnp.ones_like(selection1, dtype=bool)]
    # Generate a gaussian mesh mock with exact required theory P(k)
    mattrs = selection1.attrs
    _mesh = generate_anisotropic_gaussian_mesh(
        mattrs,
        poles=theory,
        seed=seed,
        los=los,
        unitary_amplitude=unitary_amplitude,
    )
    mesh1 = _mesh * selection1
    mesh2 = _mesh * selection2
    del _mesh
    if ric:
        dmin = jnp.min(mattrs.boxcenter - mattrs.boxsize / 2.0)
        dmax = (1.0 + 1e-9) * jnp.sqrt(jnp.sum((mattrs.boxcenter + mattrs.boxsize / 2.0) ** 2))
        edges = jnp.linspace(dmin, dmax, nbins)
        rnorm = jnp.sqrt(sum(xx**2 for xx in mattrs.rcoords(sparse=True)))
        ibin = jnp.digitize(rnorm, edges, right=False)
        for region in ric_regions:
            bw1 = jnp.bincount(ibin.ravel(), weights=(mesh1 * region).ravel(), length=len(edges) + 1)
            b1 = jnp.bincount(ibin.ravel(), weights=(selection1 * region).ravel(), length=len(edges) + 1)
            bw2 = jnp.bincount(ibin.ravel(), weights=(mesh2 * region).ravel(), length=len(edges) + 1)
            b2 = jnp.bincount(ibin.ravel(), weights=(selection2 * region).ravel(), length=len(edges) + 1)
            # Integral constraint
            bw1 = bw1 / jnp.where(b1 == 0.0, 1.0, b1)  # (integral of W * delta) / (integral of W)
            mesh1 -= bw1[ibin].reshape(selection1.shape) * selection1 * region
            bw2 = bw2 / jnp.where(b2 == 0.0, 1.0, b2)  # (integral of W * delta) / (integral of W)
            mesh2 -= bw2[ibin].reshape(selection2.shape) * selection2 * region
    pk = compute_mesh2_spectrum(mesh1, mesh2, bin=binner, los={"local": "firstpoint"}.get(los, los))
    return pk.clone(norm=norm)
