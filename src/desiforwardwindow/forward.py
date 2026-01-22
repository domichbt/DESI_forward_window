"""Forward modeling of observational effects."""

from typing import Literal
from warnings import warn

import jax
import jax.numpy as jnp
from jaxpower import (
    BinMesh2SpectrumPoles,
    FKPField,
    ParticleField,
    RealMeshField,
    compute_fkp2_shotnoise,
    compute_mesh2_spectrum,
    generate_anisotropic_gaussian_mesh,
)
from jaxpower.mesh import make_array_from_process_local_data
from lsstypes import Mesh2SpectrumPoles, ObservableTree

from .utils import bincount, bincount_2d, make_jax_dataclass, select_region

AMRArgsFKP = make_jax_dataclass(
    class_name="AMRArgsFKP",
    dynamic_fields=[
        "data_templates_digitized",
        "mask_extremes_in_data",
        "data_templates_normalized",
        "data_regions",
        "factors",
        "constants",
    ],
    aux_fields=["n_bins"],
    types_fields={
        "data_templates_digitized": jnp.ndarray,
        "mask_extremes_in_data": jnp.ndarray,
        "data_templates_normalized": jnp.ndarray,
        "data_regions": list[jnp.ndarray],
        "factors": list[jnp.ndarray],
        "constants": list[jnp.ndarray],
        "n_bins": int,
    },
)

RICArgsFKP = make_jax_dataclass(
    class_name="RICArgsFKP",
    dynamic_fields=[
        "data_distances_digitized",
        "data_to_remove",
        "randoms_weights_binned",
        "data_regions",
        "randoms_regions",
    ],
    aux_fields=["n_bins", "regions"],
    types_fields={
        "data_distances_digitized": jnp.ndarray,
        "data_to_remove": jnp.ndarray,
        "data_regions": list[jnp.ndarray],
        "randoms_regions": list[jnp.ndarray],
        "randoms_weights_binned": list[jnp.ndarray],
        "regions": list[str],
        "n_bins": int,
    },
)

NAMArgsFKP = make_jax_dataclass(
    class_name="NAMArgsFKP",
    dynamic_fields=[
        "data_pixels",
        "randoms_pixels",
        "data_to_remove",
        "randoms_weights_binned",
        "data_regions",
        "randoms_regions",
    ],
    aux_fields=["nside"],
    types_fields={
        "data_pixels": jnp.ndarray,
        "randoms_pixels": jnp.ndarray,
        "data_to_remove": jnp.ndarray,
        "data_regions": list[jnp.ndarray],
        "randoms_regions": list[jnp.ndarray],
        "randoms_weights_binned": list[jnp.ndarray],
        "nside": int,
    },
)


def prepare_AMR_FKP(
    fkp_field: FKPField,
    redshifts: tuple[jnp.ndarray, jnp.ndarray],
    regions_zranges: list[tuple[str, tuple[float, float]]],
    # AMR specific data
    template_values_data: jnp.ndarray,
    template_values_randoms: jnp.ndarray,
    # AMR specific parameters
    tail: float = 0.5,
    bin_margin: float = 1e-7,
    n_bins: int = 10,
) -> AMRArgsFKP:
    """
    Precompute all arguments necessary to get Angular Mode Removal in :py:func:`mock_survey_FKP`.

    Parameters
    ----------
    fkp_field : ParticleField
        Field containing positions and weights of all the particles.
    template_values_data : jnp.ndarray
        Values of the templates for the data.
    template_values_randoms : jnp.ndarray
        Values of the templates for the randoms.
    tail : float, optional
        Percentile of the (random's) template value distribution to remove, by default 0.5
    bin_margin : float, optional
        Margin on the side of edges, by default 1e-7
    n_bins : int, optional
        Number of bins for each template in the regression, by default 10

    Returns
    -------
    AMRArgsFKP
        Dataclass containing all information needed by :py:func:`mock_survey_FKP` to apply AMR.
    """
    # Select the regions
    data_distances = jnp.sqrt(jnp.power(fkp_field.data.positions, 2).sum(axis=-1))
    randoms_distances = jnp.sqrt(jnp.power(fkp_field.randoms.positions, 2).sum(axis=-1))
    data_ra = (jnp.arctan2(fkp_field.data.positions[..., 1], fkp_field.data.positions[..., 0]) % (2 * jnp.pi)) * 180 / jnp.pi
    randoms_ra = (jnp.arctan2(fkp_field.randoms.positions[..., 1], fkp_field.randoms.positions[..., 0]) % (2 * jnp.pi)) * 180 / jnp.pi
    data_dec = jnp.arcsin(fkp_field.data.positions[..., 2] / data_distances) * 180 / jnp.pi
    randoms_dec = jnp.arcsin(fkp_field.randoms.positions[..., 2] / randoms_distances) * 180 / jnp.pi

    data_redshift = redshifts[0]
    randoms_redshift = redshifts[1]

    templates_lower_tails = jnp.percentile(template_values_randoms.T, tail / 2, axis=1, method="higher")
    templates_upper_tails = jnp.percentile(template_values_randoms.T, 100 - tail / 2, axis=1, method="lower")

    mask_extremes_r = jnp.invert(
        jnp.any(
            (template_values_randoms < templates_lower_tails).T | (template_values_randoms > templates_upper_tails).T,
            axis=0,
        )
    )

    mask_extremes_d = jnp.invert(
        jnp.any(
            (template_values_data < templates_lower_tails).T | (template_values_data > templates_upper_tails).T,
            axis=0,
        )
    )

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
    factors = []
    constants = []

    for region, (zmin, zmax) in regions_zranges:
        data_mask = select_region(ra=data_ra, dec=data_dec, region=region)
        data_mask &= (zmin <= data_redshift) & (data_redshift <= zmax)
        randoms_mask = select_region(ra=randoms_ra, dec=randoms_dec, region=region)
        randoms_mask &= (zmin <= randoms_redshift) & (randoms_redshift <= zmax)

        # Binned weights and jacobian are used in the solution
        randoms_weights_binned = bincount_2d(
            templates_digitized_r.T,
            weights=fkp_field.randoms.weights * mask_extremes_r * randoms_mask,  # set extremes weights to 0
            length=n_bins + 1,
        )[:, 1:, ...]

        masked_randoms_weights = fkp_field.randoms.weights * mask_extremes_r * randoms_mask
        masked_templates_normalized = templates_normalized_r * mask_extremes_r[:, None]
        wt2 = jnp.concatenate([jnp.ones_like(fkp_field.randoms.weights)[..., None], masked_templates_normalized], axis=-1)

        jacobian = jnp.moveaxis(
            bincount_vmapped(
                templates_digitized_r.T,
                (masked_randoms_weights[:, None] * wt2).T,
                0,
                n_bins + 1,
            ),
            0,
            -1,
        )[:, 1:, ...]

        normalization = (
            fkp_field.data.weights * mask_extremes_d * data_mask
        ).sum() / masked_randoms_weights.sum()  # without the updated data weights: approximate

        # Ravel everything to take advantage of matrix operations
        jacobian = jacobian.reshape((-1, jacobian.shape[-1]))
        randoms_weights_binned = randoms_weights_binned.reshape((-1,))
        # Precompute a matrix that will be reused
        transpose_jw = normalization * jacobian.T * randoms_weights_binned  # matrix product with diag matrix is just numpy product with the diag vector
        factor = jnp.linalg.inv(transpose_jw.dot(jacobian)).dot(transpose_jw)
        constant = normalization * factor.dot(randoms_weights_binned)

        data_regions.append(data_mask)
        factors.append(factor)
        constants.append(constant)

    # pre-computed templates
    data_templates_digitized = templates_digitized_d
    data_templates_normalized = templates_normalized_d

    return AMRArgsFKP(
        data_templates_digitized=data_templates_digitized,
        mask_extremes_in_data=mask_extremes_d,
        data_templates_normalized=data_templates_normalized,
        data_regions=data_regions,
        factors=factors,
        constants=constants,
        n_bins=n_bins,
    )


def prepare_RIC_FKP(
    fkp_field: FKPField,
    regions: list[str],
    boxcenter: jnp.ndarray,
    boxsize: jnp.ndarray,
    # RIC specific parameters
    n_bins: int,
) -> RICArgsFKP:
    """
    Precompute all arguments necessary to :py:func:`mock_survey_FKP``.

    Parameters
    ----------
    fkp_field : ParticleField
        Field containing positions and weights of all the particles.
    boxcenter : jnp.ndarray
        Coordinates of the center of the box used for the power spectrum computation.
    boxsize : jnp.ndarray
        Sizes of the sides of the box used for the power spectrum computation.
    n_bins : int
        Number of distance bins to used.

    Returns
    -------
    RICArgsFKP
        Dataclass containing all information needed by :py:func:`mock_survey_FKP` to apply RIC.
    """
    dmin = jnp.min(boxcenter - boxsize / 2.0)
    dmax = (1.0 + 1e-9) * jnp.sqrt(jnp.sum((boxcenter + boxsize / 2.0) ** 2))
    distance_edges = jnp.linspace(dmin, dmax, n_bins)
    data_distances = jnp.sqrt(jnp.power(fkp_field.data.positions, 2).sum(axis=-1))
    randoms_distances = jnp.sqrt(jnp.power(fkp_field.randoms.positions, 2).sum(axis=-1))
    data_distances_digitized = jnp.digitize(data_distances, bins=distance_edges)  # could be made faster with jnp.floor
    randoms_distances_digitized = jnp.digitize(randoms_distances, bins=distance_edges)

    # region selection
    data_ra = (jnp.arctan2(fkp_field.data.positions[..., 1], fkp_field.data.positions[..., 0]) % (2 * jnp.pi)) * 180 / jnp.pi
    randoms_ra = (jnp.arctan2(fkp_field.randoms.positions[..., 1], fkp_field.randoms.positions[..., 0]) % (2 * jnp.pi)) * 180 / jnp.pi
    data_dec = jnp.arcsin(fkp_field.data.positions[..., 2] / data_distances) * 180 / jnp.pi
    randoms_dec = jnp.arcsin(fkp_field.randoms.positions[..., 2] / randoms_distances) * 180 / jnp.pi

    data_regions = []
    randoms_regions = []
    randoms_weights_binned = []
    for region in regions:
        data_mask = select_region(ra=data_ra, dec=data_dec, region=region)
        randoms_mask = select_region(ra=randoms_ra, dec=randoms_dec, region=region)
        randoms_weights_binned.append(jnp.bincount(randoms_distances_digitized, weights=fkp_field.randoms.weights * randoms_mask, length=n_bins + 1)[1:])
        data_regions.append(data_mask)
        randoms_regions.append(randoms_mask)

    data_distances_counts = jnp.bincount(data_distances_digitized, weights=None, length=n_bins + 1)[1:]
    randoms_distances_counts = jnp.bincount(randoms_distances_digitized, weights=None, length=n_bins + 1)[1:]
    data_to_remove = (data_distances_counts != 0) * (randoms_distances_counts == 0)

    return RICArgsFKP(
        data_distances_digitized=data_distances_digitized,
        randoms_weights_binned=randoms_weights_binned,
        data_regions=data_regions,
        randoms_regions=randoms_regions,
        data_to_remove=data_to_remove[data_distances_digitized - 1],
        regions=regions,
        n_bins=n_bins,
    )


def prepare_NAM_FKP(
    fkp_field: FKPField,
    redshifts: tuple[jnp.ndarray, jnp.ndarray],
    regions_zranges: list[tuple[str, tuple[float, float]]],
    # RIC specific parameters
    nside: int,
) -> NAMArgsFKP:
    """
    Precompute all arguments necessary to :py:func:`mock_survey_FKP`.

    Parameters
    ----------
    fkp_field : FKPField
        Field containing positions and weights of all the particles.
    nside : int
        Resolution for the pixel binning (power of two).

    Returns
    -------
    NAMArgsFKP
        Dataclass containing all information needed by :py:func:`mock_survey_FKP` to apply NAM/AIC.
    """

    def _vec2pix(positions):
        import healpy as hp

        return hp.vec2pix(nside, *positions.T)

    def vec2pix(positions):
        return jax.pure_callback(_vec2pix, jax.ShapeDtypeStruct(positions.shape[:1], jnp.int64), positions)

    from jax.experimental.shard_map import shard_map
    from jax.sharding import PartitionSpec as P
    from jaxpower.mesh import get_sharding_mesh

    sharding_mesh = get_sharding_mesh()

    if sharding_mesh.axis_names:
        vec2pix = shard_map(vec2pix, mesh=sharding_mesh, in_specs=P(sharding_mesh.axis_names), out_specs=P(sharding_mesh.axis_names))

    # Select the regions
    data_distances = jnp.sqrt(jnp.power(fkp_field.data.positions, 2).sum(axis=-1))
    randoms_distances = jnp.sqrt(jnp.power(fkp_field.randoms.positions, 2).sum(axis=-1))
    data_ra = (jnp.arctan2(fkp_field.data.positions[..., 1], fkp_field.data.positions[..., 0]) % (2 * jnp.pi)) * 180 / jnp.pi
    randoms_ra = (jnp.arctan2(fkp_field.randoms.positions[..., 1], fkp_field.randoms.positions[..., 0]) % (2 * jnp.pi)) * 180 / jnp.pi
    data_dec = jnp.arcsin(fkp_field.data.positions[..., 2] / data_distances) * 180 / jnp.pi
    randoms_dec = jnp.arcsin(fkp_field.randoms.positions[..., 2] / randoms_distances) * 180 / jnp.pi

    data_redshift = redshifts[0]
    randoms_redshift = redshifts[1]

    data_pixels = vec2pix(fkp_field.data.positions)
    randoms_pixels = vec2pix(fkp_field.randoms.positions)

    data_regions = []
    randoms_regions = []
    randoms_weights_binned = []

    for region, (zmin, zmax) in regions_zranges:
        data_mask = select_region(ra=data_ra, dec=data_dec, region=region)
        data_mask &= (zmin <= data_redshift) & (data_redshift <= zmax)
        randoms_mask = select_region(ra=randoms_ra, dec=randoms_dec, region=region)
        randoms_mask &= (zmin <= randoms_redshift) & (randoms_redshift <= zmax)

        randoms_weights_binned.append(jnp.bincount(randoms_pixels, weights=fkp_field.randoms.weights * randoms_mask, length=12 * nside**2))
        data_regions.append(data_mask)
        randoms_regions.append(randoms_mask)

    data_pixels_counts = jnp.bincount(data_pixels, weights=None, length=12 * nside**2)
    randoms_pixels_counts = jnp.bincount(randoms_pixels, weights=None, length=12 * nside**2)
    data_but_no_randoms = (data_pixels_counts != 0) * (randoms_pixels_counts == 0)

    return NAMArgsFKP(
        data_pixels=data_pixels,
        randoms_pixels=randoms_pixels,
        data_to_remove=data_but_no_randoms[data_pixels],
        randoms_weights_binned=randoms_weights_binned,
        data_regions=data_regions,
        randoms_regions=randoms_regions,
        nside=nside,
    )


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
    data: ParticleField,
    randoms: ParticleField,
    regions: list[str],
    # RIC specific parameters
    n_bins: int,
    apply_to: Literal["data", "randoms"],
) -> RIC_args:
    """
    Prepare arguments necessary to applying RIC in :py:func:`mock_survey_catalog`.

    Parameters
    ----------
    data : ParticleField
        Field containing positions and weights of the data particles.
    randoms : ParticleField
        Field containing positions and weights of the randoms particles.
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
    boxcenter = data.attrs.boxcenter
    boxsize = data.attrs.boxsize
    dmin = jnp.min(boxcenter - boxsize / 2.0)
    dmax = (1.0 + 1e-9) * jnp.sqrt(jnp.sum((boxcenter + boxsize / 2.0) ** 2))
    distance_edges = jnp.linspace(dmin, dmax, n_bins)
    data_distances = jnp.sqrt(jnp.power(data.positions, 2).sum(axis=-1))
    randoms_distances = jnp.sqrt(jnp.power(randoms.positions, 2).sum(axis=-1))
    data_distances_digitized = jnp.digitize(data_distances, bins=distance_edges)  # could be made faster with jnp.floor
    randoms_distances_digitized = jnp.digitize(randoms_distances, bins=distance_edges)

    # region selection
    data_ra = (jnp.arctan2(data.positions[..., 1], data.positions[..., 0]) % (2 * jnp.pi)) * 180 / jnp.pi
    randoms_ra = (jnp.arctan2(randoms.positions[..., 1], randoms.positions[..., 0]) % (2 * jnp.pi)) * 180 / jnp.pi
    data_dec = jnp.arcsin(data.positions[..., 2] / data_distances) * 180 / jnp.pi
    randoms_dec = jnp.arcsin(randoms.positions[..., 2] / randoms_distances) * 180 / jnp.pi

    data_regions = []
    randoms_regions = []
    for region in regions:
        data_mask = select_region(ra=data_ra, dec=data_dec, region=region)
        randoms_mask = select_region(ra=randoms_ra, dec=randoms_dec, region=region)
        if not data_mask.any():
            raise ValueError("No data in region %s. Cannot proceed.", region)
        if not randoms_mask.any():
            raise ValueError("No randoms in region %s. Cannot proceed.", region)
        data_regions.append(data_mask)
        randoms_regions.append(randoms_mask)

    data_distances_counts = jnp.bincount(data_distances_digitized, weights=None, length=n_bins + 1)[1:]
    randoms_distances_counts = jnp.bincount(randoms_distances_digitized, weights=None, length=n_bins + 1)[1:]
    data_to_remove = (data_distances_counts != 0) * (randoms_distances_counts == 0)

    data_regions = jnp.stack(data_regions)
    randoms_regions = jnp.stack(randoms_regions)

    data_coverage = data_regions.sum(axis=0)
    randoms_coverage = randoms_regions.sum(axis=0)

    if (data_coverage >= 2).any():
        warn("Some data particles are in several regions at once.", RuntimeWarning, stacklevel=3)
    if (randoms_coverage >= 2).any():
        warn("Some randoms particles are in several regions at once.", RuntimeWarning, stacklevel=3)
    if (data_coverage < 1).any():
        warn("Some data particles are in no region at all.", RuntimeWarning, stacklevel=3)
    if (randoms_coverage < 1).any():
        warn("Some randoms particles are in no region at all.", RuntimeWarning, stacklevel=3)

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
        "n_bins": int,
        "apply_to": Literal["data", "randoms"],
    },
)


def prepare_AMR(
    data: ParticleField,
    randoms: ParticleField,
    redshifts: tuple[jnp.ndarray, jnp.ndarray],
    regions_zranges: list[tuple[str, tuple[float, float]]],
    apply_to: Literal["data", "randoms"],
    # AMR specific data
    template_values_data: jnp.ndarray,
    template_values_randoms: jnp.ndarray,
    # AMR specific parameters
    tail: float = 0.5,
    bin_margin: float = 1e-7,
    n_bins: int = 10,
) -> AMR_args:
    """
    Precompute all arguments necessary to get Angular Mode Removal in :py:func:`mock_survey_FKP`.

    Parameters
    ----------
    data : ParticleField
        Field containing positions and weights of the data particles.
    randoms : ParticleField
        Field containing positions and weights of the randoms particles.
    template_values_data : jnp.ndarray
        Values of the templates for the data.
    template_values_randoms : jnp.ndarray
        Values of the templates for the randoms.
    tail : float, optional
        Percentile of the (random's) template value distribution to remove, by default 0.5
    bin_margin : float, optional
        Margin on the side of edges, by default 1e-7
    n_bins : int, optional
        Number of bins for each template in the regression, by default 10

    Returns
    -------
    AMRArgsFKP
        Dataclass containing all information needed by :py:func:`mock_survey_FKP` to apply AMR.
    """
    data_redshifts = redshifts[0]
    randoms_redshifts = redshifts[1]

    # Shard the extra metadata similarly to data/randoms if necessary
    if data.exchange_direct is not None:
        template_values_data = data.exchange_direct(make_array_from_process_local_data(template_values_data, pad="mean"), pad=0.0)
        data_redshifts = data.exchange_direct(make_array_from_process_local_data(data_redshifts, pad=0.0), pad=0.0)
    else:
        template_values_data = jnp.array(template_values_data)
        data_redshifts = jnp.array(data_redshifts)
    if randoms.exchange_direct is not None:
        template_values_randoms = randoms.exchange_direct(make_array_from_process_local_data(template_values_randoms, pad="mean"), pad=0.0)
        randoms_redshifts = randoms.exchange_direct(make_array_from_process_local_data(randoms_redshifts, pad=0.0), pad=0.0)
    else:
        template_values_randoms = jnp.array(template_values_randoms)
        randoms_redshifts = jnp.array(randoms_redshifts)

    # Select the regions
    data_distances = jnp.sqrt(jnp.power(data.positions, 2).sum(axis=-1))
    randoms_distances = jnp.sqrt(jnp.power(randoms.positions, 2).sum(axis=-1))
    data_ra = (jnp.arctan2(data.positions[..., 1], data.positions[..., 0]) % (2 * jnp.pi)) * 180 / jnp.pi
    randoms_ra = (jnp.arctan2(randoms.positions[..., 1], randoms.positions[..., 0]) % (2 * jnp.pi)) * 180 / jnp.pi
    data_dec = jnp.arcsin(data.positions[..., 2] / data_distances) * 180 / jnp.pi
    randoms_dec = jnp.arcsin(randoms.positions[..., 2] / randoms_distances) * 180 / jnp.pi

    templates_lower_tails = jnp.percentile(template_values_randoms, tail / 2, axis=0, method="higher")
    templates_upper_tails = jnp.percentile(template_values_randoms, 100 - tail / 2, axis=0, method="lower")

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
        data_mask = select_region(ra=data_ra, dec=data_dec, region=region)
        data_mask &= (zmin <= data_redshifts) & (data_redshifts <= zmax)

        randoms_mask = select_region(ra=randoms_ra, dec=randoms_dec, region=region)
        randoms_mask &= (zmin <= randoms_redshifts) & (randoms_redshifts <= zmax)

        data_regions.append(data_mask)
        randoms_regions.append(randoms_mask)

        if not data_mask.any():
            raise ValueError("No data in region %s, redshift range %.1f - %.1f. Cannot proceed.", region, zmin, zmax)
        if not randoms_mask.any():
            raise ValueError("No randoms in region %s, redshift range %.1f - %.1f. Cannot proceed.", region, zmin, zmax)

    # pre-computed templates
    data_templates_digitized = jnp.vstack(
        [jnp.full(shape=data.weights.shape, dtype=templates_digitized_d.dtype, fill_value=n_bins - 1), templates_digitized_d.T]
    )
    randoms_templates_digitized = jnp.vstack(
        [jnp.full(shape=randoms.weights.shape, dtype=templates_digitized_r.dtype, fill_value=n_bins - 1), templates_digitized_r.T]
    )

    if apply_to == "data":
        data_templates_normalized = jnp.vstack([jnp.ones_like(data.weights), templates_normalized_d.T])
    else:
        data_templates_normalized = None
    randoms_templates_normalized = jnp.vstack([jnp.ones_like(randoms.weights), templates_normalized_r.T])

    data_regions = jnp.stack(data_regions)
    randoms_regions = jnp.stack(randoms_regions)

    data_coverage = data_regions.sum(axis=0)
    randoms_coverage = randoms_regions.sum(axis=0)

    if (data_coverage >= 2).any():
        warn("Some data particles are in several regions at once.", RuntimeWarning, stacklevel=3)
    if (randoms_coverage >= 2).any():
        warn("Some randoms particles are in several regions at once.", RuntimeWarning, stacklevel=3)
    if (data_coverage < 1).any():
        warn("Some data particles are in no region at all.", RuntimeWarning, stacklevel=3)
    if (randoms_coverage < 1).any():
        warn("Some randoms particles are in no region at all.", RuntimeWarning, stacklevel=3)

    return AMR_args(
        data_regions=data_regions,
        randoms_regions=randoms_regions,
        data_extremes=mask_extremes_d,
        randoms_extremes=mask_extremes_r,
        data_templates_digitized=data_templates_digitized,
        randoms_templates_digitized=randoms_templates_digitized,
        data_templates_normalized=data_templates_normalized,
        randoms_templates_normalized=randoms_templates_normalized,
        n_bins=n_bins,
        apply_to=apply_to,
    )


# offset the output axis by one so that region axis is always first
# vmap only takes positional argument, so always explicitly set minlength to the default 0
bincount_vmapped = jax.vmap(bincount, in_axes=(0, None, None, None), out_axes=1)


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

    data_weights *= data_extremes
    randoms_weights *= randoms_extremes

    # shapes: (regions, N_sys + 1, N_bins)
    data_binned = bincount_vmapped(data_templates_digitized, data_weights * data_regions, 0, n_bins + 1)[..., 1:]
    randoms_binned = bincount_vmapped(randoms_templates_digitized, randoms_weights * randoms_regions, 0, n_bins + 1)[..., 1:]

    # shape: (regions, N_sys + 1, N_bins + 1,  N_sys + 1)
    # The last dimension is for the matrix product with the coefficients vector
    # The middle (N_sys + 1, N_bins + 1) correspond, in spirit, to one big axis
    randoms_templates_binned = bincount_vmapped(
        randoms_templates_digitized, randoms_weights[None, None, ...] * randoms_regions[:, None, ...] * randoms_templates_normalized[None, ...], 0, n_bins + 1
    )[..., 1:].swapaxes(-1, -2)

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
    data: ParticleField,
    randoms: ParticleField,
    regions_zranges: list,
    redshifts: tuple[jax.Array, jax.Array],
    # NAM specific parameters
    nside: int,
    apply_to: Literal["data", "randoms"],
) -> NAM_args:
    """
    Prepare arguments necessary to applying NAM/AIC in :py:func:`mock_survey_catalog`.

    Parameters
    ----------
    data : ParticleField
        Field containing positions and weights of the data particles.
    randoms : ParticleField
        Field containing positions and weights of the randoms particles.
    regions : list[str]
        Regions to split data in.
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
    data_redshifts = redshifts[0]
    randoms_redshifts = redshifts[1]

    # Shard the extra metadata similarly to data/randoms if necessary
    if data.exchange_direct is not None:
        data_redshifts = data.exchange_direct(make_array_from_process_local_data(data_redshifts, pad=0.0), pad=0.0)
    if randoms.exchange_direct is not None:
        randoms_redshifts = randoms.exchange_direct(make_array_from_process_local_data(randoms_redshifts, pad=0.0), pad=0.0)

    def _vec2pix(positions):
        import healpy as hp

        return hp.vec2pix(nside, *positions.T)

    def vec2pix(positions):
        return jax.pure_callback(_vec2pix, jax.ShapeDtypeStruct(positions.shape[:1], jnp.int64), positions)

    from jax.experimental.shard_map import shard_map
    from jax.sharding import PartitionSpec as P
    from jaxpower.mesh import get_sharding_mesh

    sharding_mesh = get_sharding_mesh()

    if sharding_mesh.axis_names:
        vec2pix = shard_map(vec2pix, mesh=sharding_mesh, in_specs=P(sharding_mesh.axis_names), out_specs=P(sharding_mesh.axis_names))

    # Select the regions
    data_distances = jnp.sqrt(jnp.power(data.positions, 2).sum(axis=-1))
    randoms_distances = jnp.sqrt(jnp.power(randoms.positions, 2).sum(axis=-1))
    data_ra = (jnp.arctan2(data.positions[..., 1], data.positions[..., 0]) % (2 * jnp.pi)) * 180 / jnp.pi
    randoms_ra = (jnp.arctan2(randoms.positions[..., 1], randoms.positions[..., 0]) % (2 * jnp.pi)) * 180 / jnp.pi
    data_dec = jnp.arcsin(data.positions[..., 2] / data_distances) * 180 / jnp.pi
    randoms_dec = jnp.arcsin(randoms.positions[..., 2] / randoms_distances) * 180 / jnp.pi

    data_pixels = vec2pix(data.positions)
    randoms_pixels = vec2pix(randoms.positions)

    data_regions = []
    randoms_regions = []

    for region, (zmin, zmax) in regions_zranges:
        data_mask = select_region(ra=data_ra, dec=data_dec, region=region)
        data_mask &= (zmin <= data_redshifts) & (data_redshifts <= zmax)
        randoms_mask = select_region(ra=randoms_ra, dec=randoms_dec, region=region)
        randoms_mask &= (zmin <= randoms_redshifts) & (randoms_redshifts <= zmax)

        if not data_mask.any():
            raise ValueError("No data in region %s, redshift range %.1f - %.1f. Cannot proceed.", region, zmin, zmax)
        if not randoms_mask.any():
            raise ValueError("No randoms in region %s, redshift range %.1f - %.1f. Cannot proceed.", region, zmin, zmax)

        data_regions.append(data_mask)
        randoms_regions.append(randoms_mask)

    data_pixels_counts = jnp.bincount(data_pixels, weights=None, length=12 * nside**2)
    randoms_pixels_counts = jnp.bincount(randoms_pixels, weights=None, length=12 * nside**2)
    data_but_no_randoms = (data_pixels_counts != 0) * (randoms_pixels_counts == 0)

    data_regions = jnp.stack(data_regions)
    randoms_regions = jnp.stack(randoms_regions)

    data_coverage = data_regions.sum(axis=0)
    randoms_coverage = randoms_regions.sum(axis=0)

    if (data_coverage >= 2).any():
        warn("Some data particles are in several regions at once.", RuntimeWarning, stacklevel=3)
    if (randoms_coverage >= 2).any():
        warn("Some randoms particles are in several regions at once.", RuntimeWarning, stacklevel=3)
    if (data_coverage < 1).any():
        warn("Some data particles are in no region at all.", RuntimeWarning, stacklevel=3)
    if (randoms_coverage < 1).any():
        warn("Some randoms particles are in no region at all.", RuntimeWarning, stacklevel=3)

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


def mock_survey_catalog(
    theory: ObservableTree,
    seed: jnp.ndarray,
    los: Literal["local", "x", "y", "z"],
    unitary_amplitude: bool,
    # Data catalog and effects
    fkp_field: FKPField,
    ric_args: RIC_args | None,
    amr_args: AMR_args | None,
    nam_args: NAM_args | None,
    # Final P(k) estimation
    binner: BinMesh2SpectrumPoles,
    fkp_norm: jnp.ndarray,
    # For region renormalization
    data_regions: list[jnp.ndarray] | None = None,
    randoms_regions: list[jnp.ndarray] | None = None,
):
    # don't rename mock_survey_FKP for now, for testing purposes
    mattrs = fkp_field.attrs
    mesh = generate_anisotropic_gaussian_mesh(
        mattrs,
        poles=theory,
        seed=seed,
        los=los,
        unitary_amplitude=unitary_amplitude,
    )
    # Paint it on the catalog -> catalog with geometry and ""clustering""
    data = fkp_field.data.clone(weights=fkp_field.data.weights * (1 + mesh.read(fkp_field.data.positions, resampler="cic", compensate=True)))
    randoms = fkp_field.randoms
    del mesh

    if ric_args is not None:
        ric_weights = apply_RIC(
            data_weights=data.weights,
            randoms_weights=randoms.weights,
            data_regions=ric_args.data_regions,
            randoms_regions=ric_args.randoms_regions,
            data_distances_digitized=ric_args.data_distances_digitized,
            randoms_distances_digitized=ric_args.randoms_distances_digitized,
            n_bins=ric_args.n_bins,
            apply_to=ric_args.apply_to,
        )
        if ric_args.apply_to == "data":
            data = data.clone(weights=data.weights * ric_weights)
        else:
            randoms = randoms.clone(weights=randoms.weights * ric_weights)

    if amr_args is not None:
        amr_weights = apply_AMR(
            data_weights=data.weights,
            randoms_weights=randoms.weights,
            data_regions=amr_args.data_regions,
            randoms_regions=amr_args.randoms_regions,
            data_extremes=amr_args.data_extremes,
            randoms_extremes=amr_args.randoms_extremes,
            data_templates_digitized=amr_args.data_templates_digitized,
            randoms_templates_digitized=amr_args.randoms_templates_digitized,
            data_templates_normalized=amr_args.data_templates_normalized,
            randoms_templates_normalized=amr_args.randoms_templates_normalized,
            n_bins=amr_args.n_bins,
            apply_to=amr_args.apply_to,
        )
        if amr_args.apply_to == "data":
            data = data.clone(weights=data.weights * amr_weights)
        else:
            randoms = randoms.clone(weights=randoms.weights * amr_weights)

    if nam_args is not None:
        nam_weights = apply_NAM(
            data_weights=data.weights,
            randoms_weights=randoms.weights,
            data_regions=nam_args.data_regions,
            randoms_regions=nam_args.randoms_regions,
            data_pixels=nam_args.data_pixels,
            randoms_pixels=nam_args.randoms_pixels,
            nside=nam_args.nside,
            apply_to=nam_args.apply_to,
        )
        if nam_args.apply_to == "data":
            data = data.clone(weights=data.weights * nam_weights)
        else:
            randoms = randoms.clone(weights=randoms.weights * nam_weights)

    # global randoms renormalization per region
    if randoms_regions is not None:
        global_alpha = data.weights.sum() / randoms.weights.sum()
        correction = jnp.ones_like(randoms.weights)
        for data_region, randoms_region in zip(data_regions, randoms_regions, strict=True):
            alpha = jnp.where(data_region, data.weights, 0.0).sum() / jnp.where(randoms_region, randoms.weights, 0.0).sum()
            # Multiply by one outside mask and alpha/global_alpha inside
            correction *= jnp.where(randoms_region, alpha / global_alpha, 1.0)
            # jnp.invert(randoms_region) * 1.0 + randoms_region * alpha / global_alpha
        randoms = randoms.clone(weights=randoms.weights * correction)
    fkp_field = fkp_field.clone(data=data, randoms=randoms)
    num_shotnoise = compute_fkp2_shotnoise(fkp_field, bin=binner)
    fkp_mesh = fkp_field.paint(resampler="tsc", interlacing=3, compensate=True, out="real")
    del fkp_field
    pk = compute_mesh2_spectrum(fkp_mesh, bin=binner, los={"local": "firstpoint"}.get(los, los))
    return pk.clone(
        norm=fkp_norm,
        num_shotnoise=num_shotnoise,
    )


def mock_survey_FKP(
    # Gaussian mock generation
    theory: ObservableTree,
    seed: jnp.ndarray,
    los: Literal["local", "x", "y", "z"],
    unitary_amplitude: bool,
    # Data catalog and effects
    fkp_field: FKPField,
    ric_args: RICArgsFKP | None,
    amr_args: AMRArgsFKP | None,
    nam_args: NAMArgsFKP | None,
    # Final P(k) estimation
    binner: BinMesh2SpectrumPoles,
    fkp_norm: jnp.ndarray,
    # For region renormalization
    data_regions: list[jnp.ndarray] | None = None,
    randoms_regions: list[jnp.ndarray] | None = None,
) -> Mesh2SpectrumPoles:
    """
    Get the power spectrum from a mock survey given an input theory, a seed and a set of observational effects.

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
    fkp_field : FKPField
        FKP field contaning data and randoms information. The data shouldn't be clustered (*i.e.* the "data" should be randoms), but the FKP field serves to designate data and randoms amongst the original randoms.
    ric_args : RICArgsFKP | None
        Fixed precomputed arguments for RIC weights computation; see :py:func:`prepare_RIC_FKP`. Set to ``None`` to not apply RIC.
    amr_args : AMRArgsFKP | None
        Fixed precomputed arguments for AMR weights computation; see :py:func:`prepare_AMR_FKP`. Set to ``None`` to not apply AMR.
    nam_args : NAMArgsFKP | None
            Fixed precomputed arguments for NAM/AIC weights computation; see :py:func:`prepare_NAM_FKP`. Set to ``None`` to not apply NAM/AIC.
    binner : BinMesh2SpectrumPoles
        Binning operator to compute the output power spectrum.
    fkp_norm : jnp.ndarray
        Pre-computed power spectrum norm for the FKP field ``fkp_field``, disregarding any future changes in weights.
    data_regions : list[jnp.ndarray]
        List of masks for the renormalization regions in the data. Usually "N", "S", or split by galactic caps ("N", "SNGC", "SSGC"). May include DES for quasars.
    randoms_regions : list[jnp.ndarray]
        List of masks for the renormalization regions in the randoms. Usually "N", "S", or split by galactic caps ("N", "SNGC", "SSGC"). May include DES for quasars.

    Returns
    -------
    Mesh2SpectrumPoles
        Realization of an observation of the theory power spectrum.

    Notes
    -----
    NAM is a stronger kind of AMR. Applying both AMR and NAM or just NAM will result in the same spectrum, possibly with worse non-linear effects when applying both.
    """
    # Generate a gaussian mesh mock with exact required theory P(k)
    mattrs = fkp_field.attrs
    mesh = generate_anisotropic_gaussian_mesh(
        mattrs,
        poles=theory,
        seed=seed,
        los=los,
        unitary_amplitude=unitary_amplitude,
    )
    # Paint it on the catalog -> catalog with geometry and ""clustering""
    data = fkp_field.data.clone(weights=fkp_field.data.weights * (1 + mesh.read(fkp_field.data.positions, resampler="cic", compensate=True)))
    randoms = fkp_field.randoms
    del mesh
    # Apply RIC if necessary
    if ric_args is not None:
        for data_region, randoms_region, randoms_weights_binned in zip(
            ric_args.data_regions, ric_args.randoms_regions, ric_args.randoms_weights_binned, strict=True
        ):
            alpha = jnp.where(data_region, data.weights, 0.0).sum() / jnp.where(randoms_region, randoms.weights, 0.0).sum()
            data_weights_binned = jnp.bincount(ric_args.data_distances_digitized, weights=data.weights * data_region, length=ric_args.n_bins + 1)[1:]
            ric_weights_binned = jnp.where(
                data_weights_binned == 0,
                0.0,  # don't care, will never be applied
                (alpha * randoms_weights_binned / data_weights_binned),
            )
            data = data.clone(weights=data.weights * jnp.where(data_region, ric_weights_binned[ric_args.data_distances_digitized - 1], 1.0))
    # Apply mode removal (ie linear template regression) if necessary
    # Randoms weights have not been changed yet
    if amr_args is not None:
        for data_region, factor, constant in zip(amr_args.data_regions, amr_args.factors, amr_args.constants, strict=True):
            data_weights_binned = bincount_2d(
                amr_args.data_templates_digitized.T,
                weights=data.weights * amr_args.mask_extremes_in_data * data_region,
                length=amr_args.n_bins + 1,
            )[:, 1:, ...]
            p_opt = factor.dot(data_weights_binned.reshape((-1,))) - constant
            amr_weights = 1 / (1 + p_opt[0] + amr_args.data_templates_normalized.dot(p_opt[1:]))
            data = data.clone(weights=data.weights * jnp.where(data_region, amr_weights, 1.0))
    # Apply NAM if necessary
    if nam_args is not None:
        for data_region, randoms_region, randoms_weights_binned in zip(
            nam_args.data_regions, nam_args.randoms_regions, nam_args.randoms_weights_binned, strict=True
        ):
            alpha = jnp.where(data_region, data.weights, 0.0).sum() / jnp.where(randoms_region, randoms.weights, 0.0).sum()
            data_weights_binned = jnp.bincount(nam_args.data_pixels, weights=jnp.where(data_region, data.weights, 0.0), length=12 * nam_args.nside**2)
            nam_weights_binned = jnp.where(
                randoms_weights_binned == 0,
                0.0,  # don't care, will never be applied
                data_weights_binned / (alpha * randoms_weights_binned),
            )
            randoms = randoms.clone(weights=randoms.weights * jnp.where(randoms_region, nam_weights_binned[nam_args.randoms_pixels], 1.0))
    # Paint to mesh for P(k) computation and build FKP mesh
    if randoms_regions is not None:
        # global randoms renormalization per region
        global_alpha = data.weights.sum() / randoms.weights.sum()
        correction = jnp.ones_like(randoms.weights)
        for data_region, randoms_region in zip(data_regions, randoms_regions, strict=True):
            alpha = jnp.where(data_region, data.weights, 0.0).sum() / jnp.where(randoms_region, randoms.weights, 0.0).sum()
            # Multiply by one outside mask and alpha/global_alpha inside
            correction *= jnp.where(randoms_region, alpha / global_alpha, 1.0)
            # jnp.invert(randoms_region) * 1.0 + randoms_region * alpha / global_alpha
        randoms = randoms.clone(weights=randoms.weights * correction)
    fkp_field = fkp_field.clone(data=data, randoms=randoms)
    num_shotnoise = compute_fkp2_shotnoise(fkp_field, bin=binner)
    fkp_mesh = fkp_field.paint(resampler="tsc", interlacing=3, compensate=True, out="real")
    del fkp_field
    pk = compute_mesh2_spectrum(fkp_mesh, bin=binner, los={"local": "firstpoint"}.get(los, los))
    return pk.clone(
        norm=fkp_norm,
        num_shotnoise=num_shotnoise,
    )


def mock_surveys_FKP(
    # Gaussian mock generation
    theory: ObservableTree,
    seed: jnp.ndarray,
    los: Literal["local", "x", "y", "z"],
    unitary_amplitude: bool,
    # Data catalog and effects
    fkp_field: FKPField,
    combinations: list[tuple[RICArgsFKP | None, AMRArgsFKP | None, NAMArgsFKP | None]],
    # Final P(k) estimation
    binner: BinMesh2SpectrumPoles,
    fkp_norm: jnp.ndarray,
    # For region renormalization
    data_regions: list[jnp.ndarray] | None = None,
    randoms_regions: list[jnp.ndarray] | None = None,
) -> list[Mesh2SpectrumPoles]:
    """
    Get the power spectra from different mock surveys given one input theory, one seed and different sets of observational effects.

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
    fkp_field : FKPField
        FKP field contaning data and randoms information. The data shouldn't be clustered (*i.e.* the "data" should be randoms), but the FKP field serves to designate data and randoms amongst the original randoms.
    combinations : list[tuple[RICArgsFKP | None, AMRArgsFKP | None, NAMArgsFKP | None]]
        Tuples of precomputed arguments for RIC, AMR and NAM/AIC application. See :py:func:`prepare_RIC_FKP`, :py:func:`prepare_AMR_FKP` and :py:func:`prepare_NAM_FKP`. For example, ``combinations=[(None, None, None)]`` will only return geometry.
    binner : BinMesh2SpectrumPoles
        Binning operator to compute the output power spectrum.
    fkp_norm : jnp.ndarray
        Pre-computed power spectrum norm for the FKP field ``fkp_field``, disregarding any future changes in weights.
    data_regions : list[jnp.ndarray]
        List of masks for the renormalization regions in the data. Usually "N", "S", or split by galactic caps ("N", "SNGC", "SSGC"). May include DES for quasars.
    randoms_regions : list[jnp.ndarray]
        List of masks for the renormalization regions in the randoms. Usually "N", "S", or split by galactic caps ("N", "SNGC", "SSGC"). May include DES for quasars.

    Returns
    -------
    Mesh2SpectrumPoles
        One realization of the observations of the theory power spectrum.

    Notes
    -----
    NAM is a stronger kind of AMR. Applying both AMR and NAM or just NAM will result in the same spectrum, possibly with worse non-linear effects when applying both.
    """
    # Generate a gaussian mesh mock with exact required theory P(k)
    if len(combinations) == 0:
        raise ValueError("Must provide at least one combination!")
    mattrs = fkp_field.attrs
    mesh = generate_anisotropic_gaussian_mesh(
        mattrs,
        poles=theory,
        seed=seed,
        los=los,
        unitary_amplitude=unitary_amplitude,
    )
    # Paint it on the catalog -> catalog with geometry and ""clustering""
    data = fkp_field.data.clone(weights=fkp_field.data.weights * (1 + mesh.read(fkp_field.data.positions, resampler="cic", compensate=True)))
    randoms = fkp_field.randoms
    del mesh
    spectra = []
    for ric_args, amr_args, nam_args in combinations:
        ddata = data.clone()
        rrandoms = randoms.clone()
        # Apply RIC if necessary
        if ric_args is not None:
            for data_region, randoms_region, randoms_weights_binned in zip(
                ric_args.data_regions, ric_args.randoms_regions, ric_args.randoms_weights_binned, strict=True
            ):
                alpha = jnp.where(data_region, ddata.weights, 0.0).sum() / jnp.where(randoms_region, rrandoms.weights, 0.0).sum()
                data_weights_binned = jnp.bincount(ric_args.data_distances_digitized, weights=ddata.weights * data_region, length=ric_args.n_bins + 1)[1:]
                ric_weights_binned = jnp.where(
                    data_weights_binned == 0,
                    0.0,  # don't care, will never be applied
                    (alpha * randoms_weights_binned / data_weights_binned),
                )
                ddata = ddata.clone(weights=ddata.weights * jnp.where(data_region, ric_weights_binned[ric_args.data_distances_digitized - 1], 1.0))
        # Apply mode removal (ie linear template regression) if necessary
        # Randoms weights have not been changed yet
        if amr_args is not None:
            for data_region, factor, constant in zip(amr_args.data_regions, amr_args.factors, amr_args.constants, strict=True):
                data_weights_binned = bincount_2d(
                    amr_args.data_templates_digitized.T,
                    weights=ddata.weights * amr_args.mask_extremes_in_data * data_region,
                    length=amr_args.n_bins + 1,
                )[:, 1:, ...]
                p_opt = factor.dot(data_weights_binned.reshape((-1,))) - constant
                amr_weights = 1 / (1 + p_opt[0] + amr_args.data_templates_normalized.dot(p_opt[1:]))
                ddata = ddata.clone(weights=data.weights * jnp.where(data_region, amr_weights, 1.0))
        # Apply NAM if necessary, to randoms
        if nam_args is not None:
            for data_region, randoms_region, randoms_weights_binned in zip(
                nam_args.data_regions, nam_args.randoms_regions, nam_args.randoms_weights_binned, strict=True
            ):
                alpha = jnp.where(data_region, ddata.weights, 0.0).sum() / jnp.where(randoms_region, rrandoms.weights, 0.0).sum()
                data_weights_binned = jnp.bincount(nam_args.data_pixels, weights=jnp.where(data_region, ddata.weights, 0.0), length=12 * nam_args.nside**2)
                nam_weights_binned = jnp.where(
                    randoms_weights_binned == 0,
                    0.0,  # don't care, will never be applied
                    data_weights_binned / (alpha * randoms_weights_binned),
                )
                rrandoms = rrandoms.clone(weights=rrandoms.weights * jnp.where(randoms_region, nam_weights_binned[nam_args.randoms_pixels], 1.0))
        if randoms_regions is not None:
            # global randoms renormalization per region
            global_alpha = ddata.weights.sum() / rrandoms.weights.sum()
            correction = jnp.ones_like(rrandoms.weights)
            for data_region, randoms_region in zip(data_regions, randoms_regions, strict=True):
                alpha = jnp.where(data_region, ddata.weights, 0.0).sum() / jnp.where(randoms_region, rrandoms.weights, 0.0).sum()
                # Multiply by one outside mask and alpha/global_alpha inside
                correction *= jnp.where(randoms_region, alpha / global_alpha, 1.0)
                # jnp.invert(randoms_region) * 1.0 + randoms_region * alpha / global_alpha
            rrandoms = rrandoms.clone(weights=rrandoms.weights * correction)
        fkpfield = fkp_field.clone(data=ddata, randoms=rrandoms)
        num_shotnoise = compute_fkp2_shotnoise(fkpfield, bin=binner)
        fkp_mesh = fkpfield.paint(resampler="tsc", interlacing=3, compensate=True, out="real")
        del fkpfield
        pk = compute_mesh2_spectrum(fkp_mesh, bin=binner, los={"local": "firstpoint"}.get(los, los))
        spectra.append(pk.clone(norm=fkp_norm, num_shotnoise=num_shotnoise))
    return spectra


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
