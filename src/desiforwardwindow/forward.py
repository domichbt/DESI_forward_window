"""Forward modeling of observational effects."""

from typing import Literal

import jax
import jax.numpy as jnp
from jaxpower import (
    BinMesh2SpectrumPoles,
    ParticleField,
    RealMeshField,
    FKPField,
    compute_mesh2_spectrum,
    compute_normalization,
    generate_anisotropic_gaussian_mesh,
    compute_fkp2_shotnoise,
)
from lsstypes import Mesh2SpectrumPoles, ObservableTree, tree_map

from .utils import bincount_2d, make_jax_dataclass

AICArgs = make_jax_dataclass(
    class_name="AICArgs",
    dynamic_fields=[
        "data_templates_digitized",
        "mask_extremes_in_data",
        "data_templates_normalized",
        "factor",
        "constant",
    ],
    aux_fields=["n_bins"],
    types_fields={
        "data_templates_digitized": jnp.ndarray,
        "mask_extremes_in_data": jnp.ndarray,
        "data_templates_normalized": jnp.ndarray,
        "factor": jnp.ndarray,
        "constant": jnp.ndarray,
        "n_bins": int,
    },
)

RICArgs = make_jax_dataclass(
    class_name="RICArgs",
    dynamic_fields=[
        "data_distances_digitized",
        "randoms_distances_binned",
        "randoms_sum",
    ],
    aux_fields=["n_bins"],
    types_fields={
        "data_distances_digitized": jnp.ndarray,
        "randoms_distances_binned": jnp.ndarray,
        "randoms_sum": jnp.ndarray,
        "n_bins": int,
    },
)

RICArgsFKP = make_jax_dataclass(
    class_name="RICArgsFKP",
    dynamic_fields=[
        "data_distances_digitized",
        "data_to_remove",
        "randoms_weights_binned",
    ],
    aux_fields=["n_bins"],
    types_fields={
        "data_distances_digitized": jnp.ndarray,
        "data_to_remove": jnp.ndarray,
        "randoms_weights_binned": jnp.ndarray,
        "n_bins": int,
    },
)

NAMArgsFKP = make_jax_dataclass(
    class_name="AICArgsFKP",
    dynamic_fields=[
        "data_pixels",
        "randoms_pixels",
        "data_to_remove",
        "randoms_weights_binned",
    ],
    aux_fields=["nside"],
    types_fields={
        "data_pixels": jnp.ndarray,
        "randoms_pixels": jnp.ndarray,
        "data_to_remove": jnp.ndarray,
        "randoms_weights_binned": jnp.ndarray,
        "nside": int,
    },
)

NAMArgs = make_jax_dataclass(
    class_name="NAMArgs",
    dynamic_fields=[
        "data_hpx",
        "randoms_hpx_binned",
        "randoms_sum",
    ],
    aux_fields=["nside"],
    types_fields={
        "data_hpx": jnp.ndarray,
        "randoms_hpx_binned": jnp.ndarray,
        "randoms_sum": jnp.ndarray,
        "nside": int,
    },
)


def prepare_AIC(
    data_weights: jnp.ndarray,
    randoms_weights: jnp.ndarray,
    # AIC specific data
    template_values_data: jnp.ndarray,
    template_values_randoms: jnp.ndarray,
    # AIC specific parameters
    tail: float = 0.5,
    bin_margin: float = 1e-7,
    n_bins: int = 10,
) -> AICArgs:
    """
    Precompute all arguments necessary to :py:func:`get_AIC_weights` except for the ``data_weights``.

    Parameters
    ----------
    data_weights : jnp.ndarray
        Data weights.
    randoms_weights : jnp.ndarray
        Randoms weights.
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
    AICArgs
        Dataclass containing all information needed by ``get_AIC_weights`` except ``data.weights``.
    """
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

    # Binned weights and jacobian are used in the solution
    randoms_weights_binned = bincount_2d(
        templates_digitized_r.T,
        weights=randoms_weights * mask_extremes_r,  # set extremes weights to 0
        length=n_bins + 1,
    )[:, 1:, ...]

    masked_randoms_weights = randoms_weights * mask_extremes_r
    masked_templates_normalized = templates_normalized_r * mask_extremes_r[:, None]
    wt2 = jnp.concatenate([jnp.ones_like(randoms_weights)[..., None], masked_templates_normalized], axis=-1)

    jacobian = bincount_2d(
        templates_digitized_r.T,
        weights=masked_randoms_weights[:, None] * wt2,
        length=n_bins + 1,
    )[:, 1:, ...]

    normalization = (data_weights * mask_extremes_d).sum() / (randoms_weights * mask_extremes_r).sum()  # without the updated data weights: approximate

    # Ravel everything to take advantage of matrix operations
    jacobian = jacobian.reshape((-1, jacobian.shape[-1]))
    randoms_weights_binned = randoms_weights_binned.reshape((-1,))
    # Precompute a matrix that will be reused
    transpose_jw = normalization * jacobian.T * randoms_weights_binned  # matrix product with diag matrix is just numpy product with the diag vector
    factor = jnp.linalg.inv(transpose_jw.dot(jacobian)).dot(transpose_jw)
    constant = normalization * factor.dot(randoms_weights_binned)

    # pre-computed templates
    data_templates_digitized = templates_digitized_d
    data_templates_normalized = templates_normalized_d

    return AICArgs(
        data_templates_digitized=data_templates_digitized,
        mask_extremes_in_data=mask_extremes_d,
        data_templates_normalized=data_templates_normalized,
        factor=factor,
        constant=constant,
        n_bins=n_bins,
    )


def get_AIC_weights(
    data_weights: jnp.ndarray,
    fixed_args: AICArgs,
) -> jnp.ndarray:
    data_weights_binned = bincount_2d(
        fixed_args.data_templates_digitized.T,
        weights=data_weights * fixed_args.mask_extremes_in_data,
        length=fixed_args.n_bins + 1,
    )[:, 1:, ...]
    p_opt = fixed_args.factor.dot(data_weights_binned.reshape((-1,))) - fixed_args.constant
    return 1 / (1 + p_opt[0] + fixed_args.data_templates_normalized.dot(p_opt[1:]))


def prepare_RIC(
    data_positions: jnp.ndarray,
    randoms_positions: jnp.ndarray,
    randoms_weights: jnp.ndarray,
    boxcenter: jnp.ndarray,
    boxsize: jnp.ndarray,
    # RIC specific parameters
    n_bins: int,
):
    """
    Precompute all arguments necessary to :py:func:`get_RIC_weights` except for the ``data_weights``.

    Parameters
    ----------
    data_positions : jnp.ndarray
        2D array for data positions.
    randoms_positions : jnp.ndarray
        2D array for randoms positions.
    randoms_weights : jnp.ndarray
        1D array of the random's weights.
    boxcenter : jnp.ndarray
        Coordinates of the center of the box used for the power spectrum computation.
    boxsize : jnp.ndarray
        Sizes of the sides of the box used for the power spectrum computation.
    n_bins : int
        Number of distance bins to used.

    Returns
    -------
    RICArgs
        Dataclass containing all information needed by ``get_RIC_weights`` except ``data_weights``.
    """
    dmin = jnp.min(boxcenter - boxsize / 2.0)
    dmax = (1.0 + 1e-9) * jnp.sqrt(jnp.sum((boxcenter + boxsize / 2.0) ** 2))
    distance_edges = jnp.linspace(dmin, dmax, n_bins)
    randoms_distances = jnp.sqrt(jnp.power(randoms_positions, 2).sum(axis=-1))
    randoms_distances_digitized = jnp.digitize(randoms_distances, bins=distance_edges)  # could be made faster with jnp.floor
    randoms_distances_binned = jnp.bincount(randoms_distances_digitized, weights=randoms_weights, length=n_bins + 1)[1:]

    data_distances = jnp.sqrt(jnp.power(data_positions, 2).sum(axis=-1))
    data_distances_digitized = jnp.digitize(data_distances, bins=distance_edges)

    randoms_sum = randoms_weights.sum()

    return RICArgs(
        data_distances_digitized=data_distances_digitized,
        randoms_distances_binned=randoms_distances_binned,
        randoms_sum=randoms_sum,
        n_bins=n_bins,
    )


def prepare_RIC_FKP(
    fkp_field: FKPField,
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

    randoms_weights_binned = jnp.bincount(randoms_distances_digitized, weights=fkp_field.randoms.weights, length=n_bins + 1)[1:]

    data_distances_counts = jnp.bincount(data_distances_digitized, weights=None, length=n_bins + 1)[1:]
    randoms_distances_counts = jnp.bincount(randoms_distances_digitized, weights=None, length=n_bins + 1)[1:]
    data_to_remove = (data_distances_counts != 0) * (randoms_distances_counts == 0)

    return RICArgsFKP(
        data_distances_digitized=data_distances_digitized,
        randoms_weights_binned=randoms_weights_binned,
        data_to_remove=data_to_remove[data_distances_digitized - 1],
        n_bins=n_bins,
    )


def get_RIC_weights(
    data_weights,
    fixed_args: RICArgs,
) -> jnp.ndarray:
    data_distances_binned = jnp.bincount(fixed_args.data_distances_digitized, weights=data_weights, length=fixed_args.n_bins + 1)[1:]
    tmp = (
        data_weights.sum() / fixed_args.randoms_sum * jnp.where(data_distances_binned == 0, 1.0, (fixed_args.randoms_distances_binned / data_distances_binned))
    )
    return tmp[fixed_args.data_distances_digitized - 1]


def prepare_NAM(
    data_positions: jnp.ndarray,
    randoms_positions: jnp.ndarray,
    randoms_weights: jnp.ndarray,
    # RIC specific parameters
    nside: int,
) -> NAMArgs:
    """
    Precompute all arguments necessary to :py:func:`get_NAM_weights` except for the ``data_weights``.

    Parameters
    ----------
    data_positions : jnp.ndarray
        2D array for data positions.
    randoms_positions : jnp.ndarray
        2D array for randoms positions.
    randoms_weights : jnp.ndarray
        1D array of the random's weights.
    nside : int
        Resolution for the pixel binning (power of two).

    Returns
    -------
    NAMArgs
        Dataclass containing all information needed by ``get_RIC_weights`` except ``data_weights``.
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

    data_hpx = vec2pix(data_positions)
    randoms_hpx = vec2pix(randoms_positions)
    randoms_hpx_binned = jnp.bincount(randoms_hpx, weights=randoms_weights, length=12 * nside**2)

    randoms_sum = randoms_weights.sum()  # is that not just randoms_hpx_binned.sum() ?

    return NAMArgs(
        data_hpx=data_hpx,
        nside=nside,
        randoms_hpx_binned=randoms_hpx_binned,
        randoms_sum=randoms_sum,
    )


def prepare_NAM_FKP(
    fkp_field: FKPField,
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

    data_pixels = vec2pix(fkp_field.data.positions)
    randoms_pixels = vec2pix(fkp_field.randoms.positions)
    randoms_weights_binned = jnp.bincount(randoms_pixels, weights=fkp_field.randoms.weights, length=12 * nside**2)

    data_pixels_counts = jnp.bincount(data_pixels, weights=None, length=12 * nside**2)
    randoms_pixels_counts = jnp.bincount(randoms_pixels, weights=None, length=12 * nside**2)
    data_but_no_randoms = (data_pixels_counts != 0) * (randoms_pixels_counts == 0)

    return NAMArgsFKP(
        data_pixels=data_pixels,
        randoms_pixels=randoms_pixels,
        data_to_remove=data_but_no_randoms[data_pixels],
        randoms_weights_binned=randoms_weights_binned,
        nside=nside,
    )


def get_NAM_weights(
    data_weights,
    fixed_args: NAMArgs,
) -> jnp.ndarray:
    data_hpx_binned = jnp.bincount(fixed_args.data_hpx, weights=data_weights, length=12 * fixed_args.nside**2)
    return (
        data_weights.sum()
        / fixed_args.randoms_sum
        * jnp.where(
            data_hpx_binned == 0,
            1.0,
            (fixed_args.randoms_hpx_binned / data_hpx_binned),
        )[fixed_args.data_hpx]
    )


def mock_survey(
    # Gaussian mock generation
    theory: ObservableTree,
    seed: jnp.ndarray,
    los: Literal["local", "x", "y", "z"],
    unitary_amplitude: bool,
    # Data catalog and effects
    data: ParticleField,
    RIC_args: RICArgs | None,
    AIC_args: AICArgs | None,
    NAM_args: NAMArgs | None,
    # Final P(k) estimation
    binner: BinMesh2SpectrumPoles,
    randoms_mesh: RealMeshField,
    randoms_shotnoise: float,
    fkp_norm: jnp.ndarray,
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
    RIC_args : RICArgs | None
        Fixed precomputed arguments for :py:func:`get_RIC_weights`; see :py:func:`prepare_RIC`. Set to ``None`` to not apply RIC.
    AIC_args : AICArgs | None
        Fixed precomputed arguments for :py:func:`get_AIC_weights`; see :py:func:`prepare_AIC`. Set to ``None`` to not apply AIC.
    NAM_args : NAMArgs | None
        Fixed precomputed arguments for :py:func:`get_NAM_weights`; see :py:func:`prepare_NAM`. Set to ``None`` to not apply NAM.
    binner : BinMesh2SpectrumPoles
        Binning operator to compute the output power spectrum.
    randoms_mesh : RealMeshField
        Pre-painted randoms catalog.
    randoms_shotnoise : float
        Sum of the squared weights of the randoms.
    fkp_norm : jnp.ndarray
        Pre-computed power spectrum norm for the FKP field, disregarding any changes in weights.

    Returns
    -------
    Mesh2SpectrumPoles
        Realization of an observation of the theory power spectrum.

    Notes
    -----
    NAM is a stronger kind of AIC. Applying both AIC and NAM or just NAM will result in the same spectrum.
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
    del mesh
    # Apply RIC if necessary
    if RIC_args is not None:
        RIC_weights = get_RIC_weights(data_field.weights, RIC_args)
        data_field = data_field.clone(weights=data_field.weights * RIC_weights)
    # Apply AIC if necessary
    if AIC_args is not None:
        AIC_weights = get_AIC_weights(data_field.weights, AIC_args)
        data_field = data_field.clone(weights=data_field.weights * AIC_weights)
    # Apply AIC if necessary
    if NAM_args is not None:
        NAM_weights = get_NAM_weights(data_field.weights, NAM_args)
        data_field = data_field.clone(weights=data_field.weights * NAM_weights)
    # Paint to mesh for P(k) computation and build FKP mesh
    data_mesh = data_field.paint(resampler="tsc", interlacing=3, compensate=True, out="real")
    alpha = data_mesh.sum() / randoms_mesh.sum()
    fkp_mesh = data_mesh - alpha * randoms_mesh
    shotnoise = jnp.sum(data_field.weights**2) + alpha**2 * randoms_shotnoise
    pk = compute_mesh2_spectrum(fkp_mesh, bin=binner, los={"local": "firstpoint"}.get(los, los))
    return pk.clone(
        norm=fkp_norm,
        num_shotnoise=[shotnoise * (ell == 0) * jnp.ones_like(binner.edges[..., 0]) for ell in binner.ells],
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
    nam_args: NAMArgsFKP | None,
    # Final P(k) estimation
    binner: BinMesh2SpectrumPoles,
    fkp_norm: jnp.ndarray,
) -> Mesh2SpectrumPoles:
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
        alpha = data.weights.sum() / randoms.weights.sum()
        data_weights_binned = jnp.bincount(ric_args.data_distances_digitized, weights=data.weights, length=ric_args.n_bins + 1)[1:]
        ric_weights_binned = jnp.where(
            data_weights_binned == 0,
            0.0,  # don't care, will never be applied
            (alpha * ric_args.randoms_weights_binned / data_weights_binned),
        )
        data = data.clone(weights=data.weights * ric_weights_binned[ric_args.data_distances_digitized - 1])
    # Apply NAM if necessary
    if nam_args is not None:
        alpha = data.weights.sum() / randoms.weights.sum()
        data_weights_binned = jnp.bincount(nam_args.data_pixels, weights=data.weights, length=12 * nam_args.nside**2)
        nam_weights_binned = jnp.where(
            nam_args.randoms_weights_binned == 0,
            0.0,  # don't care, will never be applied
            data_weights_binned / (alpha * nam_args.randoms_weights_binned),
        )
        randoms = randoms.clone(weights=randoms.weights * nam_weights_binned[nam_args.randoms_pixels])
    # Paint to mesh for P(k) computation and build FKP mesh
    fkp_field = fkp_field.clone(data=data, randoms=randoms)
    num_shotnoise = compute_fkp2_shotnoise(fkp_field, bin=binner)
    fkp_mesh = fkp_field.paint(resampler="tsc", interlacing=3, compensate=True, out="real")
    del fkp_field
    pk = compute_mesh2_spectrum(fkp_mesh, bin=binner, los={"local": "firstpoint"}.get(los, los))
    return pk.clone(
        norm=fkp_norm,
        num_shotnoise=num_shotnoise,
    )


def mock_survey_diff(
    # Gaussian mock generation
    theory: ObservableTree,
    seed: jnp.ndarray,
    los: Literal["local", "x", "y", "z"],
    unitary_amplitude: bool,
    # Data catalog and effects
    data: ParticleField,
    RIC_args: RICArgs | None,
    AIC_args: AICArgs | None,
    NAM_args: NAMArgs | None,
    # Final P(k) estimation
    binner: BinMesh2SpectrumPoles,
    randoms_mesh: RealMeshField,
    randoms_shotnoise: float,
    fkp_norm: jnp.ndarray,
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
    RIC_args : RICArgs | None
        Fixed precomputed arguments for :py:func:`get_RIC_weights`; see :py:func:`prepare_RIC`. Set to ``None`` to not apply RIC.
    AIC_args : AICArgs | None
        Fixed precomputed arguments for :py:func:`get_AIC_weights`; see :py:func:`prepare_AIC`. Set to ``None`` to not apply AIC.
    NAM_args : NAMArgs | None
        Fixed precomputed arguments for :py:func:`get_NAM_weights`; see :py:func:`prepare_NAM`. Set to ``None`` to not apply NAM.
    binner : BinMesh2SpectrumPoles
        Binning operator to compute the output power spectrum.
    randoms_mesh : RealMeshField
        Pre-painted randoms catalog.
    randoms_shotnoise : float
        Sum of the squared weights of the randoms.
    fkp_norm : jnp.ndarray
        Pre-computed power spectrum norm for the FKP field, disregarding any changes in weights.

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
    alpha = data.weights.sum() / randoms_mesh.sum()
    mesh_geo = mesh * alpha * randoms_mesh
    pk_geo = compute_mesh2_spectrum(mesh_geo, bin=binner, los=los).clone(norm=fkp_norm)
    # Paint data mesh on the portion of randoms designated as "data" -> data catalog w/ geometry
    data_field = data.clone(weights=data.weights * (mesh.read(data.positions, resampler="cic", compensate=True) + 1))
    # Apply RIC if necessary
    if RIC_args is not None:
        RIC_weights = get_RIC_weights(data_field.weights, RIC_args)
        data_field = data_field.clone(weights=data_field.weights * RIC_weights)
    # Apply AIC if necessary
    if AIC_args is not None:
        AIC_weights = get_AIC_weights(data_field.weights, AIC_args)
        data_field = data_field.clone(weights=data_field.weights * AIC_weights)
    # Apply AIC if necessary
    if NAM_args is not None:
        NAM_weights = get_NAM_weights(data_field.weights, NAM_args)
        data_field = data_field.clone(weights=data_field.weights * NAM_weights)
    # Paint to mesh for P(k) computation and build FKP mesh
    data_mesh = data_field.paint(resampler="tsc", interlacing=3, compensate=True, out="real")
    alpha = data_mesh.sum() / randoms_mesh.sum()
    fkp_mesh = data_mesh - alpha * randoms_mesh
    shotnoise = jnp.sum(data_field.weights**2) + alpha**2 * randoms_shotnoise
    pk_IC = compute_mesh2_spectrum(fkp_mesh, bin=binner, los={"local": "firstpoint"}.get(los, los))
    pk_IC = pk_IC.clone(
        norm=fkp_norm,
        num_shotnoise=[shotnoise * (ell == 0) * jnp.ones_like(binner.edges[..., 0]) for ell in binner.ells],
    )
    # Return the difference of the spectra
    spectra = [pk_IC, pk_geo]
    return tree_map(lambda poles: poles[0].clone(value=poles[0].value() - poles[1].value()), spectra)


def mock_survey_mesh(
    # Gaussian mock generation
    theory: ObservableTree,
    seed: jnp.ndarray,
    los: Literal["local", "x", "y", "z"],
    unitary_amplitude: bool,
    # Final P(k) estimation
    binner: BinMesh2SpectrumPoles,
    norm: jnp.array,
    # selection
    selection: RealMeshField,
    ric: bool,
) -> Mesh2SpectrumPoles:
    """
    Apply mesh-based geometry forward modeling to a theoretical power spectrum.

    Returns
    -------
    Mesh2SpectrumPoles
        Realization of an observation of the theory power spectrum.
    """
    # Generate a gaussian mesh mock with exact required theory P(k)
    mattrs = selection.attrs
    mesh = (
        generate_anisotropic_gaussian_mesh(
            mattrs,
            poles=theory,
            seed=seed,
            los=los,
            unitary_amplitude=unitary_amplitude,
        )
        * selection
    )
    if ric:
        dmin = jnp.min(mattrs.boxcenter - mattrs.boxsize / 2.0)
        dmax = (1.0 + 1e-9) * jnp.sqrt(jnp.sum((mattrs.boxcenter + mattrs.boxsize / 2.0) ** 2))
        edges = jnp.linspace(dmin, dmax, 1000)
        rnorm = jnp.sqrt(sum(xx**2 for xx in mattrs.rcoords(sparse=True)))
        ibin = jnp.digitize(rnorm, edges, right=False)
        bw = jnp.bincount(ibin.ravel(), weights=mesh.ravel(), length=len(edges) + 1)
        b = jnp.bincount(ibin.ravel(), weights=selection.ravel(), length=len(edges) + 1)
        # Integral constraint
        bw = bw / jnp.where(b == 0.0, 1.0, b)  # (integral of W * delta) / (integral of W)
        mesh -= bw[ibin].reshape(selection.shape) * selection
    pk = compute_mesh2_spectrum(mesh, bin=binner, los={"local": "firstpoint"}.get(los, los))
    return pk.clone(norm=norm)
