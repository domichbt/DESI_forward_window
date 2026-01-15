"""Forward modeling of observational effects."""

from typing import Literal

import jax
import jax.numpy as jnp
from jaxpower import (
    BinMesh2SpectrumPoles,
    FKPField,
    RealMeshField,
    compute_fkp2_shotnoise,
    compute_mesh2_spectrum,
    generate_anisotropic_gaussian_mesh,
)
from lsstypes import Mesh2SpectrumPoles, ObservableTree

from .utils import bincount_2d, make_jax_dataclass, select_region

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

        jacobian = bincount_2d(
            templates_digitized_r.T,
            weights=masked_randoms_weights[:, None] * wt2,
            length=n_bins + 1,
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


@jax.jit(static_argnames=["n_bins", "apply_to"])
def apply_RIC(data_weights, randoms_weights, data_regions, randoms_regions, data_distances_digitized, randoms_distances_digitized, n_bins, apply_to="data"):
    # TODO LOOPS SHOULD BE VMAPS !!!! I'm using all the space anyways
    if apply_to == "data":
        ric_weights_global = jnp.zeros_like(data_weights)
        data_weights, randoms_weights = data_weights, randoms_weights
        data_regions, randoms_regions = data_regions, randoms_regions
        data_distances_digitized, randoms_distances_digitized = data_distances_digitized, randoms_distances_digitized

    elif apply_to == "randoms":
        ric_weights_global = jnp.zeros_like(randoms_weights)
        data_weights, randoms_weights = randoms_weights, data_weights
        data_regions, randoms_regions = randoms_regions, data_regions
        data_distances_digitized, randoms_distances_digitized = randoms_distances_digitized, data_distances_digitized
    else:
        raise ValueError("Can only apply to randoms or data!")

    for data_region, randoms_region in zip(data_regions, randoms_regions, strict=True):
        alpha = jnp.where(data_region, data_weights, 0.0).sum() / jnp.where(randoms_region, randoms_weights, 0.0).sum()
        data_weights_binned = jnp.bincount(data_distances_digitized, weights=data_weights * data_region, length=n_bins + 1)[1:]
        randoms_weights_binned = jnp.bincount(randoms_distances_digitized, weights=randoms_weights * randoms_region, length=n_bins + 1)[1:]

        ric_weights_binned = jnp.where(
            data_weights_binned == 0,
            0.0,  # don't care, will never be applied
            (alpha * randoms_weights_binned / data_weights_binned),
        )
        ric_weights_global += jnp.where(data_region, ric_weights_binned[data_distances_digitized - 1], 0.0)
    return ric_weights_global


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
