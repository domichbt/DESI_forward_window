"""Convenience and utility functions to call on in scripts."""

import os
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from astropy.table import Table
from jaxpower import MeshAttrs, ParticleField
from jaxpower.types import ObservableLeaf, ObservableTree

from .utils import get_clustering_positions_weights


def get_randoms(
    n_randoms: int,
    region: Literal["SGC", "NGC"],
    zrange: tuple[float, float] | None,
    tracer: Literal["QSO", "LRG", "BGS", "ELG", "ELG_LOPnotqso", "ELG_notqso"],
    weight_type: str | None = "default",
    return_redshift: bool = False,
    basedir: os.PathLike = Path("/dvs_ro/cfs/cdirs/desi/survey/catalogs/Y3/LSS/loa-v1/LSScats/v2/fNL/"),
    i_random_min: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get the DESI randoms' positions and weights needed to paint the geometry mesh. By default, Y3 randoms are loaded.

    Parameters
    ----------
    n_randoms : int
        Number of randoms files to load.
    region : Literal["SGC", "NGC"]
        Galactic cap region.
    zrange : tuple[float, float] | None
        Redshift range, can be set to ``None`` so that no redshift cuts are applied.
    tracer : Literal["QSO", "LRG", "BGS", "ELG", "ELG_LOPnotqso", "ELG_notqso"]
        What tracer to use for the file name.
    weight_type : str | None, optional
        Weight type to use, by default "default". Can set to None to use no weights.
    basedir : os.PathLike, optional
        Directory for the randoms, by default Path("/dvs_ro/cfs/cdirs/desi/survey/catalogs/Y3/LSS/loa-v1/LSScats/v2/fNL/"). Will look for "clustering" randoms.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Positions and weights for the required randoms.

    Notes
    -----
    One can easily get mesh attributes from :py:fun:`jaxpower.get_mesh_attrs` after this, and paint a ParticleField.
    """
    filenames = [basedir / f"{tracer}_{region}_{i}_clustering.ran.fits" for i in range(i_random_min, i_random_min + n_randoms)]
    positions, weights, z = get_clustering_positions_weights(filenames, kind="randoms", region=region, zrange=zrange, weight_type=weight_type)
    if weight_type is None:
        weights = [np.ones_like(weights[0])]
    if return_redshift:
        return positions, weights, z
    else:
        return positions, weights


def fiducial_planck_2018(edgesin: np.ndarray) -> ObservableTree:
    """
    Get the power spectrum of Planck 2018 flat LCDM cosmology, with required k and Kaiser RSD.

    Parameters
    ----------
    edgesin : np.ndarray
        Edges of the bins for the power spectrum.

    Returns
    -------
    ObservableTree
        Matter power spectrum with Kaiser RSD based on the Planck 2018 flat LCDM cosmology.
    """
    from cosmoprimo.fiducial import Planck2018FullFlatLCDM

    # Use a basic linear P(k) with RSD multipoles for the fiducial cosmology
    cosmo = Planck2018FullFlatLCDM(engine="eisenstein_hu")
    pk = cosmo.get_fourier().pk_interpolator().to_1d(z=1.0)

    edgesin = np.column_stack([edgesin[:-1], edgesin[1:]])

    kin = np.mean(edgesin, axis=-1)
    pkin = pk(kin)
    ells = (0, 2, 4)

    f, b = 0.8, 1.5
    pkb = b**2 * pkin
    beta = f / b
    poles = [(1.0 + 2.0 / 3.0 * beta + 1.0 / 5.0 * beta**2) * pkb, 0.9 * (4.0 / 3.0 * beta + 4.0 / 7.0 * beta**2) * pkb, 8.0 / 35 * beta**2 * pkb]
    poles = np.array(poles)

    theory = ObservableTree([ObservableLeaf(k=kin, k_edges=edgesin, value=pole, coords=["k"]) for pole in poles], ells=ells)
    return theory


def fiducial_DESI(edgesin: np.ndarray) -> ObservableTree:
    """
    Get the power spectrum of Planck 2018 flat LCDM cosmology, with required k and Kaiser RSD.

    Parameters
    ----------
    edgesin : np.ndarray
        Edges of the bins for the power spectrum.

    Returns
    -------
    ObservableTree
        Matter power spectrum with Kaiser RSD based on the Planck 2018 flat LCDM cosmology.
    """
    from cosmoprimo.fiducial import DESI

    # Use a basic linear P(k) with RSD multipoles for the fiducial cosmology
    cosmo = DESI(engine="eisenstein_hu")
    pk = cosmo.get_fourier().pk_interpolator().to_1d(z=1.0)

    edgesin = np.column_stack([edgesin[:-1], edgesin[1:]])

    kin = np.mean(edgesin, axis=-1)
    pkin = pk(kin)
    ells = (0, 2, 4)

    f, b = 0.8, 1.5
    pkb = b**2 * pkin
    beta = f / b
    poles = [(1.0 + 2.0 / 3.0 * beta + 1.0 / 5.0 * beta**2) * pkb, 0.9 * (4.0 / 3.0 * beta + 4.0 / 7.0 * beta**2) * pkb, 8.0 / 35 * beta**2 * pkb]
    poles = np.array(poles)

    theory = ObservableTree([ObservableLeaf(k=kin, k_edges=edgesin, value=pole, coords=["k"]) for pole in poles], ells=ells)
    return theory


def get_sysmap(map_path: str | os.PathLike, region: str, need_maps: list[str] | None = None, ebv_path: str | None = None) -> Table:
    """
    Return a table containing the usual systematics templates for DESI.

    Parameters
    ----------
    map_path : str | os.PathLike
        Path to the FITS file containing the maps for this photometric region.
    region : str
        Photometric region (``"N"`` or ``"S"``).
    need_maps : list[str] | None, optional
        Optional additional maps, by default ``None``. Can request ``"EBV_DIFF_MPF"``, ``"SKY_RES_G"``, ``"SKY_RES_R"``, ``"SKY_RES_Z"``.
    ebv_path : str | os.PathLike | None
        Optional custom path to the directory containing the EBV file, for :py:func:`LSS.common_tools.get_debv`.

    Returns
    -------
    Table
        Table containing the usual LSS systematics.

    Note
    ----
    This function requires the LSS package to be installed to run.
    """
    import LSS.common_tools as common

    need_maps = need_maps or []
    sysmaps = Table.read(map_path)

    ebv_path = ebv_path or "/global/cfs/cdirs/desicollab/users/rongpu/data/ebv/desi_stars_y3/v0.1/final_maps/lss/desi_ebv_lss_256.fits"
    debv = common.get_debv(ebv_path)
    cols = list(sysmaps.dtype.names)  # names of templates

    for col in cols:
        if "DEPTH" in col:
            bnd = col.split("_")[-1]
            sysmaps[col] *= 10 ** (-0.4 * common.ext_coeff[bnd] * sysmaps["EBV"])
    for ec in ["GR", "RZ"]:
        sysmaps["EBV_DIFF_" + ec] = debv["EBV_DIFF_" + ec]
    if "EBV_DIFF_MPF" in need_maps:
        sysmaps["EBV_DIFF_MPF"] = sysmaps["EBV"] - sysmaps["EBV_MPF_Mean_FW15"]

    if ("SKY_RES_G" in need_maps) or ("SKY_RES_R" in need_maps) or ("SKY_RES_Z" in need_maps):
        sky_g, sky_r, sky_z = common.get_skyres()
        if "SKY_RES_G" in need_maps:
            sysmaps["SKY_RES_G"] = sky_g[region]
        if "SKY_RES_R" in need_maps:
            sysmaps["SKY_RES_R"] = sky_r[region]
        if "SKY_RES_Z" in need_maps:
            sysmaps["SKY_RES_Z"] = sky_z[region]

    return sysmaps


def split_into_fields(
    positions: np.ndarray,
    weights: np.ndarray,
    extra: dict[str, np.ndarray] | None,
    data_size: int,
    split_seed: int,
    mattrs: MeshAttrs,
    backend: Literal["jax", "mpi"] = "jax",
    exchange: bool = True,
) -> tuple[ParticleField, ParticleField, np.ndarray]:
    """
    Split particles into two particle fields.

    Parameters
    ----------
    positions : np.ndarray
        Particle positions.
    weights : np.ndarray
        Particle weights.
    extra : dict[str, np.ndarray] | None
        Particle extra attributes, if any: redshift, template values, FKP weights...
    data_size : int
        Number of particles to use as "data".
    split_seed : int
        Seed to use for the split into "data" and "randoms".
    mattrs : MeshAttrs
        Mesh attributes for the fields.
    backend : Literal["jax", "mpi"], optional
        Backend to use for exchanging particles (if any), by default "jax"
    exchange : bool, optional
        Whether to exchange particles, by default True

    Returns
    -------
    tuple[ParticleField, ParticleField, np.ndarray]
        Data and randoms fields, as well as the mask for selecting the data.

    Notes
    -----
    Arrays in ``extra`` will be sharded/distributed like `positions` and `weights`, and are available as regular attributes of the :class:`jaxpower.ParticleField` instance. See :class:`jaxpower.ParticleField` for more details.
    """
    extra = extra or {}
    rng = np.random.default_rng(seed=split_seed)
    randoms_size = weights.size - data_size
    mask_is_data = rng.uniform(size=(randoms_size + data_size)) < (data_size / (data_size + randoms_size))
    data = ParticleField(
        positions[mask_is_data],
        weights=weights[mask_is_data],
        extra={k: v[mask_is_data] for k, v in extra.items()},
        attrs=mattrs,
        exchange=exchange,
        backend=backend,
    )
    randoms = ParticleField(
        positions[~mask_is_data],
        weights=weights[~mask_is_data],
        extra={k: v[~mask_is_data] for k, v in extra.items()},
        attrs=mattrs,
        exchange=exchange,
        backend=backend,
    )
    return data, randoms, mask_is_data


def mesh_union(mattrs1: MeshAttrs, mattrs2: MeshAttrs) -> tuple[jax.Array, jax.Array]:
    """
    Get the smallest axis-aligned cube that contains both input meshes.

    Parameters
    ----------
    mattrs1 : MeshAttrs
        First mesh attributes.
    mattrs2 : MeshAttrs
        Second mesh attributes.

    Returns
    -------
    tuple[jax.Array, jax.Array]
        Boxsize and boxcenter of the bounding cube for the input meshes.
    """
    c1 = mattrs1.boxcenter
    c2 = mattrs2.boxcenter
    boxcenter = (c1 + c2) / 2

    l1 = mattrs1.boxsize
    l2 = mattrs2.boxsize

    panes1 = jnp.stack([c1 - l1 / 2, c1 + l1 / 2])
    panes2 = jnp.stack([c2 - l2 / 2, c2 + l2 / 2])

    all_panes = jnp.concatenate([panes1, panes2], axis=0)
    boxsize = (jnp.max(all_panes, axis=0) - jnp.min(all_panes, axis=0)).max()

    return boxsize, boxcenter
