"""Convenience and utility functions to call on in scripts."""

import os
from pathlib import Path
from typing import Literal

import numpy as np
from jaxpower.types import ObservableLeaf, ObservableTree

from .utils import get_clustering_positions_weights


def get_randoms(
    n_randoms: int,
    region: Literal["SGC", "NGC"],
    zrange: tuple[float, float] | None,
    tracer: Literal["QSO", "LRG", "BGS", "ELG", "ELG_LOPnotqso", "ELG_notqso"],
    weight_type: str | None = "default",
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
    positions, weights = get_clustering_positions_weights(filenames, kind="randoms", region=region, zrange=zrange, weight_type=weight_type)
    if weight_type is None:
        weights = [np.ones_like(weights[0])]
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
