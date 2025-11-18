"""Convenience and utility functions to call on in scripts."""

import os
from pathlib import Path
from typing import Literal

import numpy as np

from .utils import get_clustering_positions_weights


def get_randoms(
    n_randoms: int,
    region: Literal["SGC", "NGC"],
    zrange: tuple[float, float] | None,
    tracer: Literal["QSO", "LRG", "BGS", "ELG", "ELG_LOPnotqso", "ELG_notqso"],
    weight_type: str | None = "default",
    basedir: os.PathLike = Path("/dvs_ro/cfs/cdirs/desi/survey/catalogs/Y3/LSS/loa-v1/LSScats/v2/fNL/"),
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
    filenames = [basedir / f"{tracer}_{region}_{i}_clustering.ran.fits" for i in range(n_randoms)]
    positions, weights = get_clustering_positions_weights(filenames, kind="randoms", region=region, zrange=zrange, weight_type=weight_type)
    if weight_type is None:
        weights = [np.ones_like(weights[0])]
    return positions, weights
