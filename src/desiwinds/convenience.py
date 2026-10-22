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


# Linear local-PNG + Kaiser basis vectors, see get_png_basis_vectors
PNG_KAISER_BASIS_NAMES = ["plin_ell0", "g_plin_ell0", "g2_plin_ell0", "plin_ell2", "g_plin_ell2", "plin_ell4"]
PNG_ALL_BASIS_NAMES = [f"{shape}_ell{ell}" for ell in (0, 2, 4) for shape in ("plin", "g_plin", "g2_plin")]


def get_png_basis_vectors(
    theory: ObservableTree,
    cosmology=None,
    responses: Literal["kaiser", "all"] = "kaiser",
    k_average: Literal["modes", "center"] = "modes",
    z: float = 0.0,
) -> tuple[np.ndarray, list[str]]:
    r"""
    Basis vectors of the linear local-PNG + Kaiser model at redshift ``z`` (by default 0), on the bins of ``theory``.

    The model is :math:`P(k, \mu) = (b_1 + b_\phi f_\mathrm{NL}\, g(k) + f \mu^2)^2 P_\mathrm{lin}(k)`, with
    :math:`g(k, z) = \sqrt{P_\phi(k) / P_{dd}(k, z)}` and :math:`P_\phi = \frac{9}{25} \frac{2 \pi^2}{k^3} P_\mathrm{prim}(k) / h^3`
    (the ``'prim'`` method of desilike's PNG theory). As :math:`P_{dd}(k, z) \simeq r(z)^2 P_0(k)` and :math:`g(k, z) \simeq g_0(k) / r(z)`,
    with :math:`r` from :py:func:`get_growth_ratio`, the shapes :math:`P_0`, :math:`g_0 P_0` and :math:`g_0^2 P_0` at ``z = 0`` give
    a window that does not depend on redshift, the powers of :math:`r(z_\mathrm{eff})` going into the coefficients (see
    :py:func:`get_png_kaiser_coefficients`). This is exact up to the scale dependence of the growth of ``delta_cb`` from massive
    neutrinos: with 0.06 eV, :math:`P_{dd}(k, z) / (r^2 P_0)` departs from 1 by up to 0.2% (z = 0.8), 0.4% (z = 1.5) and
    0.6% (z = 2.5) over :math:`0.01 < k < 0.16\ h/\mathrm{Mpc}`. Shapes at ``z = z_eff`` with ``growth_ratio = 1`` are exact.

    Parameters
    ----------
    theory : ObservableTree
        Fiducial theory of the window (*e.g.* the input of :py:func:`desiwinds.window.get_window_basis`), whose bins (the same
        for all multipoles, which must include 0, 2 and 4) define the vectors.
    cosmology : cosmoprimo.Cosmology, optional
        Cosmology of :math:`P_0`, :math:`P_\mathrm{prim}` and :math:`h`; use the same as the likelihood. By default
        ``cosmoprimo.fiducial.DESI(engine="camb")``.
    responses : Literal["kaiser", "all"], optional
        ``"kaiser"``: the 6 vectors of the Kaiser model (:py:data:`PNG_KAISER_BASIS_NAMES`); ``"all"``: the 9 responses
        :math:`\{P_0, g_0 P_0, g_0^2 P_0\} \times \{\ell = 0, 2, 4\}` (:py:data:`PNG_ALL_BASIS_NAMES`), for any model linear in them.
    k_average : Literal["modes", "center"], optional
        Value of each vector in a theory bin, which the Gaussian mocks paint as a constant: ``"modes"``, the average over the bin
        weighted by :math:`k^2` (the density of Fourier modes); ``"center"``, the value at the bin center. By default ``"modes"``.
    z : float, optional
        Redshift of the shapes :math:`P_{dd}(k, z)` and :math:`g(k, z)`. By default 0.

    Returns
    -------
    tuple[np.ndarray, list[str]]
        Basis vectors of shape ``(n_vectors, n_theory)`` (multipoles concatenated as in ``theory.value()``) and their names.

    Notes
    -----
    In bins starting at :math:`k = 0` (with the usual edges, the bin of the :math:`k = 0` mode only, which the integral constraints
    remove), the :math:`g_0 P_0` and :math:`g_0^2 P_0` shapes, which diverge at :math:`k \to 0`, are set to 0.
    """
    from .window import _common_k_bins

    if cosmology is None:
        from cosmoprimo.fiducial import DESI

        cosmology = DESI(engine="camb")
    if responses not in ("kaiser", "all"):
        raise ValueError(f"Unknown responses {responses!r}, expected 'kaiser' or 'all'.")
    ells = list(theory.ells)
    missing = [ell for ell in (0, 2, 4) if ell not in ells]
    if missing:
        raise ValueError(f"The theory has no multipoles {missing}.")
    _, k_edges = _common_k_bins(theory)
    low, high = k_edges[:, 0], k_edges[:, 1]

    # k at which to evaluate the shapes in each bin, and the weights of the average
    if k_average == "modes":
        nodes, weights = np.polynomial.legendre.leggauss(32)
        k = low[:, None] + (high - low)[:, None] * (nodes[None, :] + 1) / 2
        weights = weights[None, :] * k**2
        weights /= weights.sum(axis=-1, keepdims=True)
    elif k_average == "center":
        k, weights = ((low + high) / 2)[:, None], np.ones((low.size, 1))
    else:
        raise ValueError(f"Unknown k_average {k_average!r}, expected 'modes' or 'center'.")
    pk0, g0 = _png_shapes(cosmology, np.where(k > 0, k, 1.0), z=z)
    at_zero = low <= 0.0
    shapes = {
        "plin": np.sum(weights * pk0, axis=-1),
        "g_plin": np.where(at_zero, 0.0, np.sum(weights * g0 * pk0, axis=-1)),
        "g2_plin": np.where(at_zero, 0.0, np.sum(weights * g0**2 * pk0, axis=-1)),
    }

    names = PNG_KAISER_BASIS_NAMES if responses == "kaiser" else PNG_ALL_BASIS_NAMES
    vectors = np.zeros((len(names), len(ells), low.size))
    for ivector, name in enumerate(names):
        shape, ell = name.rsplit("_ell", 1)
        vectors[ivector, ells.index(int(ell))] = shapes[shape]
    return vectors.reshape(len(names), -1), list(names)


def get_png_kaiser_coefficients(b1, bphi_fnl, f, growth_ratio):
    r"""
    Coefficients of the ``"kaiser"`` basis vectors of :py:func:`get_png_basis_vectors`, in the same order.

    The prediction at the effective redshift :math:`z_\mathrm{eff}` is ``window.value() @ coefficients``, with the window computed
    on these vectors.

    Parameters
    ----------
    b1, f : float or jax.Array
        Linear bias and growth rate at :math:`z_\mathrm{eff}`.
    bphi_fnl : float or jax.Array
        :math:`b_\phi f_\mathrm{NL}` at :math:`z_\mathrm{eff}`.
    growth_ratio : float or jax.Array
        :math:`r(z_\mathrm{eff})` from :py:func:`get_growth_ratio`.

    Returns
    -------
    jax.Array
        The 6 coefficients.
    """
    r = growth_ratio
    return jnp.stack(
        [
            r**2 * (b1**2 + 2.0 / 3.0 * b1 * f + f**2 / 5.0),
            r * (2.0 * b1 + 2.0 / 3.0 * f) * bphi_fnl,
            bphi_fnl**2 * jnp.ones_like(r),
            r**2 * (4.0 / 3.0 * b1 * f + 4.0 / 7.0 * f**2),
            r * 4.0 / 3.0 * bphi_fnl * f,
            r**2 * 8.0 / 35.0 * f**2,
        ]
    )


def get_growth_ratio(cosmology, z: float, k: float = 1e-3) -> float:
    r"""Return :math:`r(z) = \sqrt{P_{dd}(k, z) / P_{dd}(k, 0)}` at large scales: the factor between the :math:`z = 0` PNG basis shapes and redshift :math:`z`."""
    pk = cosmology.get_fourier().pk_interpolator(of="delta_cb")
    return float(np.sqrt(pk.to_1d(z=z)(k) / pk.to_1d(z=0.0)(k)))


def _png_shapes(cosmology, k: np.ndarray, z: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    r"""Return :math:`P_{dd}(k, z)` (``delta_cb``) and :math:`g(k, z) = \sqrt{P_\phi / P_{dd}}`, ``k`` in h/Mpc."""
    pk0 = cosmology.get_fourier().pk_interpolator(of="delta_cb").to_1d(z=z)(k)
    pk_prim = cosmology.get_primordial(mode="scalar").pk_interpolator()(k)
    pphi = 9.0 / 25.0 * 2.0 * np.pi**2 / k**3 * pk_prim / cosmology.h**3
    return pk0, np.sqrt(pphi / pk0)


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

    if "ZCMB" in need_maps:
        sysmaps["ZCMB"] = common.mk_zcmbmap()

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
