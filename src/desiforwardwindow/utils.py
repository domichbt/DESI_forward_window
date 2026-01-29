"""Shared utility functions **for the internal libraries**."""

from dataclasses import dataclass, make_dataclass
from functools import partial
from typing import Any

import healpy as hp
import jax
import jax.numpy as jnp
import numpy as np
from jax import shard_map
from jax.sharding import PartitionSpec as P
from mockfactory import Catalog, sky_to_cartesian

NSIDE = 256
footprint = None


@jax.jit
def _my_bincount(idx, accumulator, weights):
    return accumulator.at[..., idx].add(weights)[..., :-1]


def bincount(x: jax.Array, weights: jax.Array, minlength: int = 0, length: int | None = None) -> jax.Array:
    """
    Perform a bincount over a 1D integer array (like :py:func:`jax.numpy.bincount`), allowing n-D weights.

    Parameters
    ----------
    x : jax.Array
        Array of integers of shape (n,).
    weights : jax.Array
        Corresponding weights of shape (..., n).
    minlength : int, optional
        A minimum number of bins for the output array, by default 0.
    length : int or None, optional
        Optional fixed length for the output array. Must be set for ``jax.jit`` compatibility.

    Returns
    -------
    jax.Array
        Array of shape (..., ``length``).
    """
    if length is None:
        length = max(x.max() + 1, minlength)
    accumulator = jnp.zeros((*weights.shape[:-1], length + 1), dtype=weights.dtype)
    return _my_bincount(x, accumulator, weights)


def bincount_2d(x: jnp.ndarray, weights: jnp.ndarray, minlength: int = 0, length: int | None = None) -> jnp.ndarray:
    """
    Perform a bincount over the second axis of an integer 2D array. Must set ``length`` for this function to be jittable.

    Parameters
    ----------
    x : jnp.ndarray
        A 2D array of shape (S, N).
    weights : jnp.ndarray
        Array of weights, shape (N) or (N, ...).
    minlength : int, optional
        A minimum number of bins for the output array, by default 0
    length : int or None, optional
        Optional fixed length for the output array. Must be set for this to be jittable.

    Returns
    -------
    jnp.ndarray
        The result of binning the input array along the second axis, *ie* an array of shape (D, x.max() + 1).
        If the weights are multi-dimentional, the returned array has shape (S, x.max() + 1, ...).
        If ``length`` was set, replace ``x.xmax() + 1`` with ``length``.
    """
    if length is None:
        length = max(x.max() + 1, minlength)
    return jax.lax.map(
        f=partial(
            _my_bincount,
            accumulator=jnp.zeros((length + 1, *weights.shape[1:]), dtype=weights.dtype),
            weights=weights,
        ),
        xs=x,
    )


def bincount_sorted(
    x: jax.Array, weights: jax.Array, rearrange: jax.Array | None, length: int | None = None, sharding_mesh: jax.sharding.Mesh | None = None
) -> jax.Array:
    """
    Perform a bincount over a **sorted** 1D integer array, allowing n-D weights.

    Parameters
    ----------
    x : jax.Array
        Array of integers of shape (n,). Must be sorted or sorted by ``rearrange``.
    weights : jax.Array
        Array of weights, shape (..., n).
    rearrange: jax.Array | None
        Array of integer indices indicating how to sort ``weights`` and ``x``. If set to ``None``, ``x`` is assumed to be ordered.
    length : int | None, optional
        Optional fixed length for the output array. Must be set for ``jax.jit`` compatibility.
    sharding_mesh : jax.sharding.Mesh | None, optional
        Sharding mesh to use for the ``shard_map``, by default None.

    Returns
    -------
    jax.Array
        Array of shape (..., ``length``).

    Notes
    -----
    If a sharding mesh is provided, ``x`` need only be sorted locally on each shard. Such functionality is provided by e.g. :py:func:`desiforwardwindow.utils.local_sort`. This avoids unnecessary communication between devices.
    """
    weights = jnp.moveaxis(weights, -1, 0)
    if rearrange is not None:
        x = x[rearrange]
        weights = weights[rearrange]
    if (sharding_mesh is None) or sharding_mesh.empty:
        return jax.ops.segment_sum(data=weights, segment_ids=x, num_segments=length, indices_are_sorted=True)
    else:

        @shard_map(
            in_specs=(P((*sharding_mesh.axis_names,), *([None] * (weights.ndim - 1))), P((*sharding_mesh.axis_names,)), None),
            out_specs=P(None),
            mesh=sharding_mesh,
            check_vma=False,  # TODO: remove when jax updates to 0.8.3
        )
        def _bincount_sorted(data, segment_ids, num_segments):
            return jax.lax.psum(
                jax.ops.segment_sum(data=data, segment_ids=segment_ids, num_segments=num_segments, indices_are_sorted=True),
                axis_name=(*sharding_mesh.axis_names,),
            )

        return _bincount_sorted(weights, x, length)


def local_argsort(arr: jax.Array, axis: int | None = None, sharding_mesh: jax.sharding.Mesh | None = None) -> jax.Array:
    """
    Sharding-local implementation of :py:func:`jnp.argsort`.

    Parameters
    ----------
    arr : jax.Array
        Array to sort.
    sharding_mesh : jax.sharding.Mesh | None, optional
        Sharding mesh to use for the ``shard_map``, by default None.

    Returns
    -------
    jax.Array
        Indices that sort each shard of the array.
    """
    if (sharding_mesh is None) or sharding_mesh.empty:
        return jnp.argsort(arr, axis=axis)
    else:

        @shard_map(in_specs=(P((*sharding_mesh.axis_names,)), None), out_specs=(P((*sharding_mesh.axis_names,))))
        def _local_argsort(arr, axis):
            return jnp.argsort(arr, axis=axis)

        return _local_argsort(arr, axis)


def apply_wntmp(ntile, ntmp_table, method="ntmp"):
    frac_missing_pw, frac_zero_prob = ntmp_table
    if method == "ntmp":
        toret = 1 - frac_missing_pw[ntile]
    elif method == "ntzp":
        toret = 1 - frac_zero_prob[ntile]
    else:
        raise NotImplementedError(f"unknown method {method}")
    # ref = apply_wntmp_bak(ntile, frac_missing_pw, frac_zero_prob, ntile_range=[0,15], randoms=True)[0]
    # assert np.allclose(toret, ref)
    return toret


def _format_bitweights(bitweights):
    if bitweights.ndim == 2:
        return list(bitweights.T)
    return [bitweights]


def load_footprint():
    global footprint
    from regressis import footprint

    footprint = footprint.DR9Footprint(NSIDE, mask_lmc=False, clear_south=True, mask_around_des=False, cut_desi=False)


def select_region(ra, dec, region=None):
    # print('select', region)

    def _ang2pix(ra, dec):
        return hp.ang2pix(NSIDE, ra, dec, nest=True, lonlat=True)

    def ang2pix(ra, dec):
        return jax.pure_callback(_ang2pix, jax.ShapeDtypeStruct(ra.shape[:1], jnp.int64), ra, dec)

    from jax import shard_map
    from jax.sharding import PartitionSpec as P
    from jaxpower.mesh import get_sharding_mesh

    sharding_mesh = get_sharding_mesh()

    if sharding_mesh.axis_names:
        ang2pix = shard_map(ang2pix, mesh=sharding_mesh, in_specs=P(sharding_mesh.axis_names), out_specs=P(sharding_mesh.axis_names))

    if region in [None, "ALL", "GCcomb"]:
        if isinstance(ra, jax.Array):
            return jnp.ones_like(ra, dtype=bool)
        else:
            return np.ones_line(ra, dtype=bool)
    mask_ngc = ra > 100 - dec
    mask_ngc &= ra < 280 + dec
    mask_n = mask_ngc & (dec > 32.375)
    mask_s = (~mask_n) & (dec > -25.0)
    if region == "NGC":
        return mask_ngc
    if region == "SGC":
        return ~mask_ngc
    if region == "N":
        return mask_n
    if region == "S":
        return mask_s
    if region == "SNGC":
        return mask_ngc & mask_s
    if region == "SSGC":
        return (~mask_ngc) & mask_s
    if footprint is None:
        load_footprint()
    north, south, des = footprint.get_imaging_surveys()
    des = jnp.array(des)
    mask_des = des[ang2pix(ra, dec)]
    if region == "DES":
        return mask_des
    if region == "SnoDES":
        return mask_s & (~mask_des)
    if region == "SSGCnoDES":
        return (~mask_ngc) & mask_s & (~mask_des)
    raise ValueError("unknown region %s", region)


def get_clustering_rdzw(
    *fns,
    kind=None,
    zrange=None,
    region=None,
    tracer=None,
    weight_type="default",
    ntmp=None,
    **kwargs,
):
    from mpi4py import MPI

    mpicomm = MPI.COMM_WORLD

    weight_type = weight_type or ""

    catalogs = [None] * len(fns)
    for ifn, fn in enumerate(fns):
        irank = ifn % mpicomm.size
        catalogs[ifn] = (irank, None)
        if mpicomm.rank == irank:  # Faster to read catalogs from one rank
            catalog = Catalog.read(fn, mpicomm=MPI.COMM_SELF)
            catalog.get(catalog.columns())  # Faster to read all columns at once
            columns = [
                "RA",
                "DEC",
                "Z",
                "WEIGHT",
                "WEIGHT_SYS",
                "WEIGHT_ZFAIL",
                "WEIGHT_COMP",
                "WEIGHT_FKP",
                "BITWEIGHTS",
                "FRAC_TLOBS_TILES",
                "NTILE",
            ]
            columns = [col for col in columns if col in catalog.columns()]
            catalog = catalog[columns]
            if zrange is not None:
                mask = (catalog["Z"] >= zrange[0]) & (catalog["Z"] <= zrange[1])
                catalog = catalog[mask]
            if "bitwise" in weight_type:
                mask = catalog["FRAC_TLOBS_TILES"] != 0
                catalog = catalog[mask]
            if region is not None:
                mask = select_region(catalog["RA"], catalog["DEC"], region)
                catalog = catalog[mask]
            catalogs[ifn] = (irank, catalog)

    rdzw = []
    for irank, catalog in catalogs:
        if mpicomm.size > 1:
            catalog = Catalog.scatter(catalog, mpicomm=mpicomm, mpiroot=irank)
        individual_weight = catalog["WEIGHT"]
        bitwise_weights = []
        if "bitwise" in weight_type:
            if kind == "data":
                individual_weight = catalog["WEIGHT"] / catalog["WEIGHT_COMP"]
                bitwise_weights = _format_bitweights(catalog["BITWEIGHTS"])
            elif kind == "randoms" and ntmp is not None:
                individual_weight = catalog["WEIGHT"] * apply_wntmp(catalog["NTILE"], ntmp)
        if "FKP" in weight_type.upper():
            individual_weight *= catalog["WEIGHT_FKP"]
        rdzw.append([catalog["RA"], catalog["DEC"], catalog["Z"], individual_weight] + bitwise_weights)
    rdzw = [np.concatenate([arrays[i] for arrays in rdzw], axis=0) for i in range(len(rdzw[0]))]
    for i in range(4):
        rdzw[i] = rdzw[i].astype("f8")
    return rdzw[:3], rdzw[3:]


def get_clustering_positions_weights(*fns, **kwargs):
    from cosmoprimo.fiducial import TabulatedDESI

    fiducial = TabulatedDESI()  # faster than DESI/class (which takes ~30 s for 10 random catalogs)
    [ra, dec, z], weights = get_clustering_rdzw(*fns, **kwargs)
    dist = fiducial.comoving_radial_distance(z)
    positions = sky_to_cartesian(dist, ra, dec)
    return positions, weights, z


def make_jax_dataclass(class_name: str, dynamic_fields: list[str], aux_fields: list[str], types_fields: dict[str, type] | None = None) -> dataclass:
    """
    Create a JAX-compatible dataclass with tree_flatten / tree_unflatten.

    Parameters
    ----------
    class_name : str
        Name of the class.
    dynamic_fields : list[str]
        Fields included in the pytree leaves
    aux_fields : list[str]
        Fields stored in the pytree auxiliary data
    types_fields : dict[str, type], optional
        Optional types for the fields. Missing types will be set to ``typing.Any``.

    Returns
    -------
    dataclass
        A dataclass with required attributes and compatible with tree_flatten / tree_unflatten.
    """
    # Create dataclass fields with type=Any
    fields = [(name, types_fields.get(name, Any)) for name in (dynamic_fields + aux_fields)]

    # Create the base dataclass type
    cls = make_dataclass(class_name, fields)

    def tree_flatten(self):
        # leaves: only dynamic fields
        leaves = [getattr(self, f) for f in dynamic_fields]
        # aux: all auxiliary fields captured as a dict
        aux = {f: getattr(self, f) for f in aux_fields}
        return leaves, aux

    @classmethod
    def tree_unflatten(cls, aux, leaves):
        # reconstruct all fields in correct order
        kwargs = {}
        # dynamic fields reconstructed from `leaves`
        for f, v in zip(dynamic_fields, leaves, strict=True):
            kwargs[f] = v
        # aux fields restored directly from aux dict
        for f in aux_fields:
            kwargs[f] = aux[f]
        return cls(**kwargs)

    # attach methods
    cls.tree_flatten = tree_flatten
    cls.tree_unflatten = tree_unflatten

    return jax.tree_util.register_pytree_node_class(cls)
