"""Shared utility functions **for the internal libraries**."""

from collections.abc import Sequence
from dataclasses import dataclass, make_dataclass
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
    If a sharding mesh is provided, ``x`` need only be sorted locally on each shard. Such functionality is provided by e.g. :py:func:`desiwinds.utils.local_sort`. This avoids unnecessary communication between devices.
    """
    if (sharding_mesh is None) or sharding_mesh.empty:
        if rearrange is not None:
            x = x[rearrange]
            weights = jnp.moveaxis(weights, -1, 0)[rearrange]
        return jax.ops.segment_sum(data=weights, segment_ids=x, num_segments=length, indices_are_sorted=True)
    else:

        @shard_map(
            in_specs=(P(*([None] * (weights.ndim - 1)), (*sharding_mesh.axis_names,)), P((*sharding_mesh.axis_names,)), None, P((*sharding_mesh.axis_names,))),
            out_specs=P(None),
            mesh=sharding_mesh,
            check_vma=False,  # TODO: remove when jax updates to 0.8.3
        )
        def _bincount_sorted(data, segment_ids, num_segments, rearrange):
            if rearrange is not None:
                segment_ids = segment_ids[rearrange]
                data = jnp.moveaxis(data, -1, 0)[rearrange]
            return jax.lax.psum(
                jax.ops.segment_sum(data=data, segment_ids=segment_ids, num_segments=num_segments, indices_are_sorted=True),
                axis_name=(*sharding_mesh.axis_names,),
            )

        return _bincount_sorted(weights, x, length, rearrange)


def local_argsort(arr: jax.Array, axis: int | None = None, sharding_mesh: jax.sharding.Mesh | None = None) -> jax.Array:
    """
    Sharding-local implementation of :py:func:`jnp.argsort`.

    Parameters
    ----------
    arr : jax.Array
        Array to sort.
    axis: int | None, optional
        Axis along which to sort the array, by default None (0). Will also be considered as the sharded axis.
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
        if axis is not None:
            spec = P(*((None,) * axis + (sharding_mesh.axis_names,) + (None,) * (arr.ndim - 1 - axis)))
        else:
            spec = P(sharding_mesh.axis_names)

        @shard_map(in_specs=(spec, None), out_specs=spec, mesh=sharding_mesh)
        def _local_argsort(arr, axis):
            return jnp.argsort(arr, axis=axis)

        return _local_argsort(arr, axis)


def local_concatenate(arrays: Sequence[jax.Array], axis: int | None = None, sharding_mesh: jax.sharding.Mesh | None = None) -> jax.Array:
    """
    Sharding-local implementation of :py:func:`jnp.concatenate`.

    Parameters
    ----------
    arrays: Sequence[jax.Array]
        Arrays to concatenate, all sharded the **same** way, along the **first** axis.
    axis : int | None, optional
        Axis along which to concatenate the arrays, by default None (0). Will also be considered as the sharded axis.
    sharding_mesh : jax.sharding.Mesh | None, optional
        Sharding mesh to use for the ``shard_map``, by default None.

    Returns
    -------
    jax.Array
        Locally concatenated array.
    """
    if (sharding_mesh is None) or sharding_mesh.empty:
        return jnp.concatenate(arrays=arrays, axis=axis)
    else:
        if axis is not None:
            spec = [P(*((None,) * axis + (sharding_mesh.axis_names,) + (None,) * (array.ndim - 1 - axis))) for array in arrays]
        else:
            spec = [P(sharding_mesh.axis_names)] * len(arrays)

        @shard_map(in_specs=(spec, None), out_specs=spec[0], mesh=sharding_mesh)
        def _local_concatenate(arrays, axis):
            return jnp.concatenate(arrays, axis=axis)

        return _local_concatenate(arrays, axis)


def local_split(
    ary: jax.Array, indices_or_sections: jax.Array | Sequence[int], axis: int | None = None, sharding_mesh: jax.sharding.Mesh | None = None
) -> jax.Array:
    """
    Sharding-local implementation of :py:func:`jnp.split`.

    Parameters
    ----------
    ary: jax.Array
        Array to split.
    indices_or_sections : Sequence[int]
        Indices or sections to split the array. These indices should be global; the function will handle local splitting. Indices must all be divisible by the number of shards along ``axis``.
    axis : int | None, optional
        Axis along which to split the array, by default None (0). This axis will also be considered as the sharded axis.
    sharding_mesh : jax.sharding.Mesh | None, optional
        Sharding mesh to use for the ``shard_map``, by default None.

    Returns
    -------
    jax.Array
        Locally split array.

    Notes
    -----
    This function will not raise an error if the indices are not compatible with local splitting; it is the user's responsibility to ensure that indices are divisible by the number of shards along ``axis``.
    """
    if (sharding_mesh is None) or sharding_mesh.empty:
        return jnp.split(ary=ary, indices_or_sections=indices_or_sections, axis=axis)
    else:
        if axis is not None:
            spec = P(*((None,) * axis + (sharding_mesh.axis_names,) + (None,) * (ary.ndim - 1 - axis)))
        else:
            spec = P(sharding_mesh.axis_names)

        global_size = ary.shape[axis or 0]

        @shard_map(in_specs=(spec, None, None), out_specs=spec, mesh=sharding_mesh)
        def _local_split(ary, indices_or_sections, axis):
            local_size = ary.shape[axis]
            n_shards = global_size // local_size
            return jnp.split(ary, [idx // n_shards for idx in indices_or_sections], axis=axis)

        return _local_split(ary, indices_or_sections, axis)


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


def __ang2pix(ra, dec):
    return hp.ang2pix(NSIDE, ra, dec, nest=True, lonlat=True)


def _ang2pix(ra, dec):
    return jax.pure_callback(__ang2pix, jax.ShapeDtypeStruct(ra.shape[:1], jnp.int64), ra, dec)


def select_region(ra, dec, region=None, sharding_mesh=None):
    """Supported regions: ALL, GCcomb, NGC, SGC, N, S, SNGC, SSGC, DES, SnoDES, SSGCnoDES."""
    if (sharding_mesh is None) or sharding_mesh.empty:
        ang2pix = _ang2pix
    else:
        ang2pix = shard_map(_ang2pix, mesh=sharding_mesh, in_specs=P(sharding_mesh.axis_names), out_specs=P(sharding_mesh.axis_names))

    if region in [None, "ALL", "GCcomb"]:
        if isinstance(ra, jax.Array):
            return jnp.ones_like(ra, dtype=bool)
        else:
            return np.ones_like(ra, dtype=bool)
    mask_ngc = ra > 100 - dec
    mask_ngc &= ra < 280 + dec
    mask_n = mask_ngc & (dec > 32.375)
    mask_s = (~mask_n) & (dec > -25.0)
    # Force synchronization to avoid hangs with JAX arrays
    if isinstance(mask_s, jax.Array):
        mask_s.block_until_ready()
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
        fkp_weights = []
        if "bitwise" in weight_type:
            if kind == "data":
                individual_weight = catalog["WEIGHT"] / catalog["WEIGHT_COMP"]
                bitwise_weights = _format_bitweights(catalog["BITWEIGHTS"])
            elif kind == "randoms" and ntmp is not None:
                individual_weight = catalog["WEIGHT"] * apply_wntmp(catalog["NTILE"], ntmp)
        if "FKP" in weight_type.upper():
            fkp_weights = [catalog["WEIGHT_FKP"]]
        rdzw.append([catalog["RA"], catalog["DEC"], catalog["Z"], individual_weight] + fkp_weights + bitwise_weights)
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


# @jax.jit(static_argnames=["tail", "n_bins", "bin_margin"])
def prepare_templates(
    data_templates: jax.Array,
    randoms_templates: jax.Array,
    data_regions: list[jax.Array],
    randoms_regions: list[jax.Array],
    randoms_is_real: jax.Array,
    tail: float,
    n_bins: int,
    bin_margin: float,
    sharding_mesh: jax.sharding.Mesh | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Prepare normalized and digitized photometry templates for AMR. Inputs are treated region by region; extremes are set to bin 0.

    Parameters
    ----------
    data_templates : jax.Array
        Input data template values, shape (n_d, n_sys).
    randoms_templates : jax.Array
        Input randoms template values, shape (n_r, n_sys).
    data_regions : list[jax.Array]
        List of masks for the data, for the different regions.
    randoms_regions : list[jax.Array]
        Lists of masks for the randoms, for the different regions.
    randoms_is_real : jax.Array
        Mask indicating real (non-padding) particles for the extremal values computation.
    tail : float
        Percentage for the extremal value cutoffs.
    n_bins : int
        Number of bins for the regression.
    bin_margin : float
        Extra padding for the edges of the bins.
    sharding_mesh : jax.sharding.Mesh | None, optional
        Sharding mesh to use for the ``shard_map``, by default None.

    Returns
    -------
    data_templates_normalized, data_templates_digitized, rand_templates_normalized, rand_templates_digitized: tuple[jax.Array, jax.Array, jax.Array, jax.Array]
        Normalized and digitized template values for randoms and data. Shapes (n_sys + 1, n_d) and (n_sys + 1, n_r).

    Notes
    -----
    This function correctly carries along sharding if input is sharded along the catalog (i.e. n_d/n_r size) axis and sharding_mesh is set accordingly.
    """
    args = data_templates, randoms_templates, data_regions, randoms_regions, randoms_is_real, tail, n_bins, bin_margin
    if (sharding_mesh is None) or sharding_mesh.empty:
        return _prepare_templates(*args)
    else:
        ax = sharding_mesh.axis_names
        return shard_map(
            _prepare_templates,
            in_specs=(P(ax, None), P(ax, None), [P(ax)] * len(data_regions), [P(ax)] * len(randoms_regions), P(ax), None, None, None),
            out_specs=(P(None, ax),) * 4,
            mesh=sharding_mesh,
        )(*args)


def _prepare_templates(
    data_templates: jax.Array,
    randoms_templates: jax.Array,
    data_regions: list[jax.Array],
    randoms_regions: list[jax.Array],
    randoms_is_real: jax.Array,
    tail: float,
    n_bins: int,
    bin_margin: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    n_sys = data_templates.shape[1]
    n_dat = data_templates.shape[0]
    n_ran = randoms_templates.shape[0]
    # initialize large arrays to fill
    # careful: these are transposed compared to the input shapes
    data_templates_normalized = jnp.zeros(shape=(n_sys + 1, n_dat), dtype=float).at[0].set(1.0)
    rand_templates_normalized = jnp.zeros(shape=(n_sys + 1, n_ran), dtype=float).at[0].set(1.0)
    data_templates_digitized = jnp.zeros(shape=(n_sys + 1, n_dat), dtype=int).at[0].set(n_bins - 1)
    rand_templates_digitized = jnp.zeros(shape=(n_sys + 1, n_ran), dtype=int).at[0].set(n_bins - 1)

    for ireg, (data_sel, rand_sel) in enumerate(zip(data_regions, randoms_regions, strict=True)):
        nanfakes = jnp.where((randoms_is_real & rand_sel)[:, None], randoms_templates, jnp.nan)
        lower_tails = jnp.nanpercentile(nanfakes, tail / 2, axis=0, method="higher")
        upper_tails = jnp.nanpercentile(nanfakes, 100 - tail / 2, axis=0, method="lower")
        bin_edges = jnp.linspace(start=lower_tails - bin_margin, stop=upper_tails + bin_margin, num=n_bins + 1)

        non_extreme_data = jnp.all((data_templates >= lower_tails) & (data_templates <= upper_tails), axis=1)
        non_extreme_rand = jnp.all((randoms_templates >= lower_tails) & (randoms_templates <= upper_tails), axis=1)

        data_templates_normalized = data_templates_normalized.at[1:, data_sel].set(
            ((data_templates[data_sel] - bin_edges[0, :]) / (bin_edges[-1, :] - bin_edges[0, :])).T
        )
        rand_templates_normalized = rand_templates_normalized.at[1:, rand_sel].set(
            ((randoms_templates[rand_sel] - bin_edges[0, :]) / (bin_edges[-1, :] - bin_edges[0, :])).T
        )

        data_templates_digitized = data_templates_digitized.at[1:, data_sel].set(
            jnp.where(
                non_extreme_data[None, data_sel],
                jnp.clip(
                    jnp.floor(data_templates_normalized[1:, data_sel] * n_bins).astype(int)
                    - (data_templates_normalized[1:, data_sel] == bin_edges[-1, :, None])
                    + 1,
                    min=0,
                    max=n_bins + 1,
                )
                + ireg * (n_bins + 2),
                0,
            )
        ) + jnp.where(data_sel, ireg * (n_bins + 2), 0)  # i think this broadcasts ok
        rand_templates_digitized = rand_templates_digitized.at[1:, rand_sel].set(
            jnp.where(
                non_extreme_rand[None, rand_sel],
                jnp.clip(
                    jnp.floor(rand_templates_normalized[1:, rand_sel] * n_bins).astype(int)
                    - (rand_templates_normalized[1:, rand_sel] == bin_edges[-1, :, None])
                    + 1,
                    min=0,
                    max=n_bins + 1,
                ),
                0,
            )
        ) + jnp.where(rand_sel & non_extreme_rand, ireg * (n_bins + 2), 0)  # i think this broadcasts ok

    # In the end, normalized = normalized templates (per region) with first row of ones
    # digitized = digitized (per region), with first row of nbins-1, offset by (n_bins + 2) depending on the region, with extreme values set to bin 0 all the time
    return data_templates_normalized, data_templates_digitized, rand_templates_normalized, rand_templates_digitized
