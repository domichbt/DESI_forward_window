import jax
import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"
jax.distributed.initialize()
import logging
from functools import partial

import jax.numpy as jnp
import matplotlib.pyplot as plt
from jaxpower import BinMesh2SpectrumPoles, FKPField, ParticleField, compute_fkp2_normalization, create_sharding_mesh, get_mesh_attrs, setup_logging
from jaxpower.mesh import _get_extent, make_array_from_process_local_data
from lsstypes import mean
from tqdm import tqdm

from desiforwardwindow.convenience import fiducial_planck_2018, get_randoms
from desiforwardwindow.forward import get_AIC_forward_model, get_NAM_forward_model, get_RIC_forward_model, mock_survey

logger = logging.getLogger("sharding")

setup_logging("info")
logger.info("Log test")

LOS = "local"
UNITARY_AMPLITUDE = True
BATCH_SIZE = 1
nreal = 2

regression_maps = [
    "STARDENS",
    "PSFSIZE_G",
    "PSFSIZE_R",
    # "PSFSIZE_Z",
    # "GALDEPTH_G",
    # "GALDEPTH_R",
    # "GALDEPTH_Z",
    # "EBV_DIFF_GR",
    # "EBV_DIFF_RZ",
    # "HI",
]  # will need to create some of those, not stored

n_bins_AIC = 10  # template bins for the regression
n_bins_RIC = 1000  # distance bins for the shells in RIC # low for quick computation
nside = 64

with create_sharding_mesh() as sharding_mesh:
    tracer = "LRG"
    positions, stored_weights = get_randoms(n_randoms=1, region="SGC", zrange=(0.4, 1.1), tracer=tracer, weight_type="default")
    stored_weights = stored_weights[0][: int(1e6)]
    positions = positions[: int(1e6), :]
    boxsize = jnp.array([8000.0] * 3)
    cellsize = 50.0
    # data_size = 1677566  # amount of data in the corresponding data catalog
    data_size = int(1e5)  # easy amount for debugging
    data_size = positions.shape[0] // 2
    randoms_size = positions.shape[0] - data_size

    logger.info("Data size: %s", data_size)
    logger.info("Randoms size: %s", randoms_size)

    logger.info("Loaded data")

    import fitsio
    import healpy as hp
    import LSS.common_tools as common
    from numpy.lib.recfunctions import append_fields, structured_to_unstructured

    debv = common.get_debv()

    sys_tab = fitsio.read(f"/dvs_ro/cfs/cdirs/desi/survey/catalogs/Y3/LSS/loa-v1/LSScats/v2/hpmaps/{tracer}_mapprops_healpix_nested_nside256_S.fits")

    for col in regression_maps:
        if "DEPTH" in col:
            bnd = col.split("_")[-1]
            sys_tab[col] *= 10 ** (-0.4 * common.ext_coeff[bnd] * sys_tab["EBV"])
    sys_tab = append_fields(sys_tab, names=["EBV_DIFF_" + ec for ec in ["GR", "RZ"]], data=[debv["EBV_DIFF_" + ec] for ec in ["GR", "RZ"]], usemask=False)

    hpx_rand = hp.vec2pix(nside=256, x=positions[:, 0], y=positions[:, 1], z=positions[:, 2], nest=True)
    template_values = structured_to_unstructured(sys_tab[regression_maps][hpx_rand]).astype(float)

    del sys_tab, hpx_rand, debv

    seed = jax.random.key(42)
    data_indices_in_randoms = jax.random.choice(key=seed, a=stored_weights.size, shape=[data_size // jax.process_count()], replace=False).astype(int)
    # mask_is_data = jax.random.choice(key=seed, a=2, shape=[stored_weights.size], replace=True).astype(bool)
    mask_is_data = jnp.zeros_like(stored_weights, dtype=bool).at[data_indices_in_randoms].set(True)
    mask_is_randoms = jnp.invert(mask_is_data)

    template_values_data = template_values[mask_is_data]
    template_values_randoms = template_values[mask_is_randoms]
    del template_values

    logger.info("Template values shape: %s", template_values_data.shape)

    logger.debug("Data mask sum: %s", mask_is_data.sum())

    pos_min, pos_max = _get_extent(positions)
    boxcenter = 0.5 * (pos_min + pos_max)

    logger.info("Done with preparation.")

    # Can pre-paint the randoms, since they won't move. Can't paint data.
    mattrs = get_mesh_attrs(positions, cellsize=cellsize, boxsize=boxsize, check=True)  # [8000.0, 16000.0, 8000.0], check=True)
    randoms = ParticleField(positions[~mask_is_data], weights=stored_weights[~mask_is_data], attrs=mattrs, exchange=True, backend="jax")
    randoms_mesh = randoms.paint(resampler="tsc", interlacing=3, compensate=True)
    randoms_shotnoise = (randoms.weights**2).sum()
    data = ParticleField(positions[mask_is_data], weights=stored_weights[mask_is_data], attrs=mattrs, exchange=True, backend="jax")

    get_RIC_weights = get_RIC_forward_model(
        data_positions=data.positions,
        randoms_positions=randoms.positions,
        randoms_weights=randoms.weights,
        n_bins=n_bins_RIC,
        boxsize=boxsize,
        boxcenter=boxcenter,
    )

    logger.info("Trying to get RIC weights.")
    get_RIC_weights(data.weights).block_until_ready()
    logger.info("Got RIC weights.")

    if data.exchange_direct is not None:
        template_values_data = data.exchange_direct(make_array_from_process_local_data(template_values_data, pad="mean"), pad=0.0)
    if randoms.exchange_direct is not None:
        template_values_randoms = randoms.exchange_direct(make_array_from_process_local_data(template_values_randoms, pad="mean"), pad=0.0)

    get_AIC_weights = get_AIC_forward_model(
        data_weights=data.weights,
        randoms_weights=randoms.weights,
        template_values_data=template_values_data,
        template_values_randoms=template_values_randoms,
        n_bins=n_bins_AIC,
    )

    get_NAM_weights = get_NAM_forward_model(
        data_positions=data.positions,
        randoms_positions=randoms.positions,
        randoms_weights=randoms.weights,
        nside=nside,
    )
    logger.debug("Data size: %s", data_size)
    logger.debug("Data field weights shape: %s", data.weights.shape)

    del positions, stored_weights
    logger.info("Trying to get AIC weights.")
    get_AIC_weights(data.weights).block_until_ready()
    logger.info("Got AIC weights.")
    logger.info("Trying to get NAM weights.")
    get_NAM_weights(data.weights).block_until_ready()
    logger.info("Got NAM weights.")

    # To go from the mesh to the observed power spectrum
    # Can always rebin ell = 2 to 0.002 later
    binner = BinMesh2SpectrumPoles(randoms.attrs, edges={"step": 0.001}, ells=(0, 2, 4))

    # Precompute the FKP power spectrum norm, without accounting for future data painting
    FKP_field = FKPField(data=data, randoms=randoms)
    fkp_norm = compute_fkp2_normalization(FKP_field, bin=binner, cellsize=10)
    del FKP_field

    theory = fiducial_planck_2018(jnp.arange(0.0, mattrs.knyq.max(), 0.001))  # Stop at the selection's k_nyq
    # forward_model_pk = jax.jit(
    #     partial(
    #         mock_survey,
    #         theory=theory,
    #         los=LOS,
    #         unitary_amplitude=UNITARY_AMPLITUDE,
    #         binner=binner,
    #         randoms_shotnoise=randoms_shotnoise,
    #         fkp_norm=fkp_norm,
    #     ),
    #     static_argnames=["get_RIC_weights", "get_AIC_weights", "get_NAM_weights"],
    # )
    forward_model_pk = partial(
        mock_survey,
        theory=theory,
        los=LOS,
        unitary_amplitude=UNITARY_AMPLITUDE,
        binner=binner,
        randoms_shotnoise=randoms_shotnoise,
        fkp_norm=fkp_norm,
    )
    # left: seed, data, randoms_mesh, get_AIC_weights, get_RIC_weights
    logger.info("Getting sharded power spectra...")
    pks_geo_shard = [
        forward_model_pk(seed=jax.random.key(seed), data=data, randoms_mesh=randoms_mesh, get_RIC_weights=None, get_AIC_weights=None, get_NAM_weights=None)
        for seed in tqdm(range(nreal))
    ]
    pks_geo_shard_mean = mean(pks_geo_shard)

    pks_RIC_shard = [
        forward_model_pk(
            seed=jax.random.key(seed),
            data=data,
            randoms_mesh=randoms_mesh,
            get_RIC_weights=get_RIC_weights,
            get_AIC_weights=None,
            get_NAM_weights=None,
        )
        for seed in tqdm(range(nreal))
    ]
    pks_RIC_shard_mean = mean(pks_RIC_shard)

    pks_ARIC_shard = [
        forward_model_pk(
            seed=jax.random.key(seed),
            data=data,
            randoms_mesh=randoms_mesh,
            get_RIC_weights=get_RIC_weights,
            get_AIC_weights=get_AIC_weights,
            get_NAM_weights=None,
        )
        for seed in tqdm(range(nreal))
    ]
    pks_ARIC_shard_mean = mean(pks_ARIC_shard)

    pks_NAMRIC_shard = [
        forward_model_pk(
            seed=jax.random.key(seed),
            data=data,
            randoms_mesh=randoms_mesh,
            get_RIC_weights=get_RIC_weights,
            get_AIC_weights=None,
            get_NAM_weights=get_NAM_weights,
        )
        for seed in tqdm(range(nreal))
    ]
    pks_NAMRIC_shard_mean = mean(pks_NAMRIC_shard)


# # # Without sharding
# logger.info("Getting unsharded power spectra...")
# pks_geo_noshard = [
#     forward_model_pk(
#         seed=jax.random.key(seed),
#         data=data,
#         randoms_mesh=randoms_mesh,
#         get_RIC_weights=None,
#         get_AIC_weights=None,
#         get_NAM_weights=None,
#     )
#     for seed in tqdm(range(nreal))
# ]
# pks_geo_noshard_mean = mean(pks_geo_noshard)

# pks_RIC_noshard = [
#     forward_model_pk(
#         seed=jax.random.key(seed),
#         data=data,
#         randoms_mesh=randoms_mesh,
#         get_RIC_weights=get_RIC_weights,
#         get_AIC_weights=None,
#         get_NAM_weights=None,
#     )
#     for seed in tqdm(range(nreal))
# ]
# pks_RIC_noshard_mean = mean(pks_RIC_shard)

# pks_ARIC_noshard = [
#     forward_model_pk(
#         seed=jax.random.key(seed),
#         data=data,
#         randoms_mesh=randoms_mesh,
#         get_RIC_weights=get_RIC_weights,
#         get_AIC_weights=get_AIC_weights,
#         get_NAM_weights=None,
#     )
#     for seed in tqdm(range(nreal))
# ]
# pks_ARIC_noshard_mean = mean(pks_ARIC_shard)

# pks_NAMRIC_noshard = [
#     forward_model_pk(
#         seed=jax.random.key(seed),
#         data=data,
#         randoms_mesh=randoms_mesh,
#         get_RIC_weights=get_RIC_weights,
#         get_AIC_weights=None,
#         get_NAM_weights=get_NAM_weights,
#     )
#     for seed in tqdm(range(nreal))
# ]
# pks_NAMRIC_noshard_mean = mean(pks_NAMRIC_shard)

pks_geo_shard_mean.write("./sharding_tests/pks_geo_shard_mean.h5")
pks_RIC_shard_mean.write("./sharding_tests/pks_RIC_shard_mean.h5")
pks_ARIC_shard_mean.write("./sharding_tests/pks_ARIC_shard_mean.h5")
pks_NAMRIC_shard_mean.write("./sharding_tests/pks_NAMRIC_shard_mean.h5")

fig, lax = plt.subplots(3, 1, layout="constrained")
for ell in [0, 2, 4]:
    color = f"C{ell}"
    ax = lax[ell // 2]

    for pks_shard, label, ls in zip(
        [pks_geo_shard_mean, pks_RIC_shard_mean, pks_ARIC_shard_mean, pks_NAMRIC_shard_mean],
        ["Geometry", "RIC", "ARIC", "NAMRIC"],
        ["-", "--", "-.", ":"],
        strict=True,
    ):
        pole_shard = pks_shard.get(ell)
        ax.plot(pole_shard.coords("k"), pole_shard.coords("k") * pole_shard.value(), color=color, ls=ls, label=label)
lax[0].legend()
fig.supxlabel("$k$")
fig.supylabel(r"$k(P_\ell^\mathrm{shard}(k) - P_\mathrm{noshard}\ell(k))$")

fig.savefig("./sharding_tests/sharding.pdf")

jax.distributed.shutdown()
