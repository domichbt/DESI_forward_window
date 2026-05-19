"""Test forward modeling on sharded arrays."""

# For now do FM and autocorr only

import sys
import os

sys.path.insert(0, "/global/homes/d/dchebat/window/desiwinds/src/")
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.85"

import jax

jax.config.update("jax_enable_x64", True)
jax.distributed.initialize()

import itertools
from pathlib import Path
import healpy as hp
import jax.numpy as jnp
import lsstypes as lss
import matplotlib.pyplot as plt
import numpy as np
from jaxpower import BinMesh2SpectrumPoles, FKPField, compute_fkp2_normalization, get_mesh_attrs, create_sharding_mesh
from tqdm import tqdm

from desiwinds.convenience import fiducial_planck_2018, get_randoms, get_sysmap, split_into_fields
from desiwinds.forward import mock_survey_catalog, prepare_AMR, prepare_NAM, prepare_RIC

LOS = "local"
UNITARY_AMPLITUDE = True
nreal_fw = 10

n_randoms = 1

boxsize = 8000.0
cellsize = 40.0

tracer = "LRG"
z_range = (0.4, 1.1)

pk_regions = ["SGC", "NGC"]
photo_regions = ["N", "S"]  # Photometric regions for LRG
z_ranges = [(0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0), (1.0, 1.1)]  # Redshift ranges for the angular systematics regression

data_to_randoms = 0.5  # Use 50% of the randoms catalogs as "data" and the rest as randoms

regression_maps = [
    "STARDENS",
    "PSFSIZE_G",
    "PSFSIZE_R",
    "PSFSIZE_Z",
    "GALDEPTH_G",
    "GALDEPTH_R",
    "GALDEPTH_Z",
    "HI",
    "PSFDEPTH_W1",
    "EBV_DIFF_GR",
    "EBV_DIFF_RZ",
]  # LRG templates

nside_nam = 64

randoms_basedir = Path("/dvs_ro/cfs/cdirs/desi/survey/catalogs/Y3/LSS/loa-v1/LSScats/v2/fNL")
templates_dir = Path("/dvs_ro/cfs/cdirs/desi/survey/catalogs/Y3/LSS/loa-v1/LSScats/v2/hpmaps/")
ebvdir = Path("/dvs_ro/cfs/cdirs/desicollab/users/rongpu/data/ebv/desi_stars_y3/v0.1/final_maps/lss/")

sysmaps_south = get_sysmap(
    map_path=Path(templates_dir / f"{tracer}_mapprops_healpix_nested_nside256_S.fits"),
    region="S",
    ebv_path=ebvdir / "desi_ebv_lss_256.fits",
)
sysmaps_south.keep_columns(regression_maps)
sysmaps_south = np.lib.recfunctions.structured_to_unstructured(sysmaps_south.as_array()).astype(float)

sysmaps_north = get_sysmap(
    map_path=Path(templates_dir / f"{tracer}_mapprops_healpix_nested_nside256_N.fits"),
    region="N",
    ebv_path=ebvdir / "desi_ebv_lss_256.fits",
)
sysmaps_north.keep_columns(regression_maps)
sysmaps_north = np.lib.recfunctions.structured_to_unstructured(sysmaps_north.as_array()).astype(float)

sysmaps = np.where(sysmaps_south == 0.0, sysmaps_north, sysmaps_south)
sysmapsnan = np.where(sysmaps == 0.0, np.nan, sysmaps)

sysmaps -= np.nanmean(sysmapsnan, axis=0)
sysmaps /= np.nanstd(sysmapsnan, axis=0)

del sysmaps_south, sysmaps_north, sysmapsnan

with create_sharding_mesh() as sharding_mesh:
    fkp_fields = []
    fkp_norms = []

    data = []
    randoms = []

    for pk_region in pk_regions:
        # Load data for single tracer
        position, [weight, weight_FKP], redshift = get_randoms(
            n_randoms,
            pk_region,
            z_range,
            tracer,
            "FKP",
            basedir=randoms_basedir,
            return_redshift=True,
        )

        template_values = sysmaps[hp.vec2pix(256, *position.T, nest=True)]

        # Create mesh attributes
        mattrs = get_mesh_attrs(position, cellsize=cellsize, boxsize=boxsize, check=True)

        # Create data/randoms split
        _data, _randoms, _mask_is_data = split_into_fields(
            position,
            weight,
            extra={"Z": redshift, "weight_FKP": weight_FKP, "template_values": template_values},
            data_size=int(weight.shape[0] * data_to_randoms),
            split_seed=123,
            mattrs=mattrs,
            exchange=True,
            backend="jax",
        )

        data.append(_data)
        randoms.append(_randoms)

        # Create FKP field
        fkp_field = FKPField(data=_data, randoms=_randoms, attrs=mattrs)
        fkp_fields.append(fkp_field)

        # Compute normalization
        binner = BinMesh2SpectrumPoles(mattrs=mattrs, edges={"min": 0.001, "step": 0.001}, ells=(0, 2, 4))
        fkp_norm = compute_fkp2_normalization(fkp_field, bin=binner, cellsize=20.0)
        fkp_norms.append(fkp_norm)

        del position, weight, weight_FKP, redshift, template_values, _mask_is_data, _data, _randoms
    del sysmaps

    theory = fiducial_planck_2018(jnp.arange(0.0, jnp.pi / cellsize, 0.001))

    ric_args = prepare_RIC(
        data=tuple(data),  # i.e. (data_sgc, data_ngc)
        randoms=tuple(randoms),  # i.e. (randoms_sgc, randoms_ngc)
        regions=photo_regions,
        n_bins=1000,
        apply_to="randoms",
    )

    amr_args = prepare_AMR(
        data=tuple(data),  # i.e. (data_sgc, data_ngc)
        randoms=tuple(randoms),  # i.e. (randoms_sgc, randoms_ngc)
        regions_zranges=list(itertools.product(photo_regions, z_ranges)),
        apply_to="randoms",
    )
    amr_args_half = prepare_AMR(
        data=tuple(data),  # i.e. (data_sgc, data_ngc)
        randoms=tuple(randoms),  # i.e. (randoms_sgc, randoms_ngc)
        regions_zranges=list(itertools.product(photo_regions, z_ranges)),
        apply_to="randoms",
        half_precision=True,
    )

    nam_args = prepare_NAM(
        data=tuple(data),  # i.e. (data_sgc, data_ngc)
        randoms=tuple(randoms),  # i.e. (randoms_sgc, randoms_ngc)
        regions_zranges=list(itertools.product(photo_regions, z_ranges)),
        apply_to="randoms",
        nside=nside_nam,
        prior=None,
    )

    nam_args_prior = prepare_NAM(
        data=tuple(data),  # i.e. (data_sgc, data_ngc)
        randoms=tuple(randoms),  # i.e. (randoms_sgc, randoms_ngc)
        regions_zranges=list(itertools.product(photo_regions, z_ranges)),
        apply_to="randoms",
        nside=nside_nam,
        prior=0.2,
    )

    for i in range(len(fkp_fields)):
        fkp_fields[i] = fkp_fields[i].clone(
            data=fkp_fields[i].data.clone(
                extra={k: fkp_fields[i].data.extra[k] for k in fkp_fields[i].data.extra.keys() - {"template_values", "Z"}},
            ),
            randoms=fkp_fields[i].randoms.clone(
                extra={k: fkp_fields[i].randoms.extra[k] for k in fkp_fields[i].randoms.extra.keys() - {"template_values", "Z"}},
            ),
        )

    fw_jit = jax.jit(mock_survey_catalog, static_argnames=["los", "unitary_amplitude", "estimator_weights"])

    pks_geo = [
            fw_jit(
                *fkp_fields,
                theory=theory,
                seed=jax.random.key(i * 3 + 87),
                los=LOS,
                unitary_amplitude=UNITARY_AMPLITUDE,
                ric_args=None,
                amr_args=None,
                nam_args=None,
                fkp_norms=fkp_norms,
                binner=binner,
                estimator_weights="weight_FKP",
                data_regions=ric_args.data_regions,
                randoms_regions=ric_args.randoms_regions,
            )
        for i in tqdm(range(nreal_fw), desc="Geometry forward modeling", disable=jax.process_index() != 0)
    ]
    pks_ric = [
        fw_jit(
            *fkp_fields,
            theory=theory,
            seed=jax.random.key(i * 3 + 87),
            los=LOS,
            unitary_amplitude=UNITARY_AMPLITUDE,
            ric_args=ric_args,
            amr_args=None,
            nam_args=None,
            fkp_norms=fkp_norms,
            binner=binner,
            estimator_weights="weight_FKP",
            data_regions=ric_args.data_regions,
            randoms_regions=ric_args.randoms_regions,
        )
        for i in tqdm(range(nreal_fw), desc="RIC forward modeling", disable=jax.process_index() != 0)
    ]
    pks_amr = [
        fw_jit(
            *fkp_fields,
            theory=theory,
            seed=jax.random.key(i * 3 + 87),
            los=LOS,
            unitary_amplitude=UNITARY_AMPLITUDE,
            ric_args=ric_args,
            amr_args=amr_args,
            nam_args=None,
            fkp_norms=fkp_norms,
            binner=binner,
            estimator_weights="weight_FKP",
            data_regions=ric_args.data_regions,
            randoms_regions=ric_args.randoms_regions,
        )
        for i in tqdm(range(nreal_fw), desc="AMR forward modeling", disable=jax.process_index() != 0)
    ]

    pks_amr_half = [
        fw_jit(
            *fkp_fields,
            theory=theory,
            seed=jax.random.key(i * 3 + 87),
            los=LOS,
            unitary_amplitude=UNITARY_AMPLITUDE,
            ric_args=ric_args,
            amr_args=amr_args_half,
            nam_args=None,
            fkp_norms=fkp_norms,
            binner=binner,
            estimator_weights="weight_FKP",
            data_regions=ric_args.data_regions,
            randoms_regions=ric_args.randoms_regions,
        )
        for i in tqdm(range(nreal_fw), desc="AMR forward modeling (half precision)", disable=jax.process_index() != 0)
    ]

    pks_nam = [
        fw_jit(
            *fkp_fields,
            theory=theory,
            seed=jax.random.key(i * 3 + 87),
            los=LOS,
            unitary_amplitude=UNITARY_AMPLITUDE,
            ric_args=ric_args,
            amr_args=None,
            nam_args=nam_args,
            fkp_norms=fkp_norms,
            binner=binner,
            estimator_weights="weight_FKP",
            data_regions=ric_args.data_regions,
            randoms_regions=ric_args.randoms_regions,
        )
        for i in tqdm(range(nreal_fw), desc="NAM forward modeling", disable=jax.process_index() != 0)
    ]
    pks_nam_prior = [
        fw_jit(
            *fkp_fields,
            theory=theory,
            seed=jax.random.key(i * 3 + 87),
            los=LOS,
            unitary_amplitude=UNITARY_AMPLITUDE,
            ric_args=ric_args,
            amr_args=None,
            nam_args=nam_args_prior,
            fkp_norms=fkp_norms,
            binner=binner,
            estimator_weights="weight_FKP",
            data_regions=ric_args.data_regions,
            randoms_regions=ric_args.randoms_regions,
        )
        for i in tqdm(range(nreal_fw), desc="NAM forward modeling", disable=jax.process_index() != 0)
    ]

if jax.process_index() == 0:
    # Create and save figures

    ## RIC ##
    fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharey=True, sharex=True, layout="constrained")
    avg_effect_ric = [lss.mean([pk_ric.clone(value=pk_ric.value() - pk_geo.value()) for pk_geo, pk_ric in zip(pks_geo_group, pks_ric_group)]) for pks_geo_group, pks_ric_group in zip(zip(*pks_geo), zip(*pks_ric))]
    cov_effect_ric = [lss.cov([pk_ric.clone(value=pk_ric.value() - pk_geo.value()) for pk_geo, pk_ric in zip(pks_geo_group, pks_ric_group)]) for pks_geo_group, pks_ric_group in zip(zip(*pks_geo), zip(*pks_ric))]

    for isurvey, (ax, nsurvey) in enumerate(zip(axes, ["SGC", "NGC"], strict=False)):
        for ell in [0, 2, 4]:
            pole = avg_effect_ric[isurvey].get(ell)
            std = cov_effect_ric[isurvey].at.observable.get(ell).std().real / np.sqrt(nreal_fw)
            k = pole.coords("k")
            ax.plot(k, k * pole.value(), label=f"$\\ell = ${ell}")
            ax.fill_between(k, k * (pole.value() - std), k * (pole.value() + std), alpha=0.3)
        ax.set_title(nsurvey)

    ax = axes[2]
    cov = lss.cov([pk_ric[0].clone(value=sum(pk_ric).value() - sum(pk_geo).value()) for pk_geo, pk_ric in zip(pks_geo, pks_ric)])
    for ell in [0, 2, 4]:
        pole = (avg_effect_ric[0] + avg_effect_ric[1]).get(ell)
        std = cov.at.observable.get(ell).std().real / np.sqrt(nreal_fw)
        k = pole.coords("k")
        ax.plot(k, k * pole.value(), label=f"$\\ell = ${ell}")
        ax.fill_between(k, k * (pole.value() - std), k * (pole.value() + std), alpha=0.3)
    ax.set_title("GCcomb")

    axes[0].legend()
    fig.supxlabel("$k$ [h/Mpc]")
    fig.supylabel(r"$k \cdot (P_\mathrm{RIC} - P_\mathrm{geo})$")
    fig.suptitle("RIC effect on power spectrum")
    fig.savefig("./test_sharding_ric.pdf", bbox_inches="tight")

    ## AMR ##

    fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharey=True, sharex=True, layout="constrained")
    avg_effect_amr = [lss.mean([pk_amr.clone(value=pk_amr.value() - pk_ric.value()) for pk_ric, pk_amr in zip(pks_ric_group, pks_amr_group)]) for pks_ric_group, pks_amr_group in zip(zip(*pks_ric), zip(*pks_amr))]
    cov_effect_amr = [lss.cov([pk_amr.clone(value=pk_amr.value() - pk_ric.value()) for pk_ric, pk_amr in zip(pks_ric_group, pks_amr_group)]) for pks_ric_group, pks_amr_group in zip(zip(*pks_ric), zip(*pks_amr))]

    for isurvey, (ax, nsurvey) in enumerate(zip(axes, ["SGC", "NGC"], strict=False)):
        for ell in [0, 2, 4]:
            pole = avg_effect_amr[isurvey].get(ell)
            std = cov_effect_amr[isurvey].at.observable.get(ell).std().real / np.sqrt(nreal_fw)
            k = pole.coords("k")
            ax.plot(k, k * pole.value(), label=f"$\\ell = ${ell}")
            ax.fill_between(k, k * (pole.value() - std), k * (pole.value() + std), alpha=0.3)
        ax.set_title(nsurvey)

    ax = axes[2]
    cov = lss.cov([pk_amr[0].clone(value=sum(pk_amr).value() - sum(pk_ric).value()) for pk_ric, pk_amr in zip(pks_ric, pks_amr)])
    for ell in [0, 2, 4]:
        pole = (avg_effect_amr[0] + avg_effect_amr[1]).get(ell)
        std = cov.at.observable.get(ell).std().real / np.sqrt(nreal_fw)
        k = pole.coords("k")
        ax.plot(k, k * pole.value(), label=f"$\\ell = ${ell}")
        ax.fill_between(k, k * (pole.value() - std), k * (pole.value() + std), alpha=0.3)
    ax.set_title("GCcomb")

    axes[0].legend()
    fig.supxlabel("$k$ [h/Mpc]")
    fig.supylabel(r"$k \cdot (P_\mathrm{AMR} - P_\mathrm{RIC})$")
    fig.suptitle("AMR effect on power spectrum")
    fig.savefig("./test_sharding_amr.pdf", bbox_inches="tight")

    ## AMR HALF PRECISION ##
    fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharey=True, sharex=True, layout="constrained")
    avg_effect_amr = [lss.mean([pk_amr.clone(value=pk_amr.value() - pk_ric.value()) for pk_ric, pk_amr in zip(pks_ric_group, pks_amr_group)]) for pks_ric_group, pks_amr_group in zip(zip(*pks_ric), zip(*pks_amr))]
    avg_effect_amr_half = [lss.mean([pk_amr.clone(value=pk_amr.value() - pk_ric.value()) for pk_ric, pk_amr in zip(pks_ric_group, pks_amr_group)]) for pks_ric_group, pks_amr_group in zip(zip(*pks_ric), zip(*pks_amr_half))]

    for isurvey, (ax, nsurvey) in enumerate(zip(axes, ["SGC", "NGC"], strict=False)):
        for ell in [0, 2, 4]:
            pole = avg_effect_amr[isurvey].get(ell)
            pole_half = avg_effect_amr_half[isurvey].get(ell)
            k = pole.coords("k")
            ax.plot(k, k * (pole.value() - pole_half.value()), color=f"C{ell//2}", label=f"$\\ell = ${ell}")
        ax.set_title(nsurvey)

    ax = axes[2]
    cov = lss.cov([pk_amr[0].clone(value=sum(pk_amr).value() - sum(pk_ric).value()) for pk_ric, pk_amr in zip(pks_ric, pks_amr)])
    for ell in [0, 2, 4]:
        pole = (avg_effect_amr[0] + avg_effect_amr[1]).get(ell)
        pole_half = (avg_effect_amr_half[0] + avg_effect_amr_half[1]).get(ell)
        k = pole.coords("k")
        ax.plot(k, k * (pole.value() - pole_half.value()), color=f"C{ell//2}", label=f"$\\ell = ${ell}")
    ax.set_title("GCcomb")

    axes[0].legend()
    fig.supxlabel("$k$ [h/Mpc]")
    fig.supylabel(r"$k \cdot (P_\mathrm{AMR} - P_\mathrm{RIC})$")
    fig.suptitle("Difference between AMR effect on power spectrum, half precision or not")
    fig.savefig("./test_sharding_amr_halfprecision.pdf", bbox_inches="tight")

    ## NAM ##
    fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharey=True, sharex=True, layout="constrained")
    avg_effect_nam = [lss.mean([pk_nam.clone(value=pk_nam.value() - pk_ric.value()) for pk_ric, pk_nam in zip(pks_ric_group, pks_nam_group)]) for pks_ric_group, pks_nam_group in zip(zip(*pks_ric), zip(*pks_nam))]
    cov_effect_nam = [lss.cov([pk_nam.clone(value=pk_nam.value() - pk_ric.value()) for pk_ric, pk_nam in zip(pks_ric_group, pks_nam_group)]) for pks_ric_group, pks_nam_group in zip(zip(*pks_ric), zip(*pks_nam))]
    avg_effect_namprior = [lss.mean([pk_nam.clone(value=pk_nam.value() - pk_ric.value()) for pk_ric, pk_nam in zip(pks_ric_group, pks_nam_group)]) for pks_ric_group, pks_nam_group in zip(zip(*pks_ric), zip(*pks_nam_prior))]
    cov_effect_namprior = [lss.cov([pk_nam.clone(value=pk_nam.value() - pk_ric.value()) for pk_ric, pk_nam in zip(pks_ric_group, pks_nam_group)]) for pks_ric_group, pks_nam_group in zip(zip(*pks_ric), zip(*pks_nam_prior))]

    for isurvey, (ax, nsurvey) in enumerate(zip(axes, ["SGC", "NGC"], strict=False)):
        for ell in [0, 2, 4]:
            pole = avg_effect_nam[isurvey].get(ell)
            std = cov_effect_nam[isurvey].at.observable.get(ell).std().real / np.sqrt(nreal_fw)
            poleprior = avg_effect_namprior[isurvey].get(ell)
            stdprior = cov_effect_namprior[isurvey].at.observable.get(ell).std().real / np.sqrt(nreal_fw)
            k = pole.coords("k")
            ax.plot(k, k * pole.value(), label=rf"$\ell = {{{ell}}}$", color=f"C{ell//2}")
            ax.fill_between(k, k * (pole.value() - std), k * (pole.value() + std), alpha=0.3, color=f"C{ell//2}")
            k = poleprior.coords("k")
            ax.plot(k, k * poleprior.value(), color=f"C{ell//2}", ls='--')
            ax.fill_between(k, k * (poleprior.value() - stdprior), k * (poleprior.value() + stdprior), alpha=0.3, color=f"C{ell//2}")
        ax.set_title(nsurvey)

    ax = axes[2]
    cov = lss.cov([pk_nam[0].clone(value=sum(pk_nam).value() - sum(pk_ric).value()) for pk_ric, pk_nam in zip(pks_ric, pks_nam)])
    covprior = lss.cov([pk_nam[0].clone(value=sum(pk_nam).value() - sum(pk_ric).value()) for pk_ric, pk_nam in zip(pks_ric, pks_nam_prior)])
    for ell in [0, 2, 4]:
        pole = (avg_effect_nam[0] + avg_effect_nam[1]).get(ell)
        std = cov.at.observable.get(ell).std().real / np.sqrt(nreal_fw)
        k = pole.coords("k")
        ax.plot(k, k * pole.value(), color=f"C{ell//2}")
        ax.fill_between(k, k * (pole.value() - std), k * (pole.value() + std), alpha=0.3, color=f"C{ell//2}")
        
        pole = (avg_effect_namprior[0] + avg_effect_namprior[1]).get(ell)
        std = covprior.at.observable.get(ell).std().real / np.sqrt(nreal_fw)
        k = pole.coords("k")
        ax.plot(k, k * pole.value(), color=f"C{ell//2}", ls='--')
        ax.fill_between(k, k * (pole.value() - std), k * (pole.value() + std), alpha=0.3, color=f"C{ell//2}")
    ax.set_title("GCcomb")
    ax.plot([], [], ls='-', color="grey", label="NAM")
    ax.plot([], [], ls='--', color="grey", label="NAM, prior 0.2")
    ax.legend()

    axes[0].legend()
    fig.supxlabel("$k$ [h/Mpc]")
    fig.supylabel(r"$k \cdot (P_\mathrm{NAM} - P_\mathrm{RIC})$")
    fig.suptitle("NAM effect on power spectrum")
    fig.savefig("./test_sharding_nam.pdf", bbox_inches="tight")

jax.distributed.shutdown()
