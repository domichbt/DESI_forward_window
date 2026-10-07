"""
Small synthetic surveys for the shot-noise tests: a few hundred objects in a box, no data files or cosmology needed.

The "survey" is a slab at 1000-2000 Mpc/h from the observer (at the origin) with a radial density gradient, so that the line of sight varies, RIC is non-trivial, and two photometric regions are defined by the sign of ``y``. Templates for AMR are random smooth functions of the position.
"""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jaxpower import BinMesh2SpectrumPoles, FKPField, MeshAttrs, ParticleField, compute_fkp2_normalization

from desiwinds.forward import AMR_args, NAM_args, RIC_args
from desiwinds.shotnoise import prepare_field_weights
from desiwinds.utils import local_argsort, prepare_templates

BOXSIZE = 1000.0
BOXCENTER = np.array([1500.0, 0.0, 0.0])
RIC_BINS, AMR_BINS, NSIDE, N_SYS = 6, 3, 2, 2


@dataclass
class Toy:
    """Everything needed to build a :py:class:`desiwinds.shotnoise.ShotNoiseModel`."""

    fkp_fields: tuple
    binner: BinMesh2SpectrumPoles
    fkp_norms: list
    ric_args: RIC_args
    amr_args: AMR_args
    nam_args: NAM_args
    data_regions: jnp.ndarray
    randoms_regions: jnp.ndarray
    mattrs: MeshAttrs

    def forward_kwargs(self, ric=False, amr=False, nam=False, regions=None, **extra):
        """Keyword arguments in the style of :py:func:`desiwinds.forward.mock_survey_catalog` (``mock_whitenoise``, ``prepare_field_weights``)."""
        use_regions = ric or amr or nam if regions is None else regions
        return {
            "binner": self.binner,
            "fkp_norms": self.fkp_norms,
            "estimator_weights": "weight_FKP",
            "ric_args": self.ric_args if ric else None,
            "amr_args": self.amr_args if amr else None,
            "nam_args": self.nam_args if nam else None,
            "data_regions": self.data_regions if use_regions else None,
            "randoms_regions": self.randoms_regions if use_regions else None,
        } | extra

    def kwargs(self, ric=False, amr=False, nam=False, regions=None, gic=True):
        """Keyword arguments for ``sample_shotnoise_template_*(*toy.fkp_fields, key=..., n_real=..., **toy.kwargs(...))``."""
        fw = self.forward_kwargs(ric, amr, nam, regions)
        field_weights_args = prepare_field_weights(*self.fkp_fields, gic=gic, **{k: v for k, v in fw.items() if k not in ("binner", "fkp_norms")})
        return {"field_weights_args": field_weights_args, "binner": self.binner, "fkp_norms": self.fkp_norms}


def _positions(rng, n):
    x = BOXCENTER[0] - BOXSIZE / 2 + BOXSIZE * rng.beta(1.6, 1.2, n)  # radial gradient
    yz = rng.uniform(-BOXSIZE / 2, BOXSIZE / 2, (n, 2))
    return np.column_stack([x, yz])


def _region_masks(positions):
    north = positions[:, 1] > 0
    return np.stack([north, ~north])


def make_toy(n_data=200, n_randoms=800, seed=0, meshsize=16, ells=(0, 2), n_regions=1) -> Toy:
    """Build ``n_regions`` independent FKP fields (same box and binner) and the effect arguments, for one fake tracer. ``n_regions`` effects are concatenated."""
    rng = np.random.default_rng(seed)
    mattrs = MeshAttrs(boxsize=BOXSIZE, boxcenter=jnp.asarray(BOXCENTER), meshsize=meshsize)
    binner = BinMesh2SpectrumPoles(mattrs, edges={"min": 0.0, "step": 2 * np.pi / BOXSIZE * 1.5}, ells=ells)

    data_p, rand_p, fields = [], [], []
    for _ in range(n_regions):
        pd, pr = _positions(rng, n_data), _positions(rng, n_randoms)
        base_data_weights, base_randoms_weights = rng.uniform(0.6, 1.4, n_data), rng.uniform(0.6, 1.4, n_randoms)
        fd, fr = rng.uniform(0.5, 1.5, n_data), rng.uniform(0.5, 1.5, n_randoms)
        data = ParticleField(jnp.asarray(pd), weights=jnp.asarray(base_data_weights), attrs=mattrs, extra={"weight_FKP": jnp.asarray(fd)})
        randoms = ParticleField(jnp.asarray(pr), weights=jnp.asarray(base_randoms_weights), attrs=mattrs, extra={"weight_FKP": jnp.asarray(fr)})
        fields.append(FKPField(data, randoms, attrs=mattrs))
        data_p.append(pd)
        rand_p.append(pr)

    pd, pr = np.concatenate(data_p), np.concatenate(rand_p)
    data_regions, randoms_regions = jnp.asarray(_region_masks(pd)), jnp.asarray(_region_masks(pr))

    d_dist, r_dist = np.linalg.norm(pd, axis=1), np.linalg.norm(pr, axis=1)
    edges = np.linspace(min(d_dist.min(), r_dist.min()) * 0.99, max(d_dist.max(), r_dist.max()) * 1.01, RIC_BINS)
    ric = RIC_args(
        data_distances_digitized=jnp.digitize(jnp.asarray(d_dist), jnp.asarray(edges)),
        randoms_distances_digitized=jnp.digitize(jnp.asarray(r_dist), jnp.asarray(edges)),
        data_regions=data_regions,
        randoms_regions=randoms_regions,
        data_to_remove=jnp.zeros(len(pd), dtype=bool),
        n_bins=RIC_BINS,
        apply_to="randoms",
    )

    def templates(p):
        return np.column_stack([np.sin(p[:, 1] / 300.0) + 0.3 * np.cos(p[:, 2] / 200.0), (p[:, 0] - BOXCENTER[0]) / 400.0 + 0.5 * np.sin(p[:, 2] / 150.0)])

    d_norm, d_dig, r_norm, r_dig = prepare_templates(
        data_templates=jnp.asarray(templates(pd)),
        randoms_templates=jnp.asarray(templates(pr)),
        data_regions=list(data_regions),
        randoms_regions=list(randoms_regions),
        randoms_is_real=jnp.ones(len(pr), dtype=bool),
        tail=2.0,
        n_bins=AMR_BINS,
        bin_margin=1e-7,
    )
    amr = AMR_args(
        data_regions=data_regions,
        randoms_regions=randoms_regions,
        data_templates_digitized=d_dig,
        randoms_templates_digitized=r_dig,
        data_templates_normalized=d_norm,
        randoms_templates_normalized=r_norm,
        data_isort=local_argsort(d_dig, axis=1),
        randoms_isort=local_argsort(r_dig, axis=1),
        n_bins=AMR_BINS,
        apply_to="randoms",
    )
    npix = 12 * NSIDE**2
    nam = NAM_args(
        data_pixels=jnp.asarray(rng.integers(0, npix, len(pd))),
        randoms_pixels=jnp.asarray(rng.integers(0, npix, len(pr))),
        data_regions=data_regions,
        randoms_regions=randoms_regions,
        data_to_remove=jnp.zeros(len(pd), dtype=bool),
        invsigma2=jnp.atleast_1d(1.0),
        nside=NSIDE,
        apply_to="randoms",
    )
    norms = [compute_fkp2_normalization(f, bin=binner, cellsize=50.0) for f in fields]
    return Toy(tuple(fields), binner, norms, ric, amr, nam, data_regions, randoms_regions, mattrs)
