"""
Shared helpers for the shot-noise validation scripts (``scripts/validate_*.py``).

They build small, fast surveys out of the local EZmock QSO catalogs (``Cosmo/data/EZmocks6gpc/catalogs/QSO_Y3_v1``): the x10 randoms play the unclustered "data" (a random half of a pool of randoms) and the "randoms", and the clustered mocks can play the data (V13). The EZmocks carry no imaging systematics, so smooth synthetic template maps are used for AMR: the regression then responds to the data noise like in the real pipeline (which is all the template depends on), whatever the actual correlation with the density.

Use large cells (``--cellsize 150-250``) to run on a laptop CPU.
"""

import argparse
import itertools
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import healpy as hp
import jax

jax.config.update("jax_enable_x64", True)  # differences of spectra and finite differences need float64

import jax.numpy as jnp
import numpy as np
from cosmoprimo.fiducial import DESI
from jaxpower import BinMesh2SpectrumPoles, FKPField, ParticleField, compute_fkp2_normalization, get_mesh_attrs
from scipy import stats

from desiwinds.forward import prepare_AMR, prepare_RIC
from desiwinds.shotnoise import prepare_field_weights

EZ_DIR = Path(os.environ.get("EZMOCK_DIR", Path.home() / "Cosmo/data/EZmocks6gpc/catalogs/QSO_Y3_v1"))
REGIONS = ("NGC", "SGC")
OUT_DIR = Path(__file__).parent / "output"


# ---------------------------------------------------------------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------------------------------------------------------------


def base_parser(description: str) -> argparse.ArgumentParser:
    """Arguments common to all scripts."""
    p = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--cellsize", type=float, default=200.0, help="Mesh cell size [Mpc/h]; large values make the scripts light")
    p.add_argument("--n-pool", type=int, default=300_000, help="Objects per cap in the pool of randoms (after the redshift cut)")
    p.add_argument("--data-fraction", type=float, default=0.5, help="Fraction of the pool playing the data")
    p.add_argument("--zmin", type=float, default=0.8)
    p.add_argument("--zmax", type=float, default=1.3)
    p.add_argument("--ric-bins", type=int, default=100)
    p.add_argument("--amr-bins", type=int, default=5)
    p.add_argument("--n-sys", type=int, default=4, help="Number of synthetic imaging templates")
    p.add_argument("--regions", nargs="+", default=list(REGIONS), choices=REGIONS)
    p.add_argument("--n-real", type=int, default=32, help="Monte Carlo realizations")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=OUT_DIR, help="Output directory for arrays and figures")
    return p


# ---------------------------------------------------------------------------------------------------------------------------------
# Catalogs
# ---------------------------------------------------------------------------------------------------------------------------------

_COSMO = None


def distance(z):
    """Comoving distance [Mpc/h] in the DESI fiducial cosmology (astropy engine: no CLASS needed)."""
    global _COSMO
    if _COSMO is None:
        _COSMO = DESI(engine="astropy").get_background()
    return np.asarray(_COSMO.comoving_radial_distance(z))


def to_cartesian(ra, dec, z):
    """Sky coordinates to cartesian positions [Mpc/h] centered on the observer."""
    d, ra, dec = distance(z), np.deg2rad(ra), np.deg2rad(dec)
    return np.column_stack([d * np.cos(dec) * np.cos(ra), d * np.cos(dec) * np.sin(ra), d * np.sin(dec)])


def read_catalog(path: Path, zrange: tuple[float, float], n_read: int | None = None, n_keep: int | None = None, seed: int = 0) -> dict:
    """
    Read RA, DEC, Z and FKP weights, apply the redshift cut and optionally subsample.

    ``n_read`` limits how many rows are read from disk (the x10 randoms files are large; their rows are in random order).
    """
    with h5py.File(path) as f:
        sl = slice(None, n_read)
        ra, dec, z, w = (f[k][sl].astype(float) for k in ("RA", "DEC", "Z", "WEIGHT_FKP"))
    keep = (z > zrange[0]) & (z < zrange[1])
    ra, dec, z, w = ra[keep], dec[keep], z[keep], w[keep]
    if n_keep is not None and n_keep < len(z):
        idx = np.sort(np.random.default_rng(seed).choice(len(z), n_keep, replace=False))
        ra, dec, z, w = ra[idx], dec[idx], z[idx], w[idx]
    return dict(ra=ra, dec=dec, z=z, w_fkp=w)


def load_randoms(region: str, n: int, zrange, seed: int = 0) -> dict:
    """Pool of randoms for one cap (``n`` objects after the redshift cut)."""
    frac = max((zrange[1] - zrange[0]) / 2.3, 0.05)  # fraction of the file that survives the cut, conservatively
    return read_catalog(EZ_DIR / f"randoms-x10_QSO_Y3_{region}.h5", zrange, n_read=int(n / frac * 1.5), n_keep=n, seed=seed)


def load_mock(region: str, imock: int, zrange, n: int | None = None, seed: int = 0) -> dict:
    """Clustered EZmock (``imock`` from 1) for one cap, optionally subsampled to ``n`` objects."""
    return read_catalog(EZ_DIR / "data" / f"QSO_Y3_{region}_{imock}.h5", zrange, n_keep=n, seed=seed)


def available_mocks(region: str = "NGC") -> list[int]:
    """List the indices of the clustered mocks present locally for ``region``."""
    return sorted(int(p.stem.split("_")[-1]) for p in (EZ_DIR / "data").glob(f"QSO_Y3_{region}_*.h5"))


def datalike_weights(cat: dict, amplitude: float = 0.4) -> np.ndarray:
    """Smooth, bounded, position-dependent weights standing in for completeness/imaging weights (requirement 9 of the math). ``amplitude=0`` gives unit weights."""
    ra, dec = np.deg2rad(cat["ra"]), np.deg2rad(cat["dec"])
    return 1.0 + amplitude * np.sin(3 * ra) * np.cos(2 * dec) * (0.5 + 0.5 * np.cos(4 * cat["z"]))


def template_maps(nside: int = 32, n_sys: int = 4, seed: int = 0) -> np.ndarray:
    """Smooth synthetic imaging template maps, normalized to zero mean and unit variance, shape ``(12 nside^2, n_sys)``."""
    np.random.seed(seed)  # noqa: NPY002 -- healpy draws from the global numpy state
    cl = 1.0 / (np.arange(3 * nside) + 5.0) ** 2.5
    maps = np.stack([hp.synfast(cl, nside) for _ in range(n_sys)], -1)
    return (maps - maps.mean(0)) / maps.std(0)


# ---------------------------------------------------------------------------------------------------------------------------------
# Survey
# ---------------------------------------------------------------------------------------------------------------------------------


@dataclass
class Survey:
    """FKP fields, binner and effect arguments, ready to be passed to the functions of :py:mod:`desiwinds.shotnoise`."""

    fkp_fields: tuple
    data: tuple
    randoms: tuple
    binner: BinMesh2SpectrumPoles
    norms: list
    ric_args: object
    amr_args: object
    regions: tuple
    meta: dict = field(default_factory=dict)

    def kwargs(self, ric: bool = True, amr: bool = True, gic: bool = True, **extra) -> dict:
        """
        Keyword arguments of ``sample_shotnoise_template_*``, ``analytic_shotnoise_template`` and ``measure_power_spectrum_and_shotnoise`` (``fkp_fields`` are given separately).

        ``ric=amr=False, gic=False``: geometry only; ``gic=True``: geometry and global integral constraint. ``extra`` overrides or adds arguments.
        """
        use = ric or amr
        field_weights_args = prepare_field_weights(
            *self.fkp_fields,
            estimator_weights="weight_FKP",
            ric_args=self.ric_args if ric else None,
            amr_args=self.amr_args if amr else None,
            data_regions=self.ric_args.data_regions if use else None,
            randoms_regions=self.ric_args.randoms_regions if use else None,
            gic=gic,
        )
        return {"field_weights_args": field_weights_args, "binner": self.binner, "fkp_norms": self.norms} | extra

    @property
    def i0(self) -> np.ndarray:
        """:math:`I_0` per region, the ``norms`` expected by :py:func:`desiwinds.shotnoise.template`."""
        return np.array([np.ravel(n)[0] for n in self.norms])

    @property
    def k(self) -> np.ndarray:
        """Bin centers."""
        return np.asarray(self.binner.xavg)

    @property
    def ells(self) -> tuple[int, ...]:
        """Multipoles."""
        return tuple(self.binner.ells)


def build_survey(
    data_cats: dict[str, dict],
    randoms_cats: dict[str, dict],
    *,
    cellsize: float,
    zrange: tuple[float, float],
    ric_bins: int = 100,
    amr_bins: int = 5,
    n_sys: int = 4,
    weight_amplitude: float = 0.0,
    ells=(0, 2, 4),
    seed: int = 0,
    tmaps: np.ndarray | None = None,
    nside_tmaps: int = 32,
    mattrs=None,
    norms: list | None = None,
) -> Survey:
    r"""
    Build a :py:class:`Survey` from per-cap catalogs of "data" and "randoms".

    ``norms`` (list of per-region :math:`I_0`) and ``mattrs`` can be reused from another survey to compare surveys on the same footing.

    ``weight_amplitude`` modulates the base weights of data **and** randoms with the same smooth function: data-like weights have non-trivial :math:`\bar n \langle w^2\rangle(\mathbf x)`. The FKP weights read from the files are the (frozen) estimator weights ``weight_FKP``.
    """
    regions = tuple(data_cats)
    tmaps = template_maps(nside_tmaps, n_sys, seed) if tmaps is None else tmaps
    all_pos = np.concatenate([to_cartesian(c["ra"], c["dec"], c["z"]) for c in itertools.chain(data_cats.values(), randoms_cats.values())])
    if mattrs is None:
        mattrs = get_mesh_attrs(all_pos, cellsize=cellsize, check=True)

    def particles(cat):
        pos = to_cartesian(cat["ra"], cat["dec"], cat["z"])
        tv = tmaps[hp.ang2pix(nside_tmaps, cat["ra"], cat["dec"], lonlat=True)]
        return ParticleField(
            jnp.asarray(pos),
            weights=jnp.asarray(datalike_weights(cat, weight_amplitude) if weight_amplitude else np.ones(len(pos))),
            attrs=mattrs,
            extra={"Z": jnp.asarray(cat["z"]), "weight_FKP": jnp.asarray(cat["w_fkp"]), "template_values": jnp.asarray(tv)},
        )

    data = tuple(particles(data_cats[r]) for r in regions)
    randoms = tuple(particles(randoms_cats[r]) for r in regions)
    fkp_fields = tuple(FKPField(d, r, attrs=mattrs) for d, r in zip(data, randoms, strict=True))

    kf = 2 * np.pi / float(np.max(np.asarray(mattrs.boxsize)))
    binner = BinMesh2SpectrumPoles(mattrs, edges={"min": 0.0, "step": 2 * kf}, ells=ells)
    compute_norms = norms is None
    norms = [] if compute_norms else norms
    for fkp in fkp_fields if compute_norms else ():  # I_0 includes the estimator weights; any value would do for the template, which is a ratio
        weighted = fkp.clone(
            data=fkp.data.clone(weights=fkp.data.weights * fkp.data.extra["weight_FKP"]),
            randoms=fkp.randoms.clone(weights=fkp.randoms.weights * fkp.randoms.extra["weight_FKP"]),
        )
        norms.append(compute_fkp2_normalization(weighted, bin=binner, cellsize=max(cellsize / 2, 40.0)))

    ric = prepare_RIC(data=data, randoms=randoms, regions=list(regions), n_bins=ric_bins, apply_to="randoms")
    amr = prepare_AMR(data=data, randoms=randoms, regions_zranges=[(r, zrange) for r in regions], apply_to="randoms", n_bins=amr_bins)
    return Survey(fkp_fields, data, randoms, binner, norms, ric, amr, regions, dict(cellsize=cellsize, zrange=zrange, weight_amplitude=weight_amplitude, seed=seed))


def random_half_survey(args, *, weight_amplitude: float = 0.0, pool_seed: int = 0, split_seed: int = 123, n_pool: int | None = None, **kwargs) -> Survey:
    """Pool of randoms per cap, of which a random fraction ``args.data_fraction`` plays the data."""
    zr = (args.zmin, args.zmax)
    data_cats, randoms_cats = {}, {}
    for region in args.regions:
        pool = load_randoms(region, n_pool or args.n_pool, zr, seed=pool_seed)
        is_data = np.random.default_rng(split_seed).uniform(size=len(pool["z"])) < args.data_fraction
        data_cats[region] = {k: v[is_data] for k, v in pool.items()}
        randoms_cats[region] = {k: v[~is_data] for k, v in pool.items()}
    return build_survey(
        data_cats,
        randoms_cats,
        cellsize=args.cellsize,
        zrange=zr,
        ric_bins=args.ric_bins,
        amr_bins=args.amr_bins,
        n_sys=args.n_sys,
        weight_amplitude=weight_amplitude,
        seed=args.seed,
        **kwargs,
    )


# ---------------------------------------------------------------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------------------------------------------------------------


def chi2_report(diff: np.ndarray, cov: np.ndarray, label: str, mask: np.ndarray | None = None, n_real: int | None = None) -> dict:
    """
    ``chi2``, dof and p-value of a difference ``diff`` given its covariance ``cov`` (flattened bins).

    The full covariance (with the Hartlap debiasing factor) is used when there are at least three times more realizations than bins, otherwise the diagonal only, which is stated in the output.
    """
    diff, cov = np.ravel(diff), np.asarray(cov)
    mask = np.ones(diff.shape, bool) if mask is None else np.ravel(mask)
    mask = mask & (np.diag(cov) > 0)
    d, c = diff[mask], cov[np.ix_(mask, mask)]
    full = n_real is not None and n_real >= 3 * d.size
    chi2 = float((n_real - d.size - 2) / (n_real - 1) * d @ np.linalg.solve(c, d)) if full else float(np.sum(d**2 / np.diag(c)))
    dof = int(d.size)
    pvalue = float(stats.chi2.sf(chi2, dof)) if dof else float("nan")
    print(f"  {label:48s} chi2/dof = {chi2:8.1f}/{dof:<4d} p = {pvalue:.3f} ({'full' if full else 'diagonal'} covariance)")
    return dict(label=label, chi2=chi2, dof=dof, pvalue=pvalue, full_cov=bool(full))


def kmask(survey, kmax_fraction: float | None = None, low_k_bins: int | None = None) -> np.ndarray:
    """Boolean mask over ``(n_ells, n_k)``: ``k < kmax_fraction * k_Nyquist`` and/or the first ``low_k_bins`` bins."""
    m = np.ones((len(survey.ells), len(survey.k)), bool)
    if kmax_fraction is not None:
        knyq = float(np.min(np.pi / np.asarray(survey.binner.mattrs.cellsize)))
        m &= (survey.k < kmax_fraction * knyq)[None, :]
    if low_k_bins is not None:
        m[:, low_k_bins:] = False
    return m


def jackknife_stat(n: int, stat) -> tuple[np.ndarray, np.ndarray]:
    """
    Delete-one jackknife of an arbitrary statistic of ``n`` realizations.

    ``stat(idx)`` receives the indices of the kept realizations and returns an array; the result is its value on all realizations and the jackknife covariance over the flattened output. Use it for derived quantities (differences, extrapolations) that need correlated errors.
    """
    full = np.asarray(stat(np.arange(n)))
    loo = np.stack([np.ravel(stat(np.delete(np.arange(n), i))) for i in range(n)])
    delta = loo - loo.mean(0)
    return full, (n - 1) / n * delta.T @ delta


def ratio(num: np.ndarray, den: np.ndarray, idx=None, weights=None) -> np.ndarray:
    """Ratio of means over the realizations ``idx`` of per-region ``(n, R, L, K)`` / ``(n, R)`` arrays, regions combined with ``weights`` (``I_r``)."""
    idx = np.arange(num.shape[0]) if idx is None else idx
    w = np.ones(num.shape[1]) if weights is None else np.asarray(weights)
    return np.einsum("r,rlk->lk", w, num[idx].sum(0)) / np.einsum("r,r->", w, den[idx].sum(0))


def plot_template(ax_list, survey, entries, ells=None, title=None):
    """Plot ``entries = [(label, s, err), ...]`` (arrays ``(n_ells, n_k)``) for each multipole on a list of axes."""
    ells = ells or survey.ells
    for ax, ell in zip(ax_list, ells, strict=False):
        i = list(survey.ells).index(ell)
        for j, (label, s, err) in enumerate(entries):
            ax.errorbar(survey.k * (1 + 0.01 * j), s[i], None if err is None else err[i], marker=".", ms=3, lw=1, capsize=1.5, label=label)
        ax.set_title(rf"$\ell={ell}$" if title is None else f"{title}, $\\ell={ell}$")
        ax.set_xlabel("$k$ [$h$/Mpc]")
    ax_list[0].set_ylabel(r"$s_\ell(k)$")


def savefig(fig, out: Path, name: str):
    """Save a figure and report where."""
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"  figure: {path}")


def progress(iterable, desc=""):
    """Wrap ``iterable`` in a tqdm progress bar, only on an interactive terminal."""
    if not sys.stderr.isatty():
        return iterable
    try:
        from tqdm import tqdm

        return tqdm(iterable, desc=desc, leave=False)
    except ImportError:
        return iterable


def header(text: str):
    """Print a section header."""
    print(f"\n=== {text} ===", flush=True)
    sys.stdout.flush()
