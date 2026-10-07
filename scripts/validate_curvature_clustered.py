"""
Checks V13 (and V12) of the math: curvature of the template on clustered data, and the decision B versus C.

Option C is run on a model whose "data" is a clustered EZmock (so that ``w_0`` is the clustered field, which the curvature term depends on) and whose randoms are the pool of randoms; the same is done for the random-based setup (V12). The curvature shift ``Delta s_l`` is turned into power-spectrum units with ``S_obs`` and compared with the statistical error of the measured multipoles, estimated from the scatter between the local EZmocks: if ``S_obs |Delta s_l|`` is below about 0.1 of that error in every bin, option B suffices; otherwise use C.

Caveat printed with the result: the EZmock error bars are those of the (sub)sampled local catalogs, not of the real DESI data; rerun with the full catalogs / DESI covariance for the actual decision.

Run: ``python scripts/validate_curvature_clustered.py [--cellsize 200 --n-mocks 6 --n-real 16]``
"""

import matplotlib

matplotlib.use("Agg")
import jax
import matplotlib.pyplot as plt
import numpy as np
from shotnoise_common import (
    available_mocks,
    base_parser,
    build_survey,
    header,
    load_mock,
    load_randoms,
    plot_template,
    random_half_survey,
    savefig,
    template_maps,
)

from desiwinds.shotnoise import shotnoise_template_curvature_shift, sample_shotnoise_template_quadratic, measure_power_spectrum_and_shotnoise


def main():
    parser = base_parser(__doc__)
    parser.add_argument("--n-mocks", type=int, default=6, help="Number of local EZmocks used to estimate the statistical error")
    parser.add_argument("--randoms-pool", type=int, default=600_000, help="Randoms per cap for the clustered setup")
    args = parser.parse_args()
    zr = (args.zmin, args.zmax)
    mocks = [m for m in available_mocks(args.regions[0]) if all(m in available_mocks(r) for r in args.regions)][: args.n_mocks]
    if len(mocks) < 3:
        raise SystemExit(f"Need at least 3 local mocks present in all requested regions, found {mocks}")
    print(f"mocks: {mocks}")

    header("Catalogs")
    mock_cats = {m: {r: load_mock(r, m, zr) for r in args.regions} for m in mocks}
    n_keep = {r: min(len(mock_cats[m][r]["z"]) for m in mocks) for r in args.regions}  # same number of objects in each mock: no recompilation
    for m in mocks:
        for r in args.regions:
            cat = mock_cats[m][r]
            idx = np.sort(np.random.default_rng(m).choice(len(cat["z"]), n_keep[r], replace=False))
            mock_cats[m][r] = {k: v[idx] for k, v in cat.items()}
    randoms = {r: load_randoms(r, args.randoms_pool, zr, seed=0) for r in args.regions}
    print(f"  objects per mock: {n_keep}; randoms: { {r: len(c['z']) for r, c in randoms.items()} }")
    tmaps = template_maps(32, args.n_sys, args.seed)

    surveys = []
    ref = None
    for m in mocks:
        sv = build_survey(
            mock_cats[m],
            randoms,
            cellsize=args.cellsize,
            zrange=zr,
            ric_bins=args.ric_bins,
            amr_bins=args.amr_bins,
            n_sys=args.n_sys,
            seed=args.seed,
            tmaps=tmaps,
            mattrs=None if ref is None else ref.fkp_fields[0].attrs,
            norms=None if ref is None else ref.norms,
        )
        ref = ref or sv
        surveys.append(sv)
    survey = surveys[0]
    norms = survey.i0

    header("Statistical error from the local EZmocks (conventional estimator: Q - S in the monopole)")
    p_all, s_all = [], []
    for sv in surveys:
        q_regions, s_regions = measure_power_spectrum_and_shotnoise(*sv.fkp_fields, **sv.kwargs(ric=True, amr=True))
        q = np.einsum("r,rlk->lk", norms, np.asarray(q_regions)) / norms.sum()
        s = float(np.einsum("r,r->", norms, np.asarray(s_regions)) / norms.sum())
        q[0] -= s
        p_all.append(q)
        s_all.append(s)
    p_all, s_obs = np.stack(p_all), float(np.mean(s_all))
    sigma_p = p_all.std(0, ddof=1)
    print(f"  S_obs = {s_obs:.4g}; sigma_P(l=0) at lowest bins: {sigma_p[0, :4]}")

    header("Option C on the clustered mock (data = EZmock) and on the random-based setup")
    key = jax.random.key(args.seed)
    res_clustered = sample_shotnoise_template_quadratic(*survey.fkp_fields, key=key, n_real=args.n_real, **survey.kwargs(ric=True, amr=True))
    random_based = random_half_survey(args)
    res_random = sample_shotnoise_template_quadratic(*random_based.fkp_fields, key=key, n_real=args.n_real, **random_based.kwargs(ric=True, amr=True))

    fig, axes = plt.subplots(1, len(survey.ells), figsize=(4.2 * len(survey.ells), 3.6), constrained_layout=True)
    entries = []
    for label, res in (("clustered mock", res_clustered), ("random-based", res_random)):
        delta, cov = shotnoise_template_curvature_shift(res, norms)
        err = np.sqrt(np.diag(cov)).reshape(delta.shape)
        entries.append((f"curvature, {label}", delta, err))
        sig = delta / np.where(err > 0, err, np.nan)
        print(f"  {label}: max |delta s| = {np.abs(delta).max():.4f}; significance per bin (|delta s|/err) max = {np.nanmax(np.abs(sig)):.1f}")
        for iell, ell in enumerate(survey.ells):
            print(f"    l={ell}: delta s = {np.round(delta[iell], 4)}")
    plot_template(axes, survey, entries, title="$\\Delta s$")
    axes[0].legend(fontsize=7)
    savefig(fig, args.out, "validate_curvature_shift")

    header("V13 decision: S_obs |delta s_l| / sigma_P,l(k)")
    delta, cov = shotnoise_template_curvature_shift(res_clustered, norms)
    ratio = s_obs * np.abs(delta) / np.where(sigma_p > 0, sigma_p, np.nan)
    for iell, ell in enumerate(survey.ells):
        print(f"  l={ell}: {np.round(ratio[iell], 4)}")
    worst = float(np.nanmax(ratio))
    print(f"  worst bin: {worst:.3f} -> {'option B is sufficient (< 0.1)' if worst < 0.1 else 'use option C (>= 0.1 somewhere)'}")
    print("  (error bars from the local, possibly subsampled, EZmocks: indicative only)")


if __name__ == "__main__":
    main()
