"""
Checks V14 (key test) of the math: Poisson-mock truth test.

Many unclustered catalogs are drawn from the pool of randoms at the data density (data = random subset, randoms = the others), the **full** pipeline (RIC, AMR, RIC again, region renormalization, FKP weights, alpha) is run on each, and the standard estimator is applied. The mean of the measured ``Q_l(k)`` is the true shot-noise contribution, including discreteness and all nonlinear orders of the pipeline. It is compared to ``S_obs * s_l(k)`` from options A, B and C (and to the conventional flat ``S_obs * delta_{l0}``), where ``S_obs = <S(w_0)>`` is the conventional shot noise of the data monopole; equivalently, after the conventional subtraction, ``P_0 = S_obs (s_0 - 1)`` and ``P_{l>0} = S_obs s_l``.

The number of objects of each mock is fixed (sampling without replacement) to avoid recompilation: this removes the Poisson fluctuation of the total number, which is irrelevant for the shape. ``--data-like`` modulates the weights smoothly so that ``n <w^2>`` varies over the sky.

Run: ``python scripts/validate_poisson_truth.py [--cellsize 200 --n-mocks 24 --n-real 24]``
"""

import matplotlib

matplotlib.use("Agg")
import jax
import matplotlib.pyplot as plt
import numpy as np
from shotnoise_common import (
    base_parser,
    build_survey,
    chi2_report,
    header,
    kmask,
    load_randoms,
    random_half_survey,
    savefig,
    template_maps,
)

from desiwinds.shotnoise import (
    sample_shotnoise_template_antithetic,
    sample_shotnoise_template_linearized,
    sample_shotnoise_template_quadratic,
    shotnoise_template_from_samples,
    measure_power_spectrum_and_shotnoise,
)


def main():
    parser = base_parser(__doc__)
    parser.add_argument("--n-mocks", type=int, default=24, help="Number of Poisson mocks")
    parser.add_argument("--data-like", action="store_true", help="Smoothly modulated weights instead of unit weights")
    args = parser.parse_args()
    amplitude = 0.4 if args.data_like else 0.0
    zr = (args.zmin, args.zmax)

    header("Reference setup and templates")
    ref = random_half_survey(args, weight_amplitude=amplitude)
    kw, norms, key = ref.kwargs(ric=True, amr=True), ref.i0, jax.random.key(args.seed)
    tmaps = template_maps(32, args.n_sys, args.seed)
    templates = {
        "A (sigma=1)": sample_shotnoise_template_antithetic(*ref.fkp_fields, key=key, n_real=args.n_real, sigma=1.0, **kw),
        "B": sample_shotnoise_template_linearized(*ref.fkp_fields, key=key, n_real=args.n_real, **kw),
        "C": sample_shotnoise_template_quadratic(*ref.fkp_fields, key=key, n_real=args.n_real, **kw),
    }

    header(f"Truth: {args.n_mocks} Poisson mocks through the full pipeline")
    pools = {r: load_randoms(r, args.n_pool, zr, seed=0) for r in args.regions}  # same pool as the reference survey
    q_all, s_all = [], []
    for imock in range(args.n_mocks):
        data, rand = {}, {}
        for region, pool in pools.items():
            n_obj = len(pool["z"])
            n_data = round(args.data_fraction * n_obj)
            is_data = np.zeros(n_obj, bool)
            is_data[np.random.default_rng(10_000 + imock).choice(n_obj, n_data, replace=False)] = True
            data[region] = {k: v[is_data] for k, v in pool.items()}
            rand[region] = {k: v[~is_data] for k, v in pool.items()}
        mock = build_survey(
            data,
            rand,
            cellsize=args.cellsize,
            zrange=zr,
            ric_bins=args.ric_bins,
            amr_bins=args.amr_bins,
            n_sys=args.n_sys,
            weight_amplitude=amplitude,
            seed=args.seed,
            tmaps=tmaps,
            mattrs=ref.fkp_fields[0].attrs,
            norms=ref.norms,
        )
        q, s = measure_power_spectrum_and_shotnoise(*mock.fkp_fields, **mock.kwargs(ric=True, amr=True))  # no shot-noise subtraction
        q_all.append(np.asarray(q))
        s_all.append(np.asarray(s))
    q_all, s_all = np.stack(q_all), np.stack(s_all)
    inorm = norms.sum()
    q_comb = np.einsum("r,mrlk->mlk", norms, q_all) / inorm
    s_comb = np.einsum("r,mr->m", norms, s_all) / inorm
    n = args.n_mocks
    truth, truth_err = q_comb.mean(0), q_comb.std(0, ddof=1) / np.sqrt(n)
    s_obs = s_comb.mean()
    print(f"  S_obs = <S(w_0)> = {s_obs:.4g} (scatter {s_comb.std(ddof=1) / s_obs:.1e} relative)")

    header("Comparison: <Q_l> (truth) versus S_obs * s_l (all bins; k < k_Nyq/2; l > 0; lowest 3 bins)")
    cov_truth = np.cov(q_comb.reshape(n, -1), rowvar=False) / n
    masks = {
        "all bins": kmask(ref),
        "k < k_Nyq/2": kmask(ref, 0.5),
        "lowest 3 bins": kmask(ref, low_k_bins=3),
    }
    mask_l = np.ones_like(masks["all bins"])
    mask_l[0] = False
    masks["l > 0"] = mask_l
    flat = np.zeros_like(truth)
    flat[0] = 1.0
    predictions = {"flat (conventional)": (flat, np.zeros_like(cov_truth))}
    for name, res in templates.items():
        s, cov = shotnoise_template_from_samples(res, norms)
        predictions[name] = (s, cov)
    stats_out = {}
    for name, (s, cov_s) in predictions.items():
        pred = s_obs * s
        cov = cov_truth + s_obs**2 * cov_s
        print(f" {name}")
        for label, mask in masks.items():
            stats_out[(name, label)] = chi2_report((truth - pred), cov, f"{name}: {label}", mask, min(n, args.n_real))
    print("  (chi2 uses the diagonal of the covariance unless there are >= 3 realizations per bin; the first four masks share the same bins partially)")

    fig, axes = plt.subplots(2, len(ref.ells), figsize=(4.2 * len(ref.ells), 6), sharex=True, constrained_layout=True, height_ratios=[2, 1])
    for iell, ell in enumerate(ref.ells):
        ax, rx = axes[0, iell], axes[1, iell]
        ax.errorbar(ref.k, truth[iell] / s_obs, truth_err[iell] / s_obs, fmt="k.", label="Poisson mocks (truth)")
        for name, (s, _) in predictions.items():
            ax.plot(ref.k, s[iell], label=name, ls="--" if name.startswith("flat") else "-")
            rx.plot(ref.k, (truth[iell] / s_obs - s[iell]) / (truth_err[iell] / s_obs), label=name)
        ax.set_title(rf"$\ell={ell}$")
        rx.set_xlabel("$k$ [$h$/Mpc]")
        rx.axhline(0, color="k", lw=0.5)
        ax.set_ylabel(r"$Q_\ell / S_{\rm obs}$")
        rx.set_ylabel("(truth - template) / error")
    axes[0, 0].legend(fontsize=7)
    savefig(fig, args.out, "validate_poisson_truth")


if __name__ == "__main__":
    main()
