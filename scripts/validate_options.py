"""
Checks V7, V8, V11, V12, V16, V17, V18 of the math: comparisons between options and bookkeeping checks, on the random-based setup (RIC + AMR).

* V7  Gaussian and Rademacher noise give the same mean (option B); variance ratio per bin,
* V8  subsampling data and randoms by two leaves the template unchanged (options A and B),
* V11 sigma dependence of option A (Rademacher 0.25, 0.5, 1 and legacy Gaussian 10): differences to C scale as sigma^2; extrapolation to sigma -> 0,
* V12 curvature shift Delta s from option C and its significance per bin,
* V16 randoms' shot-noise term: ``sum_R w_R^2 / sum_a w_a^2`` versus alpha and the low-k shape it implies,
* V17 plain versus data-like weights on the "data" half,
* V18 control variate (analytic geometry + GIC template): agreement with the plain estimate, coefficient and variance reduction.

Run: ``python scripts/validate_options.py [--cellsize 200 --n-real 24]``. Use ``--skip`` to drop some checks.
"""

import matplotlib

matplotlib.use("Agg")
import jax
import matplotlib.pyplot as plt
import numpy as np
from shotnoise_common import (
    base_parser,
    chi2_report,
    header,
    jackknife_stat,
    kmask,
    plot_template,
    random_half_survey,
    ratio,
    savefig,
)

from desiwinds.shotnoise import (
    analytic_shotnoise_template,
    shotnoise_template_with_control_variate,
    combine_regions_weighted_by_normalization,
    shotnoise_template_curvature_shift,
    apply_field_weights,
    sample_shotnoise_template_antithetic,
    sample_shotnoise_template_linearized,
    sample_shotnoise_template_quadratic,
    select_realizations,
    shotnoise_template_from_samples,
    measure_power_spectrum_and_shotnoise,
)


def err_of(res, norms):
    s, cov = shotnoise_template_from_samples(res, norms)
    return s, cov, np.sqrt(np.diag(cov)).reshape(s.shape)


def independent_difference(res1, res2, label, survey, n_real, mask=None):
    """chi2 of the difference between two *independent* estimates (covariances add)."""
    s1, c1, _ = err_of(res1, survey.i0)
    s2, c2, _ = err_of(res2, survey.i0)
    return chi2_report(s1 - s2, c1 + c2, label, kmask(survey, 0.5) if mask is None else mask, n_real)


def main():
    parser = base_parser(__doc__)
    parser.add_argument("--skip", nargs="*", default=[], choices=["V7", "V8", "V11", "V12", "V16", "V17", "V18"])
    parser.add_argument("--n-real-a", type=int, default=None, help="Realizations for the (slower) option A runs; defaults to --n-real")
    args = parser.parse_args()
    n, n_a = args.n_real, args.n_real_a or args.n_real
    survey = random_half_survey(args)
    fields, norms, kw, key = survey.fkp_fields, survey.i0, survey.kwargs(ric=True, amr=True), jax.random.key(args.seed)
    summary = {}

    header("Reference runs: options B and C, RIC + AMR, Rademacher")
    res_b = sample_shotnoise_template_linearized(*fields, key=key, n_real=n, **kw)
    res_c = sample_shotnoise_template_quadratic(*fields, key=key, n_real=n, **kw)
    s_b, _, e_b = err_of(res_b, norms)
    s_c, _, e_c = err_of(res_c, norms)

    # ------------------------------------------------------------------ V7
    if "V7" not in args.skip:
        header("V7: Gaussian versus Rademacher noise (option B)")
        res_g = sample_shotnoise_template_linearized(*fields, key=jax.random.key(args.seed + 1000), n_real=n, **kw, noise_distribution="gaussian")
        independent_difference(res_b, res_g, "mean: Rademacher - Gaussian (independent draws)", survey, n)
        _, _, e_g = err_of(res_g, norms)
        vr = (e_g / np.where(e_b > 0, e_b, np.nan)) ** 2
        third = len(survey.k) // 3
        for iell, ell in enumerate(survey.ells):
            print(f"  variance ratio Gaussian/Rademacher, l={ell}: low-k third {np.nanmedian(vr[iell, :third]):.2f}, mid {np.nanmedian(vr[iell, third : 2 * third]):.2f}, high-k third {np.nanmedian(vr[iell, 2 * third :]):.2f}")
        print("  (a ratio above 1 means Rademacher is the more precise choice; the gain is expected mainly at low k)")

    # ------------------------------------------------------------------ V8
    if "V8" not in args.skip:
        header("V8: subsampling data and randoms by 2")
        half = random_half_survey(args, n_pool=args.n_pool // 2, pool_seed=1)
        kw_half = half.kwargs(ric=True, amr=True)
        res_half_b = sample_shotnoise_template_linearized(*half.fkp_fields, key=key, n_real=n, **kw_half)
        independent_difference(res_b, res_half_b, "option B: full - half catalogs", survey, n)
        res_a_full = sample_shotnoise_template_antithetic(*fields, key=key, n_real=n_a, **kw, sigma=1.0)
        res_a_half = sample_shotnoise_template_antithetic(*half.fkp_fields, key=key, n_real=n_a, sigma=1.0, **kw_half)
        independent_difference(res_a_full, res_a_half, "option A (sigma=1): full - half catalogs", survey, n_a)

    # ------------------------------------------------------------------ V11
    sigmas = (0.25, 0.5, 1.0)
    if "V11" not in args.skip:
        header("V11: sigma dependence of option A")
        runs = {s: sample_shotnoise_template_antithetic(*fields, key=key, n_real=n_a, **kw, sigma=s) for s in sigmas}
        res_c_a = sample_shotnoise_template_quadratic(*fields, key=key, n_real=n_a, **kw) if n_a != n else res_c
        low = kmask(survey, 0.5)
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
        _, _, e_stat = err_of(res_c_a, norms)
        s_c_a = shotnoise_template_from_samples(res_c_a, norms)[0]
        diffs = {s: shotnoise_template_from_samples(runs[s], norms)[0] - s_c_a for s in sigmas}  # paired draws: the noise cancels, only the sigma-dependent bias remains
        print("  A - C on common draws (the statistical error of s is the reference scale):")
        for s, d in diffs.items():
            print(f"    sigma = {s:4}: max|A - C| = {np.abs(d[low]).max():.2e} = {np.abs(d[low]).max() / np.nanmax(e_stat[low]):.1e} of the largest statistical error; rms = {np.sqrt(np.mean(d[low] ** 2)):.2e}")
        rms = [np.sqrt(np.mean(diffs[s][low] ** 2)) for s in sigmas]
        exponents = [np.log(rms[i + 1] / rms[i]) / np.log(sigmas[i + 1] / sigmas[i]) for i in range(len(sigmas) - 1)]
        ok = all(1.7 < e < 2.3 for e in exponents)
        print(f"  local power law of rms(A - C) in sigma: {np.round(exponents, 2)} (expected 2 for an O(sigma^2) bias) -> {'PASS' if ok else 'CHECK'}")
        axes[0].loglog(sigmas, rms, "o-", label="rms(A - C)")
        axes[0].loglog(sigmas, rms[-1] * (np.array(sigmas) / sigmas[-1]) ** 2, "k--", label=r"$\propto \sigma^2$")
        axes[0].set_xlabel(r"$\sigma$")
        axes[0].set_ylabel("rms(A - C), $k<k_{Nyq}/2$")
        axes[0].legend()

        def extrapolate(idx):
            ys = np.stack([ratio(runs[s]["power_spectrum_response"], runs[s]["shotnoise_response"], idx, norms) for s in sigmas])  # (3, L, K)
            x = np.square(sigmas)
            design = np.stack([np.ones_like(x), x], 1)
            coef = np.linalg.lstsq(design, ys.reshape(len(sigmas), -1), rcond=None)[0]
            return coef[0] - ratio(res_c_a["power_spectrum_response"], res_c_a["shotnoise_response"], idx, norms).ravel()

        d, _ = jackknife_stat(n_a, extrapolate)
        print(f"  A extrapolated to sigma -> 0 (linear in sigma^2) minus C: max = {np.abs(d.reshape(s_c_a.shape)[low]).max():.2e} (sigma = 0.25 point: {np.abs(diffs[sigmas[0]][low]).max():.2e})")
        print("  legacy Gaussian sigma = 10 (independent draws, not common with C):")
        res_leg = sample_shotnoise_template_antithetic(*fields, key=jax.random.key(args.seed + 2000), n_real=n_a, **kw, sigma=10.0, noise_distribution="gaussian")
        independent_difference(res_leg, res_c_a, "A (Gaussian sigma=10) - C", survey, n_a)
        entries = [(f"A sigma={s}", *err_of(runs[s], norms)[::2]) for s in sigmas] + [("C", s_c, e_c), ("A Gaussian 10", *err_of(res_leg, norms)[::2])]
        plot_template([axes[1]], survey, entries, ells=[0], title="option A")
        axes[1].legend(fontsize=7)
        savefig(fig, args.out, "validate_options_sigma")

    # ------------------------------------------------------------------ V12
    if "V12" not in args.skip:
        header("V12: curvature on the random-based setup (option C, common draws)")
        delta, cov = shotnoise_template_curvature_shift(res_c, norms)
        err = np.sqrt(np.diag(cov)).reshape(delta.shape)
        np.set_printoptions(precision=4, linewidth=200, suppress=True)
        for iell, ell in enumerate(survey.ells):
            print(f"  l={ell}: delta s = {delta[iell]}\n        significance = {delta[iell] / np.where(err[iell] > 0, err[iell], np.nan)}")
        chi2_report(delta, cov, "curvature shift compatible with zero", None, n)
        fig, axes = plt.subplots(1, len(survey.ells), figsize=(4.2 * len(survey.ells), 3.4), constrained_layout=True)
        plot_template(axes, survey, [("C - B (curvature)", delta, err)], title="$\\Delta s$")
        savefig(fig, args.out, "validate_options_curvature")

    # ------------------------------------------------------------------ V16
    if "V16" not in args.skip:
        header("V16: randoms' own shot-noise term")
        for ireg, (w, fkp) in enumerate(zip(apply_field_weights(kw["field_weights_args"].input_data_weights, kw["field_weights_args"]), fields, strict=True)):
            nd = fkp.data.weights.shape[0]
            data_term, rand_term = float(np.sum(np.asarray(w[:nd]) ** 2)), float(np.sum(np.asarray(w[nd:]) ** 2))
            alpha = float(fkp.data.weights.sum() / fkp.randoms.weights.sum())
            print(f"  region {survey.regions[ireg]}: sum_R w^2 / sum_D w^2 = {rand_term / data_term:.3f} (alpha = N_D/N_R = {alpha:.3f}; with FKP weights frozen they differ by the weight moments)")
        noise_free_spectrum, noise_free_shotnoise = (np.asarray(x) for x in measure_power_spectrum_and_shotnoise(*fields, **kw))
        n0, d0 = combine_regions_weighted_by_normalization(noise_free_spectrum[None], noise_free_shotnoise[None], norms)
        shape_poisson = n0[0] / d0[0]
        print("  low-k shape, noise-free Poisson P_0 / S (includes the randoms' own shot noise) vs injected-noise template s (data noise only):")
        for iell, ell in enumerate(survey.ells):
            print(f"    l={ell}: Poisson {np.round(shape_poisson[iell, :4], 3)} | template {np.round(s_b[iell, :4], 3)}")
        print("  -> never use the un-subtracted Poisson run as the template: its low-k shape differs (this is why the noise-free run is subtracted in option A).")

    # ------------------------------------------------------------------ V17
    if "V17" not in args.skip:
        header("V17: plain versus data-like weights on the data half (option B, common draws)")
        weighted = random_half_survey(args, weight_amplitude=0.4)
        res_w = sample_shotnoise_template_linearized(*weighted.fkp_fields, key=key, n_real=n, **weighted.kwargs(ric=True, amr=True))

        def diff(idx):
            return ratio(res_w["power_spectrum_response"], res_w["shotnoise_response"], idx, norms) - ratio(res_b["power_spectrum_response"], res_b["shotnoise_response"], idx, norms)

        d, cov = jackknife_stat(n, diff)
        chi2_report(d, cov, "data-like - plain weights, all bins", None, n)
        chi2_report(d, cov, "data-like - plain weights, low-k third", kmask(survey, low_k_bins=len(survey.k) // 3), n)
        print(f"  max |difference| in s_0 at low k: {np.abs(d.reshape(s_b.shape)[0, : len(survey.k) // 3]).max():.3f}; in s_2: {np.abs(d.reshape(s_b.shape)[1, : len(survey.k) // 3]).max():.3f}")

    # ------------------------------------------------------------------ V18
    if "V18" not in args.skip:
        header("V18: control variate (analytic geometry + GIC template)")
        res_cv = sample_shotnoise_template_linearized(*fields, key=key, n_real=n, **kw, compute_control_variate=True)
        analytic_spectrum_response, analytic_shotnoise_response, _ = analytic_shotnoise_template(*fields, **kw, include_gic=True)
        plain, cov_p = shotnoise_template_from_samples(res_cv, norms)
        for coefficient in (1.0, "fit"):
            s_cv, cov_cv, diag = shotnoise_template_with_control_variate(res_cv, norms, analytic_spectrum_response, analytic_shotnoise_response, coefficient=coefficient)

            def diff(idx, coefficient=coefficient):
                sub = select_realizations(res_cv, idx)
                return shotnoise_template_with_control_variate(sub, norms, analytic_spectrum_response, analytic_shotnoise_response, coefficient=coefficient)[0] - shotnoise_template_from_samples(sub, norms)[0]

            d, cov = jackknife_stat(n, diff)
            chi2_report(d, cov, f"CV(coefficient={coefficient}) - plain", kmask(survey, 0.5), n)
            ratio_var = np.diag(cov_cv) / np.where(np.diag(cov_p) > 0, np.diag(cov_p), np.nan)
            print(f"    median variance ratio CV/plain = {np.nanmedian(ratio_var):.3f}; median 1-rho^2 of the numerator = {np.nanmedian(diag['variance_reduction']):.3f}")
            if coefficient == "fit":
                print(f"    fitted coefficient (spectrum response), median per multipole: {[round(float(np.median(diag['spectrum_coefficient'][:, i, :])), 2) for i in range(len(survey.ells))]}")
                print(f"    fitted coefficient (shot-noise response), median: {np.median(diag['shotnoise_coefficient']):.2f}")

    print("\nDone.")
    return summary


if __name__ == "__main__":
    main()
