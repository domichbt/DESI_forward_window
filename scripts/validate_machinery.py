"""
Checks V1-V4 of the math: correctness of the machinery, on a survey built from the local EZmock randoms.

* V2  linear pipeline identity (gic frozen, no effects): ``z = 0``, ``Y = X`` draw by draw, option A independent of sigma,
* V3  JVP and nested JVP against central finite differences (float64, scan of the step ``h``),
* A -> C  per draw, ``(P(+) + P(-))/2 - P(0)`` over ``sigma^2`` tends to option C as ``sigma -> 0``,
* V4  analytic limit: with RIC/AMR off, options B and C agree with the analytic template (geometry, geometry + GIC) within MC error. Exact (Hadamard) versions of V1/V4 on tiny catalogs are in ``tests/test_shotnoise.py``.

Run: ``python scripts/validate_machinery.py [--cellsize 200 --n-real 32]``
"""

import matplotlib

matplotlib.use("Agg")
from functools import partial

import jax
import matplotlib.pyplot as plt
import numpy as np
from shotnoise_common import base_parser, chi2_report, header, kmask, plot_template, random_half_survey, savefig

from desiwinds.shotnoise import (
    analytic_shotnoise_template,
    combine_regions_weighted_by_normalization,
    draw_relative_weight_noise,
    apply_field_weights,
    sample_shotnoise_template_antithetic,
    sample_shotnoise_template_linearized,
    sample_shotnoise_template_quadratic,
    shotnoise_template_from_samples,
)


def flat(w):
    return np.concatenate([np.asarray(x) for x in w])


def main():
    parser = base_parser(__doc__)
    args = parser.parse_args()
    survey = random_half_survey(args)
    fields, norms, key = survey.fkp_fields, survey.i0, jax.random.key(args.seed)
    results = {}

    # ------------------------------------------------------------------ V3
    header("V3: derivatives vs finite differences (RIC + AMR pipeline)")
    kw = survey.kwargs(ric=True, amr=True)
    input_data_weights = kw["field_weights_args"].input_data_weights
    phi_u = jax.jit(partial(apply_field_weights, field_weights_args=kw["field_weights_args"]))
    d = input_data_weights * draw_relative_weight_noise(jax.random.key(3), input_data_weights.shape)
    v, z = jax.jit(lambda: jax.jvp(lambda x: jax.jvp(phi_u, (x,), (d,))[1], (input_data_weights,), (d,)))()
    noise_free_field_weights = flat(phi_u(input_data_weights))
    scale = np.abs(noise_free_field_weights).max()
    print("  step h      max|v_jvp - v_fd|/max|w0|   max|z_jvp - z_fd|/max|w0|")
    best = [np.inf, np.inf]
    for h in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5):
        wp, wm = flat(phi_u(input_data_weights + h * d)), flat(phi_u(input_data_weights - h * d))
        ev = np.abs((wp - wm) / (2 * h) - flat(v)).max() / scale
        ez = np.abs((wp - 2 * noise_free_field_weights + wm) / h**2 - flat(z)).max() / scale
        best = [min(best[0], ev), min(best[1], ez)]
        print(f"  {h:8.0e}   {ev:24.3e}   {ez:24.3e}")
    results["V3"] = best[0] < 1e-5 and best[1] < 1e-2
    print(f"  best: v {best[0]:.2e}, z {best[1]:.2e} -> {'PASS' if results['V3'] else 'FAIL'}")

    # ------------------------------------------------------------------ V2
    header("V2: linear pipeline (no effects, alpha frozen)")
    kw_lin = survey.kwargs(ric=False, amr=False, gic=False)
    u0_lin = kw_lin["field_weights_args"].input_data_weights
    phi_lin = partial(apply_field_weights, field_weights_args=kw_lin["field_weights_args"])
    d_lin = u0_lin * draw_relative_weight_noise(jax.random.key(1), u0_lin.shape)
    _, zlin = jax.jvp(lambda x: jax.jvp(phi_lin, (x,), (d_lin,))[1], (u0_lin,), (d_lin,))
    c = sample_shotnoise_template_quadratic(*fields, key=key, n_real=4, **kw_lin)
    b = sample_shotnoise_template_linearized(*fields, key=key, n_real=4, **kw_lin)
    ok_z = np.abs(flat(zlin)).max() == 0.0
    ok_y = np.allclose(c["power_spectrum_response"], c["extras"]["linear_part"][0], rtol=1e-10) and np.allclose(c["shotnoise_response"], c["extras"]["linear_part"][1], rtol=1e-10)
    ok_a = True
    for sigma in (0.25, 1.0):
        a = sample_shotnoise_template_antithetic(*fields, key=key, n_real=4, sigma=sigma, **kw_lin)
        ok_a &= np.allclose(a["power_spectrum_response"], sigma**2 * b["power_spectrum_response"], rtol=1e-6, atol=1e-9 * np.abs(b["power_spectrum_response"]).max())
        print(f"  option A sigma={sigma}: max|A - sigma^2 B|/max|sigma^2 B| = {np.abs(a['power_spectrum_response'] - sigma**2 * b['power_spectrum_response']).max() / np.abs(sigma**2 * b['power_spectrum_response']).max():.1e}")
    results["V2"] = bool(ok_z and ok_y and ok_a)
    print(f"  z == 0: {ok_z}; Y == X per draw: {ok_y}; A independent of sigma: {ok_a} -> {'PASS' if results['V2'] else 'FAIL'}")

    # ------------------------------------------------------------------ A -> C
    header("Option A -> option C per draw as sigma -> 0 (RIC + AMR)")
    sigma = 1e-3
    a = sample_shotnoise_template_antithetic(*fields, key=key, n_real=3, sigma=sigma, **kw)
    cc = sample_shotnoise_template_quadratic(*fields, key=key, n_real=3, **kw)
    rel = np.abs(a["power_spectrum_response"] / sigma**2 - cc["power_spectrum_response"]).max() / np.abs(cc["power_spectrum_response"]).max()
    reld = np.abs(a["shotnoise_response"] / sigma**2 - cc["shotnoise_response"]).max() / np.abs(cc["shotnoise_response"]).max()
    results["A->C"] = rel < 1e-3 and reld < 1e-3
    print(f"  sigma = {sigma}: max relative difference numerator {rel:.1e}, denominator {reld:.1e} -> {'PASS' if results['A->C'] else 'FAIL'}")

    # ------------------------------------------------------------------ V4
    header("V4: analytic limit, RIC/AMR off")
    fig, axes = plt.subplots(2, len(survey.ells), figsize=(4 * len(survey.ells), 6), sharex=True, constrained_layout=True)
    for row, (label_gic, gic) in enumerate((("geometry", False), ("geometry + gic", True))):
        kw_k = survey.kwargs(ric=False, amr=False, gic=gic)
        t, s2, _ = analytic_shotnoise_template(*fields, **kw_k, include_gic=gic)
        t_comb = combine_regions_weighted_by_normalization(np.asarray(t)[None], np.asarray(s2)[None], norms)
        analytic = t_comb[0][0] / t_comb[1][0]
        entries = [("analytic", analytic, None)]
        for name, runner in (("B", sample_shotnoise_template_linearized), ("C", sample_shotnoise_template_quadratic)):
            res = runner(*fields, key=key, n_real=args.n_real, **kw_k)
            s, cov = shotnoise_template_from_samples(res, norms)
            entries.append((f"option {name}", s, np.sqrt(np.diag(cov)).reshape(s.shape)))
            for label, mask in (("all bins", kmask(survey)), ("k < k_Nyq/2", kmask(survey, 0.5)), ("lowest 3 bins", kmask(survey, low_k_bins=3))):
                chi = chi2_report(s - analytic, cov, f"{label_gic}: option {name} - analytic ({label})", mask, args.n_real)
                results[f"V4 {label_gic} {name} {label}"] = chi
        plot_template(axes[row], survey, entries, title=label_gic)
    axes[0, 0].legend()
    savefig(fig, args.out, "validate_machinery_analytic")
    print("  (small residuals are expected from mass-assignment aliasing of the self-pair term, which the analytic formula treats exactly)")

    header("Summary")
    for key_ in ("V3", "V2", "A->C"):
        print(f"  {key_}: {'PASS' if results[key_] else 'FAIL'}")


if __name__ == "__main__":
    main()
