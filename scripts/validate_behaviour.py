"""
Checks V5, V6, V9, V10 of the math: expected behaviour of the template.

* V5  high-k normalization: ``s_0 -> 1`` and ``s_{l>0} -> 0`` for ``k >> 1/L`` (options B and C),
* V6  low-k structure: ``s_0 -> 0`` as ``k -> 0``, RIC/AMR depart more from 1 than geometry + global IC alone, ``l > 0`` only at low ``k``,
* V9  convergence: errors scale as ``1/sqrt(N_real)``, estimates from independent halves agree,
* V10 mesh: doubling the mesh leaves ``s_l`` unchanged for ``k < k_Nyq/2`` (paired draws). The mass-assignment settings are fixed, as in the forward model.

Run: ``python scripts/validate_behaviour.py [--cellsize 200 --n-real 32]``
"""

import matplotlib

matplotlib.use("Agg")
import jax
import matplotlib.pyplot as plt
import numpy as np
from jaxpower import MeshAttrs
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
    sample_shotnoise_template_linearized,
    sample_shotnoise_template_quadratic,
    shotnoise_template_from_samples,
    shotnoise_template_uncertainty,
)

PIPELINES = {
    "geometry": dict(ric=False, amr=False, gic=False),
    "geometry + GIC": dict(ric=False, amr=False, gic=True),
    "RIC": dict(ric=True, amr=False),
    "RIC + AMR": dict(ric=True, amr=True),
}


def weighted_mean_pull(res, norms, rows, bins):
    """Mean of ``s`` over the selected ``bins`` for multipole indices ``rows``, in units of its jackknife error (full covariance of the bins)."""
    s, cov = shotnoise_template_from_samples(res, norms)
    out = []
    for row in rows:
        idx = np.array([row * s.shape[1] + b for b in bins])
        mean = s[row, bins].mean()
        err = np.sqrt(cov[np.ix_(idx, idx)].sum()) / len(bins)
        out.append((mean, err))
    return out


def main():
    parser = base_parser(__doc__)
    args = parser.parse_args()
    survey = random_half_survey(args)
    nk, ells = len(survey.k), survey.ells
    n, fields, norms, key = args.n_real, survey.fkp_fields, survey.i0, jax.random.key(args.seed)

    # ------------------------------------------------------------------ run everything once
    header("Running options B (all pipelines) and C (RIC + AMR)")
    results = {("B", name): sample_shotnoise_template_linearized(*fields, key=key, n_real=n, **survey.kwargs(**kw)) for name, kw in PIPELINES.items()}
    results[("C", "RIC + AMR")] = sample_shotnoise_template_quadratic(*fields, key=key, n_real=n, **survey.kwargs(**PIPELINES["RIC + AMR"]))

    fig, axes = plt.subplots(1, len(ells), figsize=(4.2 * len(ells), 3.6), constrained_layout=True)
    entries = [(f"{opt}: {name}", *(lambda r: (r[0], np.sqrt(np.diag(r[1])).reshape(r[0].shape)))(shotnoise_template_from_samples(res, norms))) for (opt, name), res in results.items()]
    plot_template(axes, survey, entries)
    axes[0].legend(fontsize=7)
    savefig(fig, args.out, "validate_behaviour_templates")

    # ------------------------------------------------------------------ V5
    header("V5: high-k normalization")
    knyq = float(np.min(np.pi / np.asarray(survey.binner.mattrs.cellsize)))
    for label, sel in (("0.25 < k/k_Nyq < 0.5", (survey.k > 0.25 * knyq) & (survey.k < 0.5 * knyq)), ("k/k_Nyq > 0.5 (mesh effects)", survey.k > 0.5 * knyq)):
        bins = np.where(sel)[0]
        print(f"  {label}: {len(bins)} bins")
        for label_key, res in results.items():
            (m0, e0), *rest = weighted_mean_pull(res, norms, range(len(ells)), bins)
            line = f"    {label_key[0]} {label_key[1]:16s} <s_0 - 1> = {m0 - 1:+.4f} +- {e0:.4f} ({(m0 - 1) / e0:+.1f} sigma);"
            for ell, (m, e) in zip(ells[1:], rest, strict=True):
                line += f" <s_{ell}> = {m:+.4f} +- {e:.4f} ({m / e:+.1f} sigma);"
            print(line)
    print("  s_0 -> 1 is exact at all orders for k >> 1/L, where L is the *radial extent of the survey* for RIC (here ~1000 Mpc/h, i.e. k >~ 0.01 h/Mpc), so the")
    print("  first band can legitimately fall below 1 at coarse cells. Residual s_{l>0} at high k is Monte Carlo noise, strongly correlated between adjacent bins:")
    print("  it needs O(10^2) realizations to average down (see validate_machinery.py, and run with a larger --n-real).")

    # ------------------------------------------------------------------ V6
    header("V6: low-k structure")
    for name in PIPELINES:
        res = results[("B", name)]
        s, cov = shotnoise_template_from_samples(res, norms)
        err = np.sqrt(np.diag(cov)).reshape(s.shape)
        print(f"  {name:16s} s_0(lowest bins) = {np.round(s[0, :3], 3)} +- {np.round(err[0, :3], 3)} | max|s_2| = {np.abs(s[1]).max():.3f} at k = {survey.k[np.abs(s[1]).argmax()]:.4f}")
    dev = {name: np.abs(1 - shotnoise_template_from_samples(results[("B", name)], norms)[0][0][: nk // 3]).sum() for name in PIPELINES}
    print("  integrated |1 - s_0| over the lowest third of k: " + ", ".join(f"{k}: {v:.3f}" for k, v in dev.items()))
    ok = dev["RIC + AMR"] > dev["geometry + GIC"] > dev["geometry"]
    print(f"  departure(RIC+AMR) > departure(geometry+GIC) > departure(geometry): {'PASS' if ok else 'CHECK'}")
    s_ra = shotnoise_template_from_samples(results[("B", "RIC + AMR")], norms)[0]
    for iell, ell in list(enumerate(ells))[1:]:
        print(f"  RIC+AMR, l={ell}: rms(s) over low third = {np.sqrt(np.mean(s_ra[iell, : nk // 3] ** 2)):.3f}, over high third = {np.sqrt(np.mean(s_ra[iell, -(nk // 3) :] ** 2)):.3f}")

    # ------------------------------------------------------------------ V9
    header("V9: convergence with the number of realizations (option B, RIC + AMR)")
    res = results[("B", "RIC + AMR")]
    if n >= 8:
        sub = n // 4
        e_full = shotnoise_template_uncertainty(res, norms)
        e_sub = ratio_error(res, np.arange(sub), norms, (len(ells), nk))
        mask = e_full > 0
        expected = np.sqrt(sub / n)
        print(f"  median err(N={n}) / err(N={sub}) = {np.median(e_full[mask] / e_sub[mask]):.3f} (expect {expected:.3f})")
        half = n // 2
        a = ratio(res["power_spectrum_response"], res["shotnoise_response"], np.arange(half), norms)
        b = ratio(res["power_spectrum_response"], res["shotnoise_response"], np.arange(half, 2 * half), norms)
        _, cov_a = jackknife_stat(half, lambda idx: ratio(res["power_spectrum_response"][:half], res["shotnoise_response"][:half], idx, norms))
        _, cov_b = jackknife_stat(half, lambda idx: ratio(res["power_spectrum_response"][half : 2 * half], res["shotnoise_response"][half : 2 * half], idx, norms))
        chi2_report(a - b, cov_a + cov_b, "independent halves differ", kmask(survey), half)
    else:
        print("  need --n-real >= 8")

    # ------------------------------------------------------------------ V10
    header("V10: mesh settings (option B, RIC + AMR, same draws)")
    reference = results[("B", "RIC + AMR")]

    def paired(label, other):
        def diff(idx):
            return ratio(other["power_spectrum_response"], other["shotnoise_response"], idx, norms)[:, :nk] - ratio(reference["power_spectrum_response"], reference["shotnoise_response"], idx, norms)

        d, cov = jackknife_stat(n, diff)
        chi2_report(d, cov, label, kmask(survey, 0.5), n)
        return d

    attrs = survey.fkp_fields[0].attrs
    fine = random_half_survey(args, mattrs=MeshAttrs(boxsize=np.asarray(attrs.boxsize), boxcenter=np.asarray(attrs.boxcenter), meshsize=2 * np.asarray(attrs.meshsize)))
    n_k = min(len(fine.k), nk)
    assert np.allclose(fine.k[:n_k], survey.k[:n_k]), "fine and coarse binners must share their k bins"
    paired("2x finer mesh (k < k_Nyq/2 of the coarse mesh)", sample_shotnoise_template_linearized(*fine.fkp_fields, key=key, n_real=n, **fine.kwargs(ric=True, amr=True)))
    print("  (jackknife covariances of the paired differences; large chi2 at k close to the coarse Nyquist is excluded by construction)")


def ratio_error(res, idx, norms, shape):
    """Jackknife errors of the ratio of means over the first realizations ``idx``."""
    _, cov = jackknife_stat(len(idx), lambda i: ratio(res["power_spectrum_response"][idx], res["shotnoise_response"][idx], i, norms))
    return np.sqrt(np.diag(cov)).reshape(shape)


if __name__ == "__main__":
    main()
