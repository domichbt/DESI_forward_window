"""
Post-processing of window realizations: averaging, removal of the self-pair noise of the mock data, control variate.

The windows of :py:mod:`desiwinds.window` are derivatives of forward modelled spectra. The mock data being randoms weighted by
``1 + delta`` (:py:func:`desiwinds.forward.mock_survey_catalog`), their self-pairs carry a white noise proportional to the
variance of ``delta``, hence linear in the input theory: it goes through the integral constraints like the Poisson shot noise
and ends up in the window, while Poisson-sampled data have no such term. :py:func:`correct_selfpairs` removes it.

Per window run, the order is:

1. :py:func:`average_windows` over the realizations (windows and, if computed, shot-noise Jacobians);
2. :py:func:`correct_selfpairs` (or :py:func:`correct_selfpairs_analytic` without the Jacobians) on every forward modelled
   window, including the geometry one used in the control variate;
3. :py:func:`apply_control_variate` with the analytic geometry window, which has no self-pairs.
"""

import numpy as np
from jaxpower import MeshAttrs
from lsstypes import Mesh2SpectrumPoles, WindowMatrix
from lsstypes.utils import matrix_spline_interp

from .window import get_basis_vectors, get_fiducial_theory


def average_windows(windows: list[WindowMatrix]) -> WindowMatrix:
    """Average of window realizations sharing the same observable and theory."""
    return windows[0].clone(value=np.mean([np.asarray(window.value()) for window in windows], axis=0))


def correct_selfpairs(
    window: WindowMatrix, shotnoise_jacobian: WindowMatrix, data_shotnoise_jacobian: WindowMatrix, whitenoise_template: Mesh2SpectrumPoles
) -> WindowMatrix:
    """
    Remove the self-pair noise of the mock data from ``window``, with the shot-noise Jacobians of the same run.

    The window should be the derivative of a noise-free spectrum. With ``R`` the raw power (before shot-noise subtraction), the
    corrected window is the derivative of ``R - data_shotnoise * whitenoise_template``::

        window + shotnoise_jacobian - outer(whitenoise_template, data_shotnoise_jacobian)

    Parameters
    ----------
    window : WindowMatrix
        Window from :py:func:`desiwinds.window.get_window_spikes` or :py:func:`desiwinds.window.get_window_basis`.
    shotnoise_jacobian : WindowMatrix
        Derivative of the estimated shot noise (``num_shotnoise`` divided by the normalization) of the same run, same shape as ``window``.
    data_shotnoise_jacobian : WindowMatrix
        Derivative of the data-only monopole shot noise of the same run, one row by the window columns.
    whitenoise_template : Mesh2SpectrumPoles
        Response of the estimator and the observational effects to white noise on the data, normalized by its monopole shot
        noise: ``(P_ell + num_shotnoise_ell / norm_ell) / (num_shotnoise_0 / norm_0)`` of white-noise forward models
        (:py:func:`desiwinds.forward.mock_whitenoise`) with the same mesh, painting, binning, weights and effects as the window.
        Must be on the binning of the window observable.

    Returns
    -------
    WindowMatrix
        Corrected window.
    """
    value = np.asarray(window.value())
    shotnoise = np.asarray(shotnoise_jacobian.value())
    data_shotnoise = np.ravel(np.asarray(data_shotnoise_jacobian.value()))
    if shotnoise.shape != value.shape or data_shotnoise.size != value.shape[1]:
        raise ValueError(f"Jacobian shapes {shotnoise.shape} and {data_shotnoise.shape} do not match the window shape {value.shape}.")
    template = _template_on_observable(whitenoise_template, window.observable)
    return window.clone(value=value + shotnoise - np.outer(template, data_shotnoise))


def correct_selfpairs_analytic(
    window: WindowMatrix, whitenoise_template: Mesh2SpectrumPoles, unclustered_data_shotnoise: float, mattrs: MeshAttrs
) -> WindowMatrix:
    """
    Remove the self-pair noise of the mock data from ``window``, for windows computed without the shot-noise Jacobians.

    The self-pair noise is ``unclustered_data_shotnoise * variance``, with ``variance`` the variance of ``delta`` read at the
    particles (compensated CIC, as in :py:func:`desiwinds.forward.mock_survey_catalog`), which only depends on the input
    monopole. Its derivative with respect to each theory entry is :py:func:`get_density_variance_derivative`, and the
    correction removes its response through the estimator and the effects::

        window - outer(whitenoise_template - delta_ell0, unclustered_data_shotnoise * variance_derivative)

    This neglects the dependence of the randoms' shot noise on the theory, and assumes the estimated shot noise is a monopole
    (unit or FKP estimator weights); :py:func:`correct_selfpairs` makes neither approximation.

    Parameters
    ----------
    window : WindowMatrix
        Spike window (theory on bins) or basis window (theory from :py:func:`desiwinds.window.get_basis_theory`).
    whitenoise_template : Mesh2SpectrumPoles
        See :py:func:`correct_selfpairs`.
    unclustered_data_shotnoise : float
        Data-only monopole shot noise of the mock data before clustering (``1 + delta`` weights), *i.e.* the sum of the squared
        data weights over the monopole normalization.
    mattrs : MeshAttrs
        Attributes of the mesh on which the Gaussian mocks are generated.

    Returns
    -------
    WindowMatrix
        Corrected window.
    """
    variance_derivative = get_density_variance_derivative(get_fiducial_theory(window.theory), mattrs)
    if hasattr(window.theory, "types"):
        variance_derivative = get_basis_vectors(window.theory) @ variance_derivative
    template = _template_on_observable(whitenoise_template, window.observable) - _monopole_indicator(window.observable)
    return window.clone(value=np.asarray(window.value()) - np.outer(template, unclustered_data_shotnoise * variance_derivative))


def get_density_variance_derivative(theory: Mesh2SpectrumPoles, mattrs: MeshAttrs) -> np.ndarray:
    """
    Return the derivative of the variance of the Gaussian mock density read at the particles with respect to each entry of ``theory``.

    The mocks paint the theory multipoles on the modes of the mesh ``mattrs`` as top-hat bins (lower edges of the bins, plus the
    last upper edge), and the density is read with a compensated CIC, whose variance boost for a mode is
    ``prod_a (1 - 2/3 sin^2(k_a H_a / 2)) / sinc^4(k_a H_a / 2)``. Only the monopole contributes, so the entries of the other
    multipoles are zero.

    Returns
    -------
    np.ndarray
        One value per theory entry, multipoles concatenated as ``theory.value()``.
    """
    poles = list(theory)  # leaves, also for trees with more labels than ells (e.g. analytic windows)
    edges = np.asarray(poles[0].edges("k"))
    kin = np.append(edges[:, 0], edges[-1, 1])
    meshsize = np.asarray(mattrs.meshsize, dtype=int)
    boxsize, cellsize = np.asarray(mattrs.boxsize, dtype=float), np.asarray(mattrs.cellsize, dtype=float)
    kvec = [2 * np.pi * np.fft.fftfreq(n, d=h) for n, h in zip(meshsize, cellsize, strict=True)]

    def boost(k, h):
        x = k * h / 2
        return (1 - 2 / 3 * np.sin(x) ** 2) / np.sinc(x / np.pi) ** 4

    gy, gz = boost(kvec[1], cellsize[1])[:, None], boost(kvec[2], cellsize[2])[None, :]
    k2yz = kvec[1][:, None] ** 2 + kvec[2][None, :] ** 2
    nbins = kin.size - 1
    derivative = np.zeros(nbins)
    for kx, gx in zip(kvec[0], boost(kvec[0], cellsize[0]), strict=True):  # one slab at a time, to bound memory
        index = np.digitize(np.sqrt(kx**2 + k2yz), kin, right=False) - 1
        inside = (index >= 0) & (index < nbins)
        derivative += np.bincount(index[inside], weights=(gx * gy * gz)[inside], minlength=nbins)
    derivative /= np.prod(boxsize)
    out = np.zeros(theory.value().size)
    offset = 0
    for ell, pole in zip(theory.ells, poles, strict=True):
        if ell == 0:
            out[offset : offset + pole.size] = derivative
        offset += pole.size
    return out


def apply_control_variate(
    effect_window: WindowMatrix,
    geometry_window: WindowMatrix,
    analytic_geometry_window: WindowMatrix,
    theory_interpolation: str = "spline",
) -> WindowMatrix:
    """
    Control variate of a forward modelled window with the analytic geometry window.

    The forward modelled windows with the effects (``effect_window``) and with the geometry only (``geometry_window``, no
    integral constraint) share their realizations, so most of their noise cancels in the difference::

        analytic_geometry_window + (effect_window - geometry_window)

    Both forward modelled windows can be averages over the same seeds, or single realizations.

    Parameters
    ----------
    effect_window, geometry_window : WindowMatrix
        Forward modelled windows, with the same theory.
    analytic_geometry_window : WindowMatrix
        Window from :py:func:`desiwinds.window.get_window_geometry`; for basis windows, projected onto the same basis vectors
        with :py:func:`desiwinds.window.project_window`.
    theory_interpolation : str, optional
        How the difference is mapped onto the theory of ``analytic_geometry_window`` when the theory bins differ: ``"spline"``
        (cubic spline interpolation in k, the theory bins being sampled at their centers) or ``"exact"`` (raise unless the
        theories match). Basis windows must have the same basis vectors. By default ``"spline"``.

    Returns
    -------
    WindowMatrix
        Window with the observable of ``effect_window`` and the theory of ``analytic_geometry_window``.
    """
    difference = np.asarray(effect_window.value()) - np.asarray(geometry_window.value())
    fm_theory, analytic_theory = effect_window.theory, analytic_geometry_window.theory
    if hasattr(fm_theory, "types") or hasattr(analytic_theory, "types"):
        same = hasattr(fm_theory, "types") and hasattr(analytic_theory, "types")
        same = same and np.shape(get_basis_vectors(fm_theory)) == np.shape(get_basis_vectors(analytic_theory))
        if not (same and np.allclose(get_basis_vectors(fm_theory), get_basis_vectors(analytic_theory))):
            raise ValueError("Basis windows must share the same basis vectors; project the analytic window with project_window.")
        mapping = np.eye(difference.shape[1])
    else:
        fm_poles, analytic_poles = list(fm_theory), list(analytic_theory)
        xo, xt = np.asarray(fm_poles[0].coords("k")), np.asarray(analytic_poles[0].coords("k"))
        if list(fm_theory.ells) != list(analytic_theory.ells) or len(fm_poles) != len(analytic_poles):
            raise ValueError(f"Theory multipoles differ: {list(fm_theory.ells)} and {list(analytic_theory.ells)}.")
        if xo.size == xt.size and np.allclose(xo, xt):
            block = np.eye(xo.size)
        elif theory_interpolation == "spline":
            block = matrix_spline_interp(xt=xt, xo=xo)
        elif theory_interpolation == "exact":
            raise ValueError("Theory bins of the forward modelled and analytic windows differ.")
        else:
            raise ValueError(f"Unknown theory_interpolation {theory_interpolation!r}, expected 'spline' or 'exact'.")
        mapping = np.kron(np.eye(len(analytic_poles)), block)
    return effect_window.clone(value=np.asarray(analytic_geometry_window.value()) + difference @ mapping, theory=analytic_theory)


def _template_on_observable(template: Mesh2SpectrumPoles, observable: Mesh2SpectrumPoles) -> np.ndarray:
    """Values of ``template`` on the poles of ``observable``, concatenated; raise if the binnings differ."""
    values = []
    for ell in observable.ells:
        if ell not in template.ells:
            raise ValueError(f"The white-noise template has no ell = {ell}.")
        k, k_template = np.asarray(observable.get(ell).coords("k")), np.asarray(template.get(ell).coords("k"))
        if k.shape != k_template.shape or not np.allclose(k, k_template):
            raise ValueError(f"The white-noise template is not on the binning of the window observable (ell = {ell}); match it first.")
        values.append(np.asarray(template.get(ell).value()).real)
    return np.concatenate(values)


def _monopole_indicator(observable: Mesh2SpectrumPoles) -> np.ndarray:
    """1 on the monopole entries of ``observable``, 0 elsewhere."""
    return np.concatenate([np.full(observable.get(ell).size, float(ell == 0)) for ell in observable.ells])
