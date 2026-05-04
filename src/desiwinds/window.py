"""Window computations."""

import os
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from jaxpower import BinMesh2SpectrumPoles, RealMeshField, compute_mesh2_spectrum_window
from lsstypes import Mesh2SpectrumPoles, WindowMatrix
from tqdm import tqdm


def get_window_geometry(
    selection1: RealMeshField,
    selection2: RealMeshField | None,
    theory_edges: np.ndarray,
    theory_ells: tuple,
    binner: BinMesh2SpectrumPoles,
    norm: jnp.ndarray,
    los: Literal["local", "x", "y", "z", "firstpoint", "endpoint"] = "local",
    flags: tuple[str, ...] = (),
    pbar: bool = True,
    **kwargs,
) -> WindowMatrix:
    """
    Get the exact window matrix corresponding to the ``selection`` geometry through analytical derivation.

    Parameters
    ----------
    selection1 : RealMeshField
        Mesh describing the geometry.
    selection2 : RealMeshField
        Optional second mesh also describing the geometry, to remove any noise due to the autocorrelation of ``selection1``.
    theory_edges : np.ndarray
        Input bin edges, can be an :py:class:`lsstypes.ObservableTree` object with stored bins.
    theory_ells : tuple
        Input multipoles, can use ``theory_edges``'s ``ells`` if it is an :py:class:`lsstypes.ObservableTree`.
    binner : BinMesh2SpectrumPoles
        Binning operator to compute the output power spectrum.
    norm : jnp.ndarray
        The same norm as for the mock survey P(k) computation (*i.e.*, with CIC resampling and a cellsize of 10.)
    los : Literal["local", "x", "y", "z", "firstpoint", "endpoint"], optional
        Line of sight, by default "local"
    flags : tuple[str], optional
        Options for the computation, see :py:func:`jaxpower.compute_mesh2_spectrum_window` documentation for details. By default ().
    pbar : bool, optional
        Whether to show a progress bar, by default True
    kwargs
        Extra parameters for :py:func:`jaxpower.compute_mesh2_spectrum_window`.

    Returns
    -------
    WindowMatrix
        Window matrix accounting for the geometry of the survey.

    Notes
    -----
    This is simply a thin wrapper around :py:func:`jaxpower.compute_mesh2_spectrum_window`.
    """
    if selection2:
        return compute_mesh2_spectrum_window(
            selection1, selection2, edgesin=theory_edges, ellsin=theory_ells, bin=binner, norm=norm, los=los, flags=flags, pbar=pbar, **kwargs
        )
    else:
        return compute_mesh2_spectrum_window(
            selection1, edgesin=theory_edges, ellsin=theory_ells, bin=binner, norm=norm, los=los, flags=flags, pbar=pbar, **kwargs
        )


def get_window_spikes(
    mock_survey: Callable,
    theory: Mesh2SpectrumPoles,
    nreal: int = 10,
    seeds: list[int] | None = None,
    batch_size: int = 1,
    mock_survey_args: tuple = (),
    mock_survey_kwargs: dict | None = None,
    static_argnums: int | tuple[int, ...] | None = None,
    static_argnames: list[str] | None = None,
    tmpdir: str | os.PathLike | None = None,
    survey_names: list[str] | None = None,
):
    """
    Estimate the response (window matrix component) of a given observation forward modelling ``mock_survey`` for some fiducial theory input ``theory``.

    The estimation is done by injecting individual spikes of the theory power spectrum in mocks, forward modeling the selection effects and taking the derivative of the response at fiducial ``theory``.

    Parameters
    ----------
    mock_survey : Callable
        Selection to be forward modeled on ``theory``. Signature should be ``observe(theory, seed, **kwargs) -> list[Mesh2SpectrumPoles]``. Must support jax forward differentation and be jittable.
    theory : lsstypes.BinMesh2SpectrumPoles
        Fiducial theoretical power spectrum for the Jacobian estimation.
    nreal : int, optional
        Number of realizations for the average computation, by default 10
    seeds : list[int] | None, optional
        Individual random integer seeds for the `nreal` realizations. If ``None``, defaults to ``2 * i + 3``.
    batch_size : int, optional
        How many spikes to run in parallel, by default 4.
    mock_survey_args : tuple, optional
        Positional arguments for the ``mock_survey`` function.
    mock_survey_kwargs : dict, optional
        Additional keyword arguments for the ``mock_survey`` function, aside from ``theory`` and ``seed``.
    static_argnums: int | tuple[int, ...] | None, optional
        List of argument indices in ``mock_survey_kwargs`` that should passed to ``static_argnums`` when JITting.
    static_argnames: list[str] | None, optional
        List of arguments in ``mock_survey_kwargs`` that should passed to ``static_argnames`` when JITting.
    tmpdir: str | os.PathLike | None
        Directory where individual realizations can be saved as soon as they are computed, to avoid losing them to a timeout. Files will be overwritten and the default name is ``f"{seed:010d}.h5"``.
    survey_names: list[str] | None
        Name of the subdirectory for the power spectra output by ``mock_survey``.

    Returns
    -------
    tuple[list[lsstypes.WindowMatrix], list[list[lsstypes.WindowMatrix]]]
        The average window matrices over ``nreal`` realizations and the individual realizations.

    Notes
    -----
    The individual realizations are returned as a list of length ``nreal``, each entry being a list of ``lsstypes.WindowMatrix`` of length equal to the number of output power spectra of ``mock_survey``.
    """
    mock_survey_args = mock_survey_args or ()
    mock_survey_kwargs = mock_survey_kwargs or {}
    static_argnames = static_argnames or []
    static_argnames = [*static_argnames, "mock_surveys"]
    if tmpdir is not None:
        tmpdir = Path(tmpdir)

    # Get some empty theory and observable to use their shapes when creating the window matrix
    observables = mock_survey(*mock_survey_args, theory=theory, seed=jax.random.key(42), **mock_survey_kwargs)
    observables = [observable.clone(value=0.0 * observable.value()) for observable in observables]

    survey_names = survey_names or [f"survey_{idx_survey:02d}" for idx_survey in range(len(observables))]

    theory_zeros = jnp.zeros_like(theory.value())
    # JIT the function retrieving the window component
    get_window = jax.jit(
        get_windows_component,
        static_argnums=static_argnums,
        static_argnames=static_argnames,
    )
    if seeds is None:
        seeds = [2 * imock + 3 for imock in range(nreal)]

    # Initialize a list of windows to fill later
    windows = [[None for j in range(len(observables))] for i in range(nreal)]

    # Given batch size, how many loops do we run?
    nsplits = (theory.size + batch_size - 1) // batch_size
    for imock in tqdm(range(nreal), desc="Realization", disable=(jax.process_index() != 0)):
        seed = jax.random.key(seeds[imock])
        for isplit in tqdm(range(nsplits), desc=f"Iterations (realization {imock})", disable=(jax.process_index() != 0)):
            islice = isplit * theory_zeros.size // nsplits, (isplit + 1) * theory_zeros.size // nsplits
            spikes = jnp.array([theory_zeros.at[ii].set(1.0) for ii in range(*islice)])
            spectra = [
                spectrum.T
                for spectrum in get_window(
                    *mock_survey_args, fiducial_theory=theory, injected_theory=spikes, seed=seed, mock_surveys=mock_survey, **mock_survey_kwargs
                )
            ]
            for idx_spectrum, spectrum in enumerate(spectra):
                if windows[imock][idx_spectrum] is None:
                    windows[imock][idx_spectrum] = np.zeros((spectrum.shape[0], theory.size))
                windows[imock][idx_spectrum][..., slice(*islice)] = spectrum
        for idx_window, window in enumerate(windows[imock]):
            windows[imock][idx_window] = WindowMatrix(value=window, theory=theory, observable=observables[idx_window])
            if (tmpdir is not None) and jax.process_index() == 0:
                windows[imock][idx_window].write(tmpdir / survey_names[idx_window] / f"{seeds[imock]:010d}.h5")
    window = [
        WindowMatrix(value=np.mean([window[idx_survey].value() for window in windows], axis=0), theory=theory, observable=observables[idx_survey])
        for idx_survey in range(len(observables))
    ]
    return window, windows


def get_windows_spikes(
    mock_surveys: Callable,
    theory: Mesh2SpectrumPoles,
    nreal: int = 10,
    seeds: list[int] | None = None,
    batch_size: int = 1,
    mock_surveys_args: tuple | None = None,
    mock_surveys_kwargs: dict | None = None,
    static_argnums: int | tuple[int, ...] | None = None,
    static_argnames: list[str] | None = None,
    tmpdir: str | os.PathLike | None = None,
    survey_names: list[str] | None = None,
):
    """
    Estimate the response (window matrix component) of a given set of observation forward modellings ``mock_surveys`` for some fiducial theory input ``theory``.

    The estimation is done by injecting individual spikes of the theory power spectrum in mocks, forward modeling the selection effects and taking the derivative of the response at fiducial ``theory``.

    Compared to :py:fun:`get_window_spikes` (notice the lack of `s`), this is designed to get lots of power spectra corresponding to the same mesh but to different observational effects. This way, one can easily apply a control variate on the more complex modellings.

    Parameters
    ----------
    mock_surveys : Callable
        Selections to be forward modeled on ``theory``. Signature should be ``observe(theory, seed, **kwargs) -> list[Mesh2SpectrumPoles]``. Must support jax forward differentation and be jittable.
    theory : lsstypes.BinMesh2SpectrumPoles
        Fiducial theoretical power spectrum for the Jacobian estimation.
    nreal : int, optional
        Number of realizations for the average computation, by default 10
    seeds : list[int] | None, optional
        Individual random integer seeds for the `nreal` realizations. If ``None``, defaults to ``2 * i + 3``.
    batch_size : int, optional
        How many spikes to run in parallel, by default 4.
    mock_surveys_args : tuple, optional
        Positional arguments for the ``mock_surveys`` function.
    mock_surveys_kwargs : dict, optional
        Keyword arguments for the ``mock_surveys`` function, aside from ``theory`` and ``seed``.
    static_argnums: int | tuple[int, ...] | None, optional
        List of argument indices in ``mock_surveys_args`` that should passed to ``static_argnums`` when JITting.
    static_argnames: list[str] | None, optional
        List of arguments in ``mock_surveys_kwargs`` that should passed to ``static_argnames`` when JITting.
    tmpdir: str | os.PathLike | None
        Directory where individual realizations can be saved as soon as they are computed, to avoid losing them to a timeout. Files will be overwritten and the default name is ``f"{seed:010d}.h5"``.
    survey_names: list[str] | None
        Name of the subdirectories for each survey of ``mock_surveys``.

    Returns
    -------
    tuple[list[lsstypes.WindowMatrix], list[list[lsstypes.WindowMatrix]]]
        The average window matrices over ``nreal`` realizations (for each set of observationnal effects) and the individual realizations (shape (nreal, nsurveys)).
    """
    mock_surveys_args = mock_surveys_args or ()
    mock_surveys_kwargs = mock_surveys_kwargs or {}
    static_argnames = static_argnames or []
    static_argnames = [*static_argnames, "mock_surveys"]
    if tmpdir is not None:
        tmpdir = Path(tmpdir)

    # Get some empty theory and observable to use their shapes when creating the window matrix
    observables = mock_surveys(*mock_surveys_args, theory=theory, seed=jax.random.key(42), **mock_surveys_kwargs)
    nsurveys = len(observables)
    observable = observables[0].clone(value=0.0 * observables[0].value())
    theory_zeros = jnp.zeros_like(theory.value())
    # JIT the function retrieving the window component
    get_windows = jax.jit(
        get_windows_component,
        static_argnums=static_argnums,
        static_argnames=static_argnames,
    )
    if seeds is None:
        seeds = [2 * imock + 3 for imock in range(nreal)]

    # Initialize a list of windows to fill later
    windows = [[None for j in range(nsurveys)] for i in range(nreal)]

    # Given batch size, how many loops do we run?
    nsplits = (theory.size + batch_size - 1) // batch_size
    for imock in tqdm(range(nreal), desc="Realization", disable=(jax.process_index() != 0)):
        seed = jax.random.key(seeds[imock])
        for isplit in tqdm(range(nsplits), desc=f"Iterations (realization {imock})", disable=(jax.process_index() != 0)):
            islice = isplit * theory_zeros.size // nsplits, (isplit + 1) * theory_zeros.size // nsplits
            spikes = jnp.array([theory_zeros.at[ii].set(1.0) for ii in range(*islice)])
            spectra = [
                wd.T
                for wd in get_windows(
                    *mock_surveys_args,
                    fiducial_theory=theory,
                    injected_theory=spikes,
                    seed=seed,
                    mock_surveys=mock_surveys,
                    **mock_surveys_kwargs,
                )
            ]
            for isurvey in range(nsurveys):
                if windows[imock][isurvey] is None:
                    windows[imock][isurvey] = np.zeros((spectra[isurvey].shape[0], theory.size))
                windows[imock][isurvey][..., slice(*islice)] = spectra[isurvey]
        for isurvey in range(nsurveys):
            windows[imock][isurvey] = WindowMatrix(value=windows[imock][isurvey], theory=theory, observable=observable)
        if (tmpdir is not None) and jax.process_index() == 0:
            for isurvey in range(nsurveys):
                windows[imock][isurvey].write(tmpdir / survey_names[isurvey] / f"{seeds[imock]:010d}.h5")
    windows_avg = [
        WindowMatrix(value=np.mean([windows[imock][isurvey].value() for imock in range(nreal)], axis=0), theory=theory, observable=observable)
        for isurvey in range(nsurveys)
    ]
    return windows_avg, windows


def get_windows_component(*mock_surveys_args, injected_theory, fiducial_theory, seed, mock_surveys, **mock_surveys_kw):
    """By definition, the window is the derivative of the observed power spectrum relative to the input theory evaluated at some fiducial theory value."""

    # Get a mock-based observed P(k) from a theory power spectrum that looks like "input_theory" and concatenate the poles
    def get_responses(input_value):
        # theory, observe -> global
        responses = mock_surveys(*mock_surveys_args, theory=fiducial_theory.clone(value=input_value), seed=seed, **mock_surveys_kw)
        return [jnp.concatenate(response.value(concatenate=False)).real for response in responses]

    # Get the Jacobian of this, differentated wrt argument `input_value`, evaluated in fiducial_value, dot product with s
    def derivative(s):
        # fiducial_value -> global
        return jax.jvp(get_responses, primals=(jnp.concatenate(fiducial_theory.value(concatenate=False)),), tangents=(s,))[1]
        # return get_responses(jnp.concatenate(fiducial_theory.value(concatenate=False)) * s)

    return jax.vmap(derivative)(injected_theory)
