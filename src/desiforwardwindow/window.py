"""Window computations."""

from collections.abc import Callable
from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from jaxpower import BinMesh2SpectrumPoles, RealMeshField, compute_mesh2_spectrum_window, compute_normalization
from lsstypes import Mesh2SpectrumPoles, WindowMatrix
from tqdm import tqdm


def get_window_geometry(
    selection: RealMeshField,
    theory_edges: np.ndarray,
    theory_ells: tuple,
    binner: BinMesh2SpectrumPoles,
    los: Literal["local", "x", "y", "z", "firstpoint", "endpoint"] = "local",
    flags: tuple[str] = (),
    pbar: bool = True,
    **kwargs,
) -> WindowMatrix:
    """
    Get the exact window matrix corresponding to the ``selection`` geometry through analytical derivation.

    Parameters
    ----------
    selection : RealMeshField
        Mesh describing the geometry.
    theory_edges : np.ndarray
        Input bin edges, can be an :py:class:`lsstypes.ObservableTree` object with stored bins.
    theory_ells : tuple
        Input multipoles, can use ``theory_edges``'s ``ells`` if it is an :py:class:`lsstypes.ObservableTree`.
    binner : BinMesh2SpectrumPoles
        Binning operator to compute the output power spectrum.
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
    norm = compute_normalization(selection, selection, bin=binner)
    return compute_mesh2_spectrum_window(selection, edgesin=theory_edges, ellsin=theory_ells, bin=binner, norm=norm, los=los, flags=flags, pbar=pbar, **kwargs)


def get_window_spikes(
    mock_survey: Callable,
    theory: Mesh2SpectrumPoles,
    nreal: int = 10,
    seeds: list[jnp.ndarray] | None = None,
    batch_size: int = 1,
    mock_survey_kw: dict | None = None,
    unhashable: list[str] | None = None,
):
    """
    Estimate the response (window matrix component) of a given observation forward modelling ``mock_survey`` for some fiducial theory input ``theory``.

    The estimation is done by injecting individual spikes of the theory power spectrum in mocks, forward modeling the selection effects and taking the derivative of the response at fiducial ``theory``.

    Parameters
    ----------
    mock_survey : Callable
        Selection to be forward modeled on ``theory``. Signature should be ``observe(theory, seed, **kwargs) -> Mesh2SpectrumPoles``. Must support jax forward differentation and be jittable.
    theory : lsstypes.BinMesh2SpectrumPoles
        Fiducial theoretical power spectrum for the Jacobian estimation.
    nreal : int, optional
        Number of realizations for the average computation, by default 10
    seeds : list[jnp.ndarray] | None, optional
        Individual random keys for the `nreal` realizations. If ``None``, defaults to ``2 * i + 3``.
    batch_size : int, optional
        How many spikes to run in parallel, by default 4.
    mock_survey_kw : dict, optional
        Additional keyword arguments for the ``mock_survey`` function, aside from ``theory`` and ``seed``.


    Returns
    -------
    tuple[lsstypes.WindowMatrix, list[lsstypes.WindowMatrix]]
        The average window matrix over ``nreal`` realizations and the individual realizations.
    """
    mock_survey_kw = mock_survey_kw or {}
    unhashable = unhashable or []

    static_argnames = ["mock_survey", *(mock_survey_kw.keys() - set(unhashable))]
    print(static_argnames)

    # Initialize a list of windows to fill later
    windows = [None for i in range(nreal)]

    # Get some empty theory and observable to use their shapes when creating the window matrix
    observable = mock_survey(theory, seed=jax.random.key(42), **mock_survey_kw)
    observable = observable.clone(value=0.0 * observable.value())
    theory_zeros = jnp.zeros_like(theory.value())

    # JIT the function retrieving the window component
    get_window = jax.jit(
        partial(get_window_component, fiducial_theory=theory),
        static_argnames=static_argnames,
    )

    if seeds is None:
        seeds = [jax.random.key(2 * imock + 3) for imock in range(nreal)]

    # Given batch size, how many loops do we run?
    nsplits = (theory.size + batch_size - 1) // batch_size
    for isplit in tqdm(range(nsplits)):
        islice = isplit * theory_zeros.size // nsplits, (isplit + 1) * theory_zeros.size // nsplits
        spikes = jnp.array([theory_zeros.at[ii].set(1.0) for ii in range(*islice)])
        for imock in range(nreal):
            spectrum = get_window(injected_theory=spikes, mock_survey=mock_survey, seed=seeds[imock], **mock_survey_kw).T
            if windows[imock] is None:
                windows[imock] = np.zeros((spectrum.shape[0], theory.size))
            windows[imock][..., slice(*islice)] = spectrum
    windows = [WindowMatrix(value=window, theory=theory, observable=observable) for window in windows]
    window = WindowMatrix(value=np.mean([window.value() for window in windows], axis=0), theory=theory, observable=observable)
    return window, windows


def get_window_component(injected_theory, fiducial_theory, mock_survey, **mock_survey_kw):
    """By definition, the window is the derivative of the observed power spectrum relative to the input theory evaluated at some fiducial theory value."""

    # Get a mock-based observed P(k) from a theory power spectrum that looks like "input_theory" and concatenate the poles
    def get_response(input_value):
        # theory, observe -> global
        response = mock_survey(fiducial_theory.clone(value=input_value), **mock_survey_kw)
        return jnp.concatenate(response.value(concatenate=False)).real

    # Get the Jacobian of this, differentated wrt argument `input_value`, evaluated in fiducial_value, dot product with s
    def derivative(s):
        # fiducial_value -> global
        return jax.jvp(get_response, primals=(fiducial_theory.value(),), tangents=(s,))[1]

    return jax.vmap(derivative)(injected_theory)
