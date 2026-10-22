"""Window computations."""

import os
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from jaxpower import BinMesh2SpectrumPoles, RealMeshField, compute_mesh2_spectrum_window
from lsstypes import Mesh2SpectrumPole, Mesh2SpectrumPoles, ObservableLeaf, ObservableTree, WindowMatrix
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
    return_shotnoise_jacobians: bool = False,
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
    return_shotnoise_jacobians: bool, optional
        If ``True``, also estimate the Jacobians of the shot noise, needed to remove the self-pair noise of the mock data from the
        window (see :py:func:`desiwinds.post.correct_selfpairs`). ``mock_survey`` must then accept ``return_data_shotnoise=True``,
        as :py:func:`desiwinds.forward.mock_survey_catalog`. By default ``False``.

    Returns
    -------
    tuple[list[lsstypes.WindowMatrix], list[list[lsstypes.WindowMatrix]]]
        The average window matrices over ``nreal`` realizations and the individual realizations.
        If ``return_shotnoise_jacobians``, both are dictionaries with keys ``"window"``, ``"shotnoise_jacobian"`` and
        ``"data_shotnoise_jacobian"``, each holding such a list (see :py:func:`get_window_basis`).

    Notes
    -----
    The individual realizations are returned as a list of length ``nreal``, each entry being a list of ``lsstypes.WindowMatrix`` of length equal to the number of output power spectra of ``mock_survey``.
    """
    theory = _real_theory(theory)
    n = theory.value().size
    return _compute_window_columns(
        mock_survey,
        theory,
        injected_theory=jnp.eye(n, dtype=theory.value().dtype),
        window_theory=theory,
        nreal=nreal,
        seeds=seeds,
        batch_size=batch_size,
        mock_survey_args=mock_survey_args,
        mock_survey_kwargs=mock_survey_kwargs,
        static_argnums=static_argnums,
        static_argnames=static_argnames,
        tmpdir=tmpdir,
        survey_names=survey_names,
        common_observable=False,
        return_shotnoise_jacobians=return_shotnoise_jacobians,
    )


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
    return_shotnoise_jacobians: bool = False,
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
    return_shotnoise_jacobians: bool, optional
        If ``True``, also estimate the Jacobians of the shot noise, see :py:func:`get_window_spikes`. By default ``False``.

    Returns
    -------
    tuple[list[lsstypes.WindowMatrix], list[list[lsstypes.WindowMatrix]]]
        The average window matrices over ``nreal`` realizations (for each set of observationnal effects) and the individual realizations (shape (nreal, nsurveys)).
        If ``return_shotnoise_jacobians``, dictionaries of such lists, see :py:func:`get_window_spikes`.
    """
    theory = _real_theory(theory)
    n = theory.value().size
    return _compute_window_columns(
        mock_surveys,
        theory,
        injected_theory=jnp.eye(n, dtype=theory.value().dtype),
        window_theory=theory,
        nreal=nreal,
        seeds=seeds,
        batch_size=batch_size,
        mock_survey_args=mock_surveys_args,
        mock_survey_kwargs=mock_surveys_kwargs,
        static_argnums=static_argnums,
        static_argnames=static_argnames,
        tmpdir=tmpdir,
        survey_names=survey_names,
        common_observable=True,
        return_shotnoise_jacobians=return_shotnoise_jacobians,
    )


def get_window_basis(
    mock_survey: Callable,
    theory: Mesh2SpectrumPoles,
    basis_vectors: np.ndarray,
    nreal: int = 10,
    seeds: list[int] | None = None,
    batch_size: int = 1,
    mock_survey_args: tuple = (),
    mock_survey_kwargs: dict | None = None,
    static_argnums: int | tuple[int, ...] | None = None,
    static_argnames: list[str] | None = None,
    tmpdir: str | os.PathLike | None = None,
    survey_names: list[str] | None = None,
    return_shotnoise_jacobians: bool = False,
    basis_names: list[str] | None = None,
):
    """
    Estimate the response of the observation forward modelling ``mock_survey`` to the theory directions ``basis_vectors``, at the fiducial ``theory``.

    Same as :py:func:`get_window_spikes`, but each window column is the derivative of the observed power spectrum along one basis
    vector (a power spectrum shape, with any support) instead of along one theory bin. The derivative being linear in the
    direction, a column equals, realization by realization, the spike window times the basis vector; the cost scales with the
    number of basis vectors instead of the number of theory bins.

    Parameters
    ----------
    mock_survey : Callable
        Selection to be forward modeled on ``theory``, see :py:func:`get_window_spikes`.
    theory : lsstypes.Mesh2SpectrumPoles
        Fiducial theoretical power spectrum, at which the derivative is taken. All multipoles must share the same k bins, which
        the Gaussian mocks paint as top-hat bins.
    basis_vectors : np.ndarray
        Basis vectors on the bins of ``theory``, of shape ``(n_vectors, n_theory)`` with the multipoles concatenated as in
        ``theory.value()``, or ``(n_vectors, n_ells, n_k)``.
    nreal, seeds, batch_size, mock_survey_args, mock_survey_kwargs, static_argnums, static_argnames, tmpdir, survey_names, return_shotnoise_jacobians
        See :py:func:`get_window_spikes`.
    basis_names : list[str] | None, optional
        Names of the basis vectors, stored with them (see :py:func:`get_basis_theory`).

    Returns
    -------
    tuple[list[lsstypes.WindowMatrix], list[list[lsstypes.WindowMatrix]]]
        The average window matrices and the individual realizations, see :py:func:`get_window_spikes`. Their theory is
        :py:func:`get_basis_theory`, which stores the basis vectors and the fiducial ``theory``.
        If ``return_shotnoise_jacobians``, dictionaries with keys:

        - ``"window"``: the windows;
        - ``"shotnoise_jacobian"``: derivative of the estimated shot noise (``num_shotnoise`` divided by the normalization), with
          the same observable and theory as the windows;
        - ``"data_shotnoise_jacobian"``: derivative of the data-only monopole shot noise, one row by the window columns.
    """
    theory = _real_theory(theory)
    basis_vectors = _flat_basis_vectors(theory, basis_vectors)
    return _compute_window_columns(
        mock_survey,
        theory,
        injected_theory=jnp.asarray(basis_vectors, dtype=theory.value().dtype),
        window_theory=get_basis_theory(theory, basis_vectors, names=basis_names),
        nreal=nreal,
        seeds=seeds,
        batch_size=batch_size,
        mock_survey_args=mock_survey_args,
        mock_survey_kwargs=mock_survey_kwargs,
        static_argnums=static_argnums,
        static_argnames=static_argnames,
        tmpdir=tmpdir,
        survey_names=survey_names,
        common_observable=False,
        return_shotnoise_jacobians=return_shotnoise_jacobians,
    )


def get_basis_theory(theory: Mesh2SpectrumPoles, basis_vectors: np.ndarray, names: list[str] | None = None) -> ObservableTree:
    """
    Theory side of a window computed on ``basis_vectors`` at the fiducial ``theory`` (see :py:func:`get_window_basis`).

    A tree with one branch, of type ``"basis"``, holding a leaf with:

    - ``value``: one entry per window column, the least-squares coefficients of the fiducial ``theory`` on the basis vectors;
    - ``vectors``: the basis vectors, of shape ``(n_vectors, n_ells, n_k)``;
    - ``fiducial``: the fiducial ``theory``, where the derivative is taken, of shape ``(n_ells, n_k)``;
    - ``k_theory``, ``k_edges_theory`` and ``ells_theory``: the bins of ``theory``;
    - ``names``: if given, the names of the basis vectors.

    Use :py:func:`get_basis_vectors` and :py:func:`get_fiducial_theory` to read them back.
    """
    theory = _real_theory(theory)
    basis_vectors = _flat_basis_vectors(theory, basis_vectors)
    k, k_edges = _common_k_bins(theory)
    ells = list(theory.ells)
    fiducial = np.asarray(theory.value()).real
    coefficients = np.linalg.lstsq(basis_vectors.T, fiducial, rcond=None)[0]
    leaf = ObservableLeaf(
        value=coefficients,
        vectors=basis_vectors.reshape(len(basis_vectors), len(ells), len(k)),
        fiducial=fiducial.reshape(len(ells), len(k)),
        k_theory=k,
        k_edges_theory=k_edges,
        ells_theory=np.array(ells),
        **({} if names is None else {"names": _basis_names(names, len(basis_vectors))}),
    )
    return ObservableTree([leaf], types=["basis"])


def get_basis_vectors(theory: ObservableTree, return_names: bool = False) -> np.ndarray | tuple[np.ndarray, list[str] | None]:
    """
    Basis vectors of a window theory from :py:func:`get_basis_theory`, of shape ``(n_vectors, n_theory)`` (multipoles concatenated).

    If ``return_names``, also return their names (``None`` if they were not stored).
    """
    leaf = theory.get(types="basis")
    vectors = np.asarray(leaf.values("vectors"))
    vectors = vectors.reshape(len(vectors), -1)
    if return_names:
        names = leaf.values().get("names", None)
        names = None if names is None else [str(name) for name in names]
        return vectors, names
    return vectors


def get_fiducial_theory(theory: ObservableTree | Mesh2SpectrumPoles) -> Mesh2SpectrumPoles:
    """
    Fiducial theory at which a window was computed, from its ``theory``.

    For spike windows (:py:func:`get_window_spikes`), the window theory is the fiducial itself. For basis windows
    (:py:func:`get_window_basis`), it is rebuilt from the ``"basis"`` leaf.
    """
    if not hasattr(theory, "types"):
        return theory
    leaf = theory.get(types="basis")
    k, k_edges = np.asarray(leaf.values("k_theory")), np.asarray(leaf.values("k_edges_theory"))
    poles = [
        Mesh2SpectrumPole(k=k, k_edges=k_edges, num_raw=value, ell=int(ell))
        for ell, value in zip(leaf.values("ells_theory"), leaf.values("fiducial"), strict=True)
    ]
    return Mesh2SpectrumPoles(poles, ells=[int(ell) for ell in leaf.values("ells_theory")])


def project_window(window: WindowMatrix, basis_vectors: np.ndarray, names: list[str] | None = None) -> WindowMatrix:
    """
    Project a window computed on the bins of its theory (*e.g.* :py:func:`get_window_geometry`, or spikes) onto ``basis_vectors``.

    ``basis_vectors`` are given on these bins (see :py:func:`get_window_basis`). The result has the theory
    :py:func:`get_basis_theory`, so that it can be combined with windows from :py:func:`get_window_basis`, as in
    :py:func:`desiwinds.post.apply_control_variate`. ``names`` are stored with the vectors.
    """
    basis_vectors = _flat_basis_vectors(window.theory, basis_vectors)
    return window.clone(value=np.asarray(window.value()) @ basis_vectors.T, theory=get_basis_theory(window.theory, basis_vectors, names=names))


def _real_theory(theory: Mesh2SpectrumPoles) -> Mesh2SpectrumPoles:
    if jnp.iscomplexobj(theory.value()):
        if jnp.iscomplex(theory.value()).any():
            raise ValueError("Cannot handle complex-valued theory.")
        else:
            theory = theory.clone(theory.value().real)
    return theory


def _common_k_bins(theory: Mesh2SpectrumPoles) -> tuple[np.ndarray, np.ndarray]:
    """Return the k and k edges shared by all multipoles of ``theory``."""
    poles = list(theory)  # leaves, also for trees with more labels than ells (e.g. analytic windows)
    k, k_edges = np.asarray(poles[0].coords("k")), np.asarray(poles[0].edges("k"))
    for pole in poles[1:]:
        if not np.array_equal(np.asarray(pole.edges("k")), k_edges):
            raise ValueError("All theory multipoles must share the same k bins.")
    return k, k_edges


def _basis_names(names: list[str], n_vectors: int) -> np.ndarray:
    names = np.array([str(name) for name in names])
    if names.shape != (n_vectors,):
        raise ValueError(f"Got {names.size} basis names for {n_vectors} basis vectors.")
    return names


def _flat_basis_vectors(theory: Mesh2SpectrumPoles, basis_vectors: np.ndarray) -> np.ndarray:
    """Basis vectors as ``(n_vectors, n_theory)``, checked against the size of ``theory``."""
    basis_vectors = np.asarray(basis_vectors)
    if basis_vectors.ndim not in (2, 3):
        raise ValueError(f"basis_vectors must have shape (n_vectors, n_theory) or (n_vectors, n_ells, n_k), got {basis_vectors.shape}.")
    basis_vectors = basis_vectors.reshape(len(basis_vectors), -1)
    if basis_vectors.shape[1] != theory.value().size:
        raise ValueError(f"basis_vectors have {basis_vectors.shape[1]} theory entries, the theory has {theory.value().size}.")
    return basis_vectors


def _compute_window_columns(
    mock_survey,
    theory,
    injected_theory,
    window_theory,
    nreal,
    seeds,
    batch_size,
    mock_survey_args,
    mock_survey_kwargs,
    static_argnums,
    static_argnames,
    tmpdir,
    survey_names,
    common_observable,
    return_shotnoise_jacobians,
):
    """Average and individual realizations of the derivatives of ``mock_survey`` along each row of ``injected_theory``, at ``theory``."""
    mock_survey_args = mock_survey_args or ()
    mock_survey_kwargs = mock_survey_kwargs or {}
    static_argnames = static_argnames or []
    static_argnames = [*static_argnames, "mock_surveys", "batch_size", "return_shotnoise_jacobians"]
    if tmpdir is not None:
        tmpdir = Path(tmpdir)

    # Get some empty observables to use their shapes when creating the window matrix
    observables = mock_survey(*mock_survey_args, theory=theory, seed=jax.random.key(42), **mock_survey_kwargs)
    nsurveys = len(observables)
    if common_observable:
        observables = [observables[0].clone(value=0.0 * observables[0].value())] * nsurveys
    else:
        observables = [observable.clone(value=0.0 * observable.value()) for observable in observables]
    data_observable = ObservableLeaf(value=np.zeros(1))
    survey_names = survey_names or [f"survey_{idx_survey:02d}" for idx_survey in range(nsurveys)]

    # JIT the function retrieving the window component
    get_window = jax.jit(
        get_windows_component,
        static_argnums=static_argnums,
        static_argnames=static_argnames,
    )
    if seeds is None:
        seeds = [2 * imock + 3 for imock in range(nreal)]
    injected_theory = injected_theory.reshape((len(injected_theory), *theory.value().shape))

    products = ["window", "shotnoise_jacobian", "data_shotnoise_jacobian"] if return_shotnoise_jacobians else ["window"]
    # Initialize lists of windows to fill later
    realizations = {product: [[None for j in range(nsurveys)] for i in range(nreal)] for product in products}

    for imock in tqdm(range(nreal), desc="Realization", disable=(jax.process_index() != 0)):
        seed = jax.random.key(seeds[imock])
        responses = get_window(
            *mock_survey_args,
            fiducial_theory=theory,
            injected_theory=injected_theory,
            seed=seed,
            mock_surveys=mock_survey,
            batch_size=batch_size,
            return_shotnoise_jacobians=return_shotnoise_jacobians,
            **mock_survey_kwargs,
        )
        for isurvey, response in enumerate(responses):
            if return_shotnoise_jacobians:
                value, shotnoise, data_shotnoise = response
                values = {"window": value.T, "shotnoise_jacobian": shotnoise.T, "data_shotnoise_jacobian": data_shotnoise[None, :]}
            else:
                values = {"window": response.T}
            for product, value in values.items():
                observable = data_observable if product == "data_shotnoise_jacobian" else observables[isurvey]
                realizations[product][imock][isurvey] = WindowMatrix(value=np.asarray(value), theory=window_theory, observable=observable)
                if (tmpdir is not None) and jax.process_index() == 0:
                    subdir = tmpdir / survey_names[isurvey] / ("" if product == "window" else product)
                    realizations[product][imock][isurvey].write(subdir / f"{seeds[imock]:010d}.h5")
    averages = {
        product: [
            realizations[product][0][isurvey].clone(value=np.mean([realization[isurvey].value() for realization in realizations[product]], axis=0))
            for isurvey in range(nsurveys)
        ]
        for product in products
    }
    if return_shotnoise_jacobians:
        return averages, realizations
    return averages["window"], realizations["window"]


def get_windows_component(
    *mock_surveys_args, injected_theory, fiducial_theory, seed, mock_surveys, batch_size=None, return_shotnoise_jacobians=False, **mock_surveys_kw
):
    """
    By definition, the window is the derivative of the observed power spectrum relative to the input theory evaluated at some fiducial theory value.

    With ``return_shotnoise_jacobians``, also return the derivatives of the estimated shot noise of each pole (``num_shotnoise``
    divided by the normalization) and of the data-only monopole shot noise (``mock_surveys`` must accept ``return_data_shotnoise``).
    """

    # Get a mock-based observed P(k) from a theory power spectrum that looks like "input_theory" and concatenate the poles
    def get_responses(input_value):
        # theory, observe -> global
        theory = fiducial_theory.clone(value=input_value)
        if return_shotnoise_jacobians:
            responses, data_shotnoises = mock_surveys(*mock_surveys_args, theory=theory, seed=seed, return_data_shotnoise=True, **mock_surveys_kw)
            return [
                (jnp.concatenate(response.value(concatenate=False)).real, _get_shotnoise(response), jnp.real(data_shotnoise))
                for response, data_shotnoise in zip(responses, data_shotnoises, strict=True)
            ]
        responses = mock_surveys(*mock_surveys_args, theory=theory, seed=seed, **mock_surveys_kw)
        return [jnp.concatenate(response.value(concatenate=False)).real for response in responses]

    # Get the Jacobian of this, differentated wrt argument `input_value`, evaluated in fiducial_value, dot product with s
    def derivative(s):
        # fiducial_value -> global
        return jax.jvp(get_responses, primals=(jnp.concatenate(fiducial_theory.value(concatenate=False)),), tangents=(s,))[1]
        # return get_responses(jnp.concatenate(fiducial_theory.value(concatenate=False)) * s)

    return jax.lax.map(derivative, injected_theory, batch_size=batch_size)


def _get_shotnoise(spectrum: Mesh2SpectrumPoles) -> jax.Array:
    """Estimated shot noise of each pole, ``num_shotnoise / norm``, concatenated as ``spectrum.value()``."""
    return jnp.concatenate([jnp.ravel(spectrum.get(ell).values("num_shotnoise") / spectrum.get(ell).values("norm")) for ell in spectrum.ells]).real
