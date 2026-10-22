# 🚀 `desiwinds`: DESI Window Matrix Forward Model

The `desiwinds` package (DESI *wind*ow matrix *s*ampling) provides tools to compute **DESI window matrices** through **forward modeling** of observational and survey-specific effects. 
Instead of relying on approximate window function calculation, we forward model at the field level how the survey actually distorts clustering signals, producing realistic, analysis-ready window functions.

It is built on top of **[jaxpower](https://github.com/adematti/jax-power)** — a fast and flexible JAX framework for computing power spectra and window functions.

---

## 🔭 Forward-modeled observational effects
Accurately capture how DESI modifies the underlying density field through realistic modeling of:

- **Survey geometry**  
  Full sky mask, coverage variations, and angular selection effects.

- **Integral constraints**
  - Radial integral constraint  
  - Angular mode removal due to templates-based angular systematics correction
  - Angular integral constraint / nulled angular modes

These effects can bias clustering measurements if unaccounted for: this framework models them consistently and naturally, both on the **auto-** and **cross-power spectra**.

## 📦 Installation

The module is pip-installable, e.g. through
```
pip install git+https://github.com/domichbt/DESI_forward_window.git
```
The required dependencies are
 * jax
 * numpy
 * healpy
 * tqdm
 * [lsstypes](https://github.com/adematti/lsstypes)
 * [jax-power](https://github.com/adematti/jax-power)

and should be handled automatically by the `pip` command.
Python ≥ 3.10 is also required.

Optional dependencies are available as `[convenience]` in order to be able to use the `desiwinds.convenience` submodule.

## 🧪 Usage

`desiwinds` functions are meant to be run on GPU, although they generally work on CPU as well.
They can be distributed on several GPUs to accommodate memory requirements.
Examples of forward models and window computation are provided in the `./notebooks/` directory; these were run on a single NERSC (Perlmutter) GPU.

In order to distribute computations, follow the distribution and sharding setup of `jax-power` as in [its documentation](https://github.com/adematti/jax-power?tab=readme-ov-file#-quick-example-auto-power-spectrum-with-multi-gpu).
In short:
```python
import jax
jax.distributed.initialize()

from jaxpower import create_sharding_mesh

with create_sharding_mesh() as sharding_mesh:
    # Here do whatever computations
    ...

jax.distributed.shutdown()
```

### Windows and post-processing

Windows are Jacobians of the forward model at a fiducial theory, averaged over realizations (`desiwinds.window`):

- `get_window_spikes` / `get_windows_spikes`: one column per theory bin;
- `get_window_basis`: one column per basis vector (any power spectrum shape, given on the theory bins), at a cost that scales with
  the number of vectors. The window theory (`get_basis_theory`) stores the vectors and the fiducial theory at which the derivative
  is taken; read them back with `get_basis_vectors` and `get_fiducial_theory`, and project an analytic window onto the same
  vectors with `project_window`.

With `return_shotnoise_jacobians=True`, they also return the Jacobians of the estimated shot noise. `desiwinds.post` then
post-processes a window run, in this order:

1. `average_windows` over the realizations;
2. `correct_selfpairs` (or `correct_selfpairs_analytic` without the Jacobians): the mock data are randoms weighted by `1 + delta`,
   so their self-pairs add a white noise linear in the theory, which the Jacobian puts into the window; this removes it, given the
   white-noise template of the same effects;
3. `apply_control_variate` with the analytic geometry window (`get_window_geometry`).

For the linear local-PNG + Kaiser model, `desiwinds.convenience.get_png_basis_vectors` gives the basis vectors
\{P, gP, g²P\} × multipoles on the window's theory bins (6 for Kaiser, or all 9 responses), with their names, and
`get_png_kaiser_coefficients` the matching coefficients from (b1, b_phi f_NL, f) at the effective redshift. Shapes at z = 0 with
`get_growth_ratio(cosmology, z_eff)` keep the window independent of redshift, up to the scale-dependent growth from massive
neutrinos (≤ 0.6% for 0.06 eV at z ≤ 2.5); shapes built at `z = z_eff` are exact.
