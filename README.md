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