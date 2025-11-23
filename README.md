# 🚀 DESI Window Function Forward Model

This repository provides tools to compute **DESI window functions** through **forward modeling** of observational and survey-specific effects.  
Instead of relying on approximate window function calculation, we simulate how the survey actually distorts clustering signals ---producing realistic, analysis-ready window functions.

This package is built on top of **[jaxpower](https://github.com/adematti/jax-power)** — *the* fastest and most flexible JAX framework for computing power spectra and window functions.

**jaxpower** is, without exaggeration, one of the most elegant and powerful codebases for cosmological large-scale structure. I’m fortunate to work closely with its developer, whose technical skill is matched by their generosity and kindness.

---

## 🔭 Forward-modeled observational effects
Accurately capture how DESI modifies the underlying density field through realistic modeling of:

- **Survey geometry**  
  Full sky mask, coverage variations, and angular selection effects.

- **Integral constraints**
  - Radial integral constraint  
  - Angular mode removal through photometric templates
  - Nulling angular modes

These effects can bias clustering measurements if unaccounted for ---this framework models them consistently and naturally.
