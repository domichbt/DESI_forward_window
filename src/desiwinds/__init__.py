"""``desiwinds`` is a Python package performing window matrix computations for DESI power spectrum measurements.

It works by sampling forward modeled DESI-like surveys on gaussian mocks realizations. Designed to run on GPU and take into account survey geometry, radial integral constraints, angular mode removal from imaging systematics regressions and angular integral constraint (nulled angular modes).
"""
from . import forward, window

try:
    from . import convenience
except ImportError as e:
    import warnings

    warnings.warn(f"Could not import convenience function, likely due to missing dependencies: {e}", stacklevel=2)
