"""Check that :py:func:`desiwinds.shotnoise.analytic_shotnoise_template` does not depend on the number of shards."""

import os

N_DEVICES = 4
os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + f" --xla_force_host_platform_device_count={N_DEVICES}"

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest
from jaxpower import FKPField, ParticleField, compute_fkp2_normalization, create_sharding_mesh
from toy_survey import make_toy

from desiwinds.shotnoise import analytic_shotnoise_template, prepare_field_weights


def _reshard(particles, mattrs):
    """Exchange the particles over the shards of the current sharding mesh, as for real catalogs."""
    return ParticleField(
        np.asarray(particles.positions),
        weights=np.asarray(particles.weights),
        extra={name: np.asarray(value) for name, value in particles.extra.items()},
        attrs=mattrs,
        exchange=True,
        backend="jax",
    )


def run(n_devices, estimator_weights, include_gic):
    with create_sharding_mesh(device_mesh_shape=(n_devices,)):
        # the mesh attributes and binner must be built within the sharding mesh, as for real catalogs
        toy = make_toy(n_data=2000, n_randoms=8000, seed=3, meshsize=16, ells=(0, 2, 4), n_regions=2)
        fkp_fields, fkp_norms = toy.fkp_fields, toy.fkp_norms
        if n_devices > 1:
            fkp_fields = tuple(FKPField(_reshard(f.data, toy.mattrs), _reshard(f.randoms, toy.mattrs), attrs=toy.mattrs) for f in toy.fkp_fields)
            fkp_norms = [compute_fkp2_normalization(f, bin=toy.binner, cellsize=50.0) for f in fkp_fields]
        args = prepare_field_weights(*fkp_fields, estimator_weights=estimator_weights, gic=include_gic)
        outputs = analytic_shotnoise_template(*fkp_fields, field_weights_args=args, binner=toy.binner, fkp_norms=fkp_norms, include_gic=include_gic)
        return [np.asarray(out) for out in outputs]


@pytest.mark.parametrize("include_gic", [False, True])
@pytest.mark.parametrize("estimator_weights", ["weight_FKP", ("weight_FKP", None)])
@pytest.mark.parametrize("n_devices", [2, 4])
def test_analytic_sharding_invariance(n_devices, estimator_weights, include_gic):
    assert len(jax.devices()) >= n_devices
    reference = run(1, estimator_weights, include_gic)
    for name, ref, out in zip(["spectrum_response", "shotnoise_response", "template"], reference, run(n_devices, estimator_weights, include_gic), strict=True):
        assert ref.shape == out.shape, name
        np.testing.assert_allclose(out, ref, rtol=1e-8, atol=1e-10 * np.abs(ref).max(), err_msg=f"{name} with {n_devices} devices")
