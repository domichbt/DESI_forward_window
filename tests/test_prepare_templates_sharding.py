"""Check that :py:func:`desiwinds.utils.prepare_templates` does not depend on the number of shards."""

import os

N_DEVICES = 8
os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + f" --xla_force_host_platform_device_count={N_DEVICES}"

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from desiwinds.utils import prepare_templates

N_DAT = 8 * 250
N_RAN = 8 * 1000
N_SYS = 3
TAIL = 1.0
N_BINS = 10
BIN_MARGIN = 1e-7


def make_inputs(seed=42):
    rng = np.random.default_rng(seed)
    data_templates = rng.normal(size=(N_DAT, N_SYS))
    randoms_templates = rng.normal(size=(N_RAN, N_SYS))
    # Ties: rounded values in the last template, and a block of identical values
    randoms_templates[:, -1] = np.round(randoms_templates[:, -1], 1)
    data_templates[:, -1] = np.round(data_templates[:, -1], 1)
    randoms_templates[100:300, 0] = 2.5
    # Padding particles, with values that would be extreme if taken into account
    randoms_is_real = np.ones(N_RAN, dtype=bool)
    randoms_is_real[-N_RAN // 10 :] = False
    randoms_templates[-N_RAN // 10 :] = 100.0

    # Region 0: first half of the catalog (i.e. only on the first shards), region 1: random, region 2: overlaps both
    data_regions = [np.arange(N_DAT) < N_DAT // 2, rng.random(N_DAT) < 0.5, rng.random(N_DAT) < 0.3]
    randoms_regions = [np.arange(N_RAN) < N_RAN // 2, rng.random(N_RAN) < 0.5, rng.random(N_RAN) < 0.3]
    return data_templates, randoms_templates, data_regions, randoms_regions, randoms_is_real


def run(n_devices):
    data_templates, randoms_templates, data_regions, randoms_regions, randoms_is_real = make_inputs()
    if n_devices is None:
        sharding_mesh = None
        put = jnp.asarray
    else:
        sharding_mesh = jax.sharding.Mesh(np.array(jax.devices()[:n_devices]), ("x",))

        def put(arr):
            spec = P("x", *([None] * (arr.ndim - 1)))
            return jax.device_put(arr, NamedSharding(sharding_mesh, spec))

    outputs = prepare_templates(
        data_templates=put(data_templates),
        randoms_templates=put(randoms_templates),
        data_regions=[put(m) for m in data_regions],
        randoms_regions=[put(m) for m in randoms_regions],
        randoms_is_real=put(randoms_is_real),
        tail=TAIL,
        n_bins=N_BINS,
        bin_margin=BIN_MARGIN,
        sharding_mesh=sharding_mesh,
    )
    return [np.asarray(out) for out in outputs]


@pytest.fixture(scope="module")
def reference():
    return run(None)


@pytest.mark.parametrize("n_devices", [1, 2, 4, 8])
def test_sharding_invariance(reference, n_devices):
    assert len(jax.devices()) >= n_devices
    names = ["data_normalized", "data_digitized", "randoms_normalized", "randoms_digitized"]
    for name, ref, out in zip(names, reference, run(n_devices), strict=True):
        assert ref.shape == out.shape, name
        assert np.array_equal(ref, out), f"{name}: {np.sum(ref != out)} differing entries with {n_devices} devices"
