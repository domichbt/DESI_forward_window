"""Unit tests for utility functions in :py:module:`desiforwardwindow.utils`."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from desiforwardwindow.utils import bincount


class TestBincount:
    """Tests for custom bincount implementation with n-D weights."""

    def test_basic_1d_weights(self):
        """Basic test with 1D weights - compare to jnp.bincount."""
        x = jnp.array([0, 1, 1, 2, 2, 2])
        weights = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

        result = bincount(x, weights, length=3)
        expected = jnp.bincount(x, weights, length=3)

        np.testing.assert_allclose(result, expected)

    def test_2d_weights(self):
        """Test with 2D weights."""
        x = jnp.array([0, 1, 1, 2])
        weights = jnp.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])

        result = bincount(x, weights, length=3)

        # Manual expectation
        expected = jnp.array(
            [
                [1.0, 5.0, 4.0],  # bin 0: 1, bin 1: 2+3, bin 2: 4
                [5.0, 13.0, 8.0],  # bin 0: 5, bin 1: 6+7, bin 2: 8
            ]
        )

        np.testing.assert_allclose(result, expected)

    def test_3d_weights(self):
        """Test with 3D weights."""
        x = jnp.array([0, 1, 2])
        weights = jnp.ones((2, 3, 3))  # (batch1, batch2, n)

        result = bincount(x, weights, length=3)

        assert result.shape == (2, 3, 3)
        # Chaque bin devrait avoir 1.0
        np.testing.assert_allclose(result, jnp.ones((2, 3, 3)))

    def test_minlength(self):
        """Test that minlength parameter is working properly."""
        x = jnp.array([0, 1])
        weights = jnp.array([1.0, 2.0])

        # If minlength isn't set, ``length`` should automatically be set to 2 = x.max() + 1
        result = bincount(x, weights)
        assert result.shape == (2,)
        np.testing.assert_allclose(result, jnp.array([1.0, 2.0]))

        # If minlength is set to 5, ``length`` should automatically be set to 5 = max(x.max() + 1, minlength)
        result = bincount(x, weights, minlength=5)
        assert result.shape == (5,)
        np.testing.assert_allclose(result, jnp.array([1.0, 2.0, 0.0, 0.0, 0.0]))

    def test_explicit_length(self):
        """Test that length parameter is working properly."""
        x = jnp.array([0, 1, 1])
        weights = jnp.array([1.0, 2.0, 3.0])

        result = bincount(x, weights, length=5)

        assert result.shape == (5,)
        np.testing.assert_allclose(result, jnp.array([1.0, 5.0, 0.0, 0.0, 0.0]))

    def test_empty_array(self):
        """Test that empty arrays are correctly handled."""
        x = jnp.array([], dtype=int)
        weights = jnp.array([[]], dtype=float)

        result = bincount(x, weights, length=3)

        assert result.shape == (1, 3)
        np.testing.assert_allclose(result, jnp.zeros((1, 3)))

    def test_duplicate_indices(self):
        """Test that duplicated indices are correctly handled."""
        x = jnp.array([1, 1, 1])
        weights = jnp.array([1.0, 2.0, 3.0])

        result = bincount(x, weights, length=2)

        # Everything goes to bin 1
        np.testing.assert_allclose(result, jnp.array([0.0, 6.0]))

    def test_all_zeros_indices(self):
        """Edge case: all zeros."""
        x = jnp.array([0, 0, 0, 0])
        weights = jnp.array([1.0, 1.0, 1.0, 1.0])

        result = bincount(x, weights, length=3)

        np.testing.assert_allclose(result, jnp.array([4.0, 0.0, 0.0]))

    def test_comparison_with_jnp_bincount_multiple_cases(self):
        """More thorough bincount comparison."""
        test_cases = [
            (jnp.array([0, 1, 2, 3]), jnp.array([1.0, 2.0, 3.0, 4.0]), 4),
            (jnp.array([3, 2, 1, 0]), jnp.array([4.0, 3.0, 2.0, 1.0]), 4),
            (jnp.array([0, 0, 1, 1, 2, 2]), jnp.ones(6), 3),
            (jnp.array([5, 5, 5]), jnp.array([1.0, 1.0, 1.0]), 6),
        ]

        for x, weights, length in test_cases:
            result = bincount(x, weights, length=length)
            expected = jnp.bincount(x, weights, length=length)
            np.testing.assert_allclose(result, expected, err_msg=f"Failed for x={x}, weights={weights}")

    def test_integer_weights(self):
        """Integer weights."""
        x = jnp.array([0, 1, 1, 2])
        weights = jnp.array([1, 2, 3, 4])

        result = bincount(x, weights, length=3)

        np.testing.assert_allclose(result, jnp.array([1, 5, 4]))

    def test_negative_weights(self):
        """Negative weights."""
        x = jnp.array([0, 1, 2])
        weights = jnp.array([-1.0, 2.0, -3.0])

        result = bincount(x, weights, length=3)

        np.testing.assert_allclose(result, jnp.array([-1.0, 2.0, -3.0]))

    def test_jit_compatibility_with_fixed_length(self):
        """As long as length is static, jax.jit should work."""

        @jax.jit
        def jitted_bincount(x, weights):
            return bincount(x, weights, length=5)

        x = jnp.array([0, 1, 2, 1])
        weights = jnp.array([[1.0, 2.0, 3.0, 4.0]])

        result = jitted_bincount(x, weights)

        assert result.shape == (1, 5)
        np.testing.assert_allclose(result, jnp.array([[1.0, 6.0, 3.0, 0.0, 0.0]]))

    def test_dtype_preservation(self):
        """Test that weights dtype is preserved."""
        x = jnp.array([0, 1, 2])

        # Float32
        weights_f32 = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
        result_f32 = bincount(x, weights_f32, length=3)
        assert result_f32.dtype == jnp.float32

        # Int32
        weights_i32 = jnp.array([1, 2, 3], dtype=jnp.int32)
        result_i32 = bincount(x, weights_i32, length=3)
        assert result_i32.dtype == jnp.int32

    def test_max_index_at_boundary(self):
        """Edge case: max index is right at the boundary."""
        x = jnp.array([0, 1, 2, 4])  # max = 4
        weights = jnp.array([1.0, 1.0, 1.0, 1.0])

        # auto length should adjust
        result = bincount(x, weights)
        assert result.shape == (5,)  # 0, 1, 2, 3, 4
        np.testing.assert_allclose(result, jnp.array([1.0, 1.0, 1.0, 0.0, 1.0]))

    def test_consistency_across_runs(self):
        """Test deterministic results."""
        x = jnp.array([0, 1, 1, 2, 2, 2])
        weights = jnp.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])

        result1 = bincount(x, weights, length=3)
        result2 = bincount(x, weights, length=3)

        np.testing.assert_array_equal(result1, result2)


if __name__ == "__main__":
    # Exécuter les tests
    pytest.main([__file__, "-v"])
