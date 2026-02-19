"""Extensions to jaxpower's ParticleField and FKPField to allow for auxiliary fields."""

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from jaxpower import ParticleField
from jaxpower.mesh import make_array_from_process_local_data


@jax.tree_util.register_pytree_node_class
@dataclass(init=False, frozen=True)
class ExtendedParticleField(ParticleField):
    """Extension of jaxpower's ParticleField to allow for auxiliary field ``extra_weights``."""

    extra_weights: jax.Array = field(repr=False)

    def __init__(
        self,
        positions: jax.Array,
        weights: jax.Array | None = None,
        extra_weights: jax.Array | None = None,
        attrs=None,
        exchange=False,
        backend="auto",
        **kwargs,
    ):
        super().__init__(positions=positions, weights=weights, attrs=attrs, exchange=exchange, backend=backend)
        for key, value in kwargs.items():
            setattr(self, key, value)

        if extra_weights is None:
            extra_weights = jnp.ones_like(positions[..., 0])  # do not provide shape to preserve sharding
        else:
            extra_weights = jnp.asarray(extra_weights)

        if self.exchange_direct is not None:
            # TODO: this won't work with all backends
            extra_weights = self.exchange_direct(make_array_from_process_local_data(extra_weights, pad=0.0), pad=0.0)

        self.__dict__.update(extra_weights=extra_weights)

    def __repr__(self):
        return f"{self.__class__.__name__}(size={self.size})"

    def tree_flatten(self):
        return tuple(getattr(self, name) for name in ["positions", "weights", "attrs", "extra_weights"]), {}

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        new = cls.__new__(cls)
        new.__dict__.update({name: value for name, value in zip(["positions", "weights", "attrs", "extra_weights"], children)})
        return new
