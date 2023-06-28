from flax.linen.linear import Array
import jax
from jax import lax
import jax.numpy as jnp
from jax._src import dtypes
from flax import linen as nn
from typing import (Any, Callable, Iterable, List, Optional, Sequence, Tuple,
                    Union)

PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any  # this could be a real type?
Array = Any
PrecisionLike = Union[None, str, lax.Precision, Tuple[str, str],
                      Tuple[lax.Precision, lax.Precision]]
KeyArray = Union[jax.Array, jax.random.KeyArray]

default_kernel_init = nn.initializers.lecun_normal()

def normal_init(mean, stddev, dtype = jnp.float_):
    def init(key: KeyArray,
           shape: jax.core.Shape,
           dtype: Any = dtype) -> Array:
        dtype = dtypes.canonicalize_dtype(dtype)
        return jax.random.normal(key, shape, dtype) * stddev + mean
    return init

def normal_inv_softplus_init(mean, stddev, dtype = jnp.float_):
    # This initializer is for the posterior std.
    def init(key: KeyArray,
           shape: jax.core.Shape,
           dtype: Any = dtype) -> Array:
        dtype = dtypes.canonicalize_dtype(dtype)
        samples = jax.random.normal(key, shape, dtype) * stddev + mean
        return jnp.log(jnp.expm1(jnp.abs(samples)))
    return init

def draw_samples(indices, mean, std, rng, dtype):
    mean = mean[indices]
    std = std[indices]
    noise = jax.random.normal(rng, jnp.shape(mean), dtype)
    return mean + std*noise

class Dense(nn.Module):
    """A linear transformation applied over the last dimension of the input.

    Attributes:
        features: the number of output features.
        use_bias: whether to add a bias to the output (default: True).
        rank: rank of the approximation
        num_components: number of components in the mixture posterior
        dtype: the dtype of the computation (default: infer from input and params).
        param_dtype: the dtype passed to parameter initializers (default: float32).
        precision: numerical precision of the computation see `jax.lax.Precision`
        for details.
        kernel_init: initializer function for the weight matrix.
        bias_init: initializer function for the bias.
    """
    features: int
    rank: int = 2
    num_components: int = 4
    use_bias: bool = True
    rng_collection: str = 'low-rank'
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    precision: PrecisionLike = None
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.zeros
    posterior_mean_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.zeros
    posterior_std_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.zeros

    @nn.compact
    def __call__(self, inputs: Array, indices: Array) -> Array:
        """Applies a linear transformation to the inputs along the last dimension.

        Args:
        inputs: The nd-array to be transformed. 
        indices: The [batch_size] array containing the indices of the components for each batch input sample from which to draw the posterior sample.

        Returns:
        The transformed input.
        """
        kernel = self.param('kernel',
                            self.kernel_init,
                            (jnp.shape(inputs)[-1], self.features),
                            self.param_dtype)
        posterior_mean_in = self.param('posterior_mean_in',
                                    self.posterior_mean_init,
                                    (self.num_components, self.rank, jnp.shape(inputs)[-1]),
                                    self.param_dtype)
        posterior_std_in = self.param('posterior_std_in',
                                    self.posterior_std_init,
                                    jnp.shape(posterior_mean_in),
                                    self.param_dtype)
        posterior_mean_out = self.param('posterior_mean_out',
                                    self.posterior_mean_init,
                                    (self.num_components, self.rank, self.features),
                                    self.param_dtype)
        posterior_std_out = self.param('posterior_std_out',
                                    self.posterior_std_init,
                                    jnp.shape(posterior_mean_out),
                                    self.param_dtype)
        
        if self.use_bias:
            bias = self.param('bias', self.bias_init, (self.features,),
                                self.param_dtype)
        else:
            bias = None
        inputs, kernel, bias, posterior_mean_in, posterior_std_in, posterior_mean_out, posterior_std_out = nn.dtypes.promote_dtype(inputs, kernel, bias, posterior_mean_in, posterior_std_in, posterior_mean_out, posterior_std_out, dtype=self.dtype)
        posterior_std_in = nn.softplus(posterior_std_in)
        posterior_std_out = nn.softplus(posterior_std_out)
        rng_in = self.make_rng(self.rng_collection)
        rng_out = self.make_rng(self.rng_collection)
        in_samples = draw_samples(indices, posterior_mean_in, posterior_std_in, rng_in, self.dtype) # [batch_size, rank, input_size]
        out_samples = draw_samples(indices, posterior_mean_out, posterior_std_out, rng_out, self.dtype) # [batch_size, rank, features]
        
        inputs = lax.mul(lax.expand_dims(inputs, (inputs.ndim - 1,)), in_samples) # [batch_size, rank, input_size]

        y = lax.dot_general(inputs, kernel,
                            (((inputs.ndim - 1,), (0,)), ((), ())),
                            precision=self.precision)
        
        y = lax.mul(y, out_samples) # [batch_size, rank, features]
        y = jnp.sum(y, axis=y.ndim - 2) # sum over the rank dimension. The output tensor dimension: [batch_size, features]
        if bias is not None:
            y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))
        return y
    
if __name__ == '__main__':
    layer = Dense(features=5, rank=9, num_components=4, posterior_mean_init=normal_init(1.0, 0.05), posterior_std_init=normal_inv_softplus_init(0.0, 0.5))
    key = jax.random.PRNGKey(44)
    key1, key2, key3, key4 = jax.random.split(key, 4)
    inputs = jax.random.uniform(key4, (8, 5))
    indices = jnp.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=jnp.int32)
    init_variables = layer.init({'params': key2, 'low-rank': key3}, inputs, indices)
    print(init_variables)
    outputs = layer.apply(init_variables, inputs, indices, rngs={'low-rank': key3})
    print(outputs)