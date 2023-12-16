import jax
import jax.numpy as jnp


def mse(y_true, y_pred):
    return jnp.mean((y_true-y_pred)**2)


def mape(y_true, y_pred):
    return jnp.mean(jnp.abs((y_true-y_pred)/y_true))
