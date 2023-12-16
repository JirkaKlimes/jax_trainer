import optax

rmsprop = optax.inject_hyperparams(optax.rmsprop)
adam = optax.inject_hyperparams(optax.adam)
sgd = optax.inject_hyperparams(optax.sgd)
