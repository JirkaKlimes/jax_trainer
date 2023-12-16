import optax


def warmup_cosine(warmup_steps: int, cosine_steps: int, lr: float):
    warmup_fn = optax.linear_schedule(
        init_value=0.0, end_value=lr, transition_steps=warmup_steps)
    cosine_fn = optax.cosine_decay_schedule(
        init_value=lr, decay_steps=cosine_steps)
    return optax.join_schedules(
        schedules=[warmup_fn, cosine_fn],
        boundaries=[warmup_steps],
    )
