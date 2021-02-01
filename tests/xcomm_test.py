w_final, w = jax.lax.scan(mimo_sgd, w_init, y_f)
