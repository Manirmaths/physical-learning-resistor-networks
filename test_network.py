import jax.numpy as jnp

from src.network import generate_connected_random_network, predict_output_voltage

network = generate_connected_random_network()
num_edges = len(network.edges)

theta = jnp.zeros(num_edges)

y = predict_output_voltage(network, theta, input_values=(1.0, 0.0))

print("Number of edges:", num_edges)
print("Output voltage:", float(y))