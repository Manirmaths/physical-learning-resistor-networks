import jax.numpy as jnp

from src.network import generate_connected_random_network
from src.tasks import get_task_a
from src.train import evaluate_task, predict_task_outputs, train_on_task

network = generate_connected_random_network()
num_edges = len(network.edge_i)

theta = jnp.zeros(num_edges)
task_a = get_task_a()

initial_loss = evaluate_task(theta, network, task_a)
print("Initial Task A loss:", initial_loss)
print("Initial predictions:")
for input_values, target, prediction in predict_task_outputs(theta, network, task_a):
    print(f"  input={input_values}, target={target}, prediction={prediction:.4f}")

theta_trained, loss_history = train_on_task(
    theta_init=theta,
    network=network,
    task=task_a,
    learning_rate=0.1,
    num_steps=300,
)

final_loss = evaluate_task(theta_trained, network, task_a)
print("\nFinal Task A loss:", final_loss)
print("Final predictions:")
for input_values, target, prediction in predict_task_outputs(theta_trained, network, task_a):
    print(f"  input={input_values}, target={target}, prediction={prediction:.4f}")

print("\nFirst training loss:", loss_history[0])
print("Last training loss:", loss_history[-1])