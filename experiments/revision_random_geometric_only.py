import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import os

import jax.numpy as jnp
import networkx as nx
import numpy as np
import pandas as pd

from src.network import generate_connected_network, resistor_network_to_networkx
from src.tasks import get_task_a, get_task_b
from src.train import evaluate_task, train_on_task


def graph_metrics(network) -> dict:
    graph = resistor_network_to_networkx(network)
    degrees = np.array([degree for _, degree in graph.degree()], dtype=float)

    return {
        "num_edges": graph.number_of_edges(),
        "mean_degree": float(np.mean(degrees)),
        "degree_variance": float(np.var(degrees)),
        "average_shortest_path_length": float(nx.average_shortest_path_length(graph)),
        "clustering_coefficient": float(nx.average_clustering(graph)),
        "density": float(nx.density(graph)),
    }


def run_one(
    seed: int,
    num_nodes: int = 80,
    geometric_radius: float = 0.17,
    learning_rate: float = 0.1,
    steps_a: int = 500,
    steps_b: int = 500,
) -> dict:
    network = generate_connected_network(
        num_nodes=num_nodes,
        topology="random_geometric",
        seed=seed,
        geometric_radius=geometric_radius,
        input_nodes=(0, 1),
        output_node=num_nodes - 1,
    )

    task_a = get_task_a()
    task_b = get_task_b()

    theta0 = jnp.zeros(len(network.edge_i))

    theta_a, _ = train_on_task(
        theta_init=theta0,
        network=network,
        task=task_a,
        learning_rate=learning_rate,
        num_steps=steps_a,
        print_every=None,
    )

    loss_a_before = evaluate_task(theta_a, network, task_a)
    loss_b_before = evaluate_task(theta_a, network, task_b)

    theta_b, _ = train_on_task(
        theta_init=theta_a,
        network=network,
        task=task_b,
        learning_rate=learning_rate,
        num_steps=steps_b,
        print_every=None,
    )

    loss_a_after = evaluate_task(theta_b, network, task_a)
    loss_b_after = evaluate_task(theta_b, network, task_b)

    row = {
        "topology": "random_geometric",
        "seed": seed,
        "num_nodes": num_nodes,
        "geometric_radius": geometric_radius,
        "learning_rate": learning_rate,
        "steps_a": steps_a,
        "steps_b": steps_b,
        "loss_a_before": loss_a_before,
        "loss_b_before": loss_b_before,
        "loss_a_after": loss_a_after,
        "loss_b_after": loss_b_after,
        "forgetting": loss_a_after - loss_a_before,
        "task_b_improvement": loss_b_before - loss_b_after,
    }

    row.update(graph_metrics(network))
    return row


def main() -> None:
    os.makedirs("results", exist_ok=True)

    # First test with range(3). If edge counts are reasonable, change to range(20).
    seeds = range(20)

    num_nodes = 80
    geometric_radius = 0.17
    learning_rate = 0.1
    steps_a = 500
    steps_b = 500

    rows = []

    print("\n=== Corrected random geometric topology sweep ===")
    print(f"N={num_nodes}, radius={geometric_radius}, seeds={list(seeds)}")

    for seed in seeds:
        print(f"\nseed={seed}")

        row = run_one(
            seed=seed,
            num_nodes=num_nodes,
            geometric_radius=geometric_radius,
            learning_rate=learning_rate,
            steps_a=steps_a,
            steps_b=steps_b,
        )

        rows.append(row)

        print(
            f"edges={row['num_edges']}, "
            f"mean_degree={row['mean_degree']:.2f}, "
            f"ASPL={row['average_shortest_path_length']:.3f}, "
            f"C={row['clustering_coefficient']:.3f}, "
            f"deg_var={row['degree_variance']:.3f}, "
            f"forgetting={row['forgetting']:.4f}, "
            f"TaskB={row['loss_b_after']:.4f}"
        )

    df = pd.DataFrame(rows)

    out_path = "results/revision_random_geometric_only.csv"
    df.to_csv(out_path, index=False)

    summary = (
        df.groupby("topology")
        .agg(
            forgetting_mean=("forgetting", "mean"),
            forgetting_std=("forgetting", "std"),
            taskB_mean=("loss_b_after", "mean"),
            taskB_std=("loss_b_after", "std"),
            taskB_improvement_mean=("task_b_improvement", "mean"),
            num_edges_mean=("num_edges", "mean"),
            num_edges_std=("num_edges", "std"),
            mean_degree_mean=("mean_degree", "mean"),
            degree_variance_mean=("degree_variance", "mean"),
            average_shortest_path_length_mean=("average_shortest_path_length", "mean"),
            clustering_coefficient_mean=("clustering_coefficient", "mean"),
            density_mean=("density", "mean"),
        )
        .reset_index()
    )

    summary_path = "results/revision_random_geometric_only_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\nSaved:")
    print(out_path)
    print(summary_path)

    print("\nSummary:")
    print(summary)


if __name__ == "__main__":
    main()