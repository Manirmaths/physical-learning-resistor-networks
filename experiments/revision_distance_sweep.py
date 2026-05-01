import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import os
import random

import jax.numpy as jnp
import networkx as nx
import numpy as np
import pandas as pd

from src.network import (
    generate_connected_network,
    resistor_network_to_networkx,
    candidate_outputs_at_distance,
    with_output_node,
)
from src.tasks import get_task_a, get_task_b
from src.train import evaluate_task, train_on_task


def run_one(
    seed: int,
    target_distance: int,
    num_nodes: int = 80,
    edge_prob: float = 0.08,
    learning_rate: float = 0.1,
    steps_a: int = 500,
    steps_b: int = 500,
) -> dict | None:
    """Run one A -> B experiment for an output node at a specified graph distance."""

    base_network = generate_connected_network(
        num_nodes=num_nodes,
        topology="erdos_renyi",
        seed=seed,
        edge_prob=edge_prob,
        input_nodes=(0, 1),
        output_node=num_nodes - 1,
    )

    candidates = candidate_outputs_at_distance(base_network, target_distance)

    if len(candidates) == 0:
        return None

    rng = random.Random(seed + 10_000 * target_distance)
    output_node = rng.choice(candidates)

    network = with_output_node(base_network, output_node)

    graph = resistor_network_to_networkx(network)
    d0 = nx.shortest_path_length(graph, source=network.input_nodes[0], target=output_node)
    d1 = nx.shortest_path_length(graph, source=network.input_nodes[1], target=output_node)
    actual_distance = min(d0, d1)

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

    forgetting = loss_a_after - loss_a_before
    task_b_improvement = loss_b_before - loss_b_after

    degrees = np.array([degree for _, degree in graph.degree()], dtype=float)

    return {
        "seed": seed,
        "target_distance": target_distance,
        "actual_distance": actual_distance,
        "distance_to_input0": d0,
        "distance_to_input1": d1,
        "output_node": output_node,
        "num_nodes": num_nodes,
        "num_edges": graph.number_of_edges(),
        "mean_degree": float(np.mean(degrees)),
        "degree_variance": float(np.var(degrees)),
        "average_shortest_path_length": float(nx.average_shortest_path_length(graph)),
        "clustering_coefficient": float(nx.average_clustering(graph)),
        "loss_a_before": loss_a_before,
        "loss_b_before": loss_b_before,
        "loss_a_after": loss_a_after,
        "loss_b_after": loss_b_after,
        "forgetting": forgetting,
        "task_b_improvement": task_b_improvement,
        "task_a_learned": loss_a_before < 0.25,
    }


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("actual_distance")
        .agg(
            n_runs=("seed", "count"),
            forgetting_mean=("forgetting", "mean"),
            forgetting_std=("forgetting", "std"),
            taskB_mean=("loss_b_after", "mean"),
            taskB_std=("loss_b_after", "std"),
            taskB_improvement_mean=("task_b_improvement", "mean"),
            taskB_improvement_std=("task_b_improvement", "std"),
            lossA_before_mean=("loss_a_before", "mean"),
            lossA_before_std=("loss_a_before", "std"),
            num_edges_mean=("num_edges", "mean"),
            clustering_mean=("clustering_coefficient", "mean"),
            path_length_mean=("average_shortest_path_length", "mean"),
        )
        .reset_index()
    )


def main() -> None:
    os.makedirs("results", exist_ok=True)

    # Test first with range(3). Use range(20) for the final run.
    seeds = range(20)

    # For N=80, p=0.08, distances beyond 5 or 6 may be rare.
    target_distances = [1, 2, 3, 4, 5, 6]

    rows = []

    print("\n=== Source-target graph-distance sweep ===")

    for seed in seeds:
        print(f"\nSeed {seed}")

        for target_distance in target_distances:
            print(f"  target distance={target_distance}")

            row = run_one(
                seed=seed,
                target_distance=target_distance,
            )

            if row is None:
                print("    no candidate output node at this distance")
                continue

            rows.append(row)

            print(
                f"    output={row['output_node']}, "
                f"actual_d={row['actual_distance']}, "
                f"L_A_before={row['loss_a_before']:.4f}, "
                f"forgetting={row['forgetting']:.4f}, "
                f"TaskB={row['loss_b_after']:.4f}, "
                f"learned_A={row['task_a_learned']}"
            )

    df = pd.DataFrame(rows)

    full_path = "results/revision_distance_sweep.csv"
    df.to_csv(full_path, index=False)

    # Filter runs where Task A was actually learned.
    df_ok = df[df["task_a_learned"]].copy()

    filtered_path = "results/revision_distance_sweep_filtered.csv"
    df_ok.to_csv(filtered_path, index=False)

    summary_all = summarise(df)
    summary_filtered = summarise(df_ok)

    summary_all_path = "results/revision_distance_sweep_summary.csv"
    summary_filtered_path = "results/revision_distance_sweep_filtered_summary.csv"

    summary_all.to_csv(summary_all_path, index=False)
    summary_filtered.to_csv(summary_filtered_path, index=False)

    print("\nSaved:")
    print(full_path)
    print(filtered_path)
    print(summary_all_path)
    print(summary_filtered_path)

    print("\nFiltered summary:")
    print(summary_filtered)


if __name__ == "__main__":
    main()