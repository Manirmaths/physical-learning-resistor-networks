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


def edge_prob_for_mean_degree(num_nodes: int, mean_degree: float = 6.0) -> float:
    """Choose ER probability so expected mean degree stays approximately fixed."""
    return mean_degree / (num_nodes - 1)


def run_one(
    seed: int,
    num_nodes: int,
    num_steps: int,
    mean_degree: float = 6.0,
    learning_rate: float = 0.1,
) -> dict:
    edge_prob = edge_prob_for_mean_degree(num_nodes, mean_degree)

    network = generate_connected_network(
        num_nodes=num_nodes,
        topology="erdos_renyi",
        seed=seed,
        edge_prob=edge_prob,
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
        num_steps=num_steps,
        print_every=None,
    )

    loss_a_before = evaluate_task(theta_a, network, task_a)
    loss_b_before = evaluate_task(theta_a, network, task_b)

    theta_b, _ = train_on_task(
        theta_init=theta_a,
        network=network,
        task=task_b,
        learning_rate=learning_rate,
        num_steps=num_steps,
        print_every=None,
    )

    loss_a_after = evaluate_task(theta_b, network, task_a)
    loss_b_after = evaluate_task(theta_b, network, task_b)

    row = {
        "seed": seed,
        "num_nodes": num_nodes,
        "num_steps": num_steps,
        "mean_degree_target": mean_degree,
        "edge_prob": edge_prob,
        "learning_rate": learning_rate,
        "loss_a_before": loss_a_before,
        "loss_b_before": loss_b_before,
        "loss_a_after": loss_a_after,
        "loss_b_after": loss_b_after,
        "forgetting": loss_a_after - loss_a_before,
        "task_b_improvement": loss_b_before - loss_b_after,
        "task_a_learned": loss_a_before < 0.25,
    }

    row.update(graph_metrics(network))
    return row


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["num_nodes", "num_steps"])
        .agg(
            n_runs=("seed", "count"),
            forgetting_mean=("forgetting", "mean"),
            forgetting_std=("forgetting", "std"),
            taskB_mean=("loss_b_after", "mean"),
            taskB_std=("loss_b_after", "std"),
            lossA_before_mean=("loss_a_before", "mean"),
            lossA_before_std=("loss_a_before", "std"),
            num_edges_mean=("num_edges", "mean"),
            num_edges_std=("num_edges", "std"),
            mean_degree_mean=("mean_degree", "mean"),
            average_shortest_path_length_mean=("average_shortest_path_length", "mean"),
            clustering_coefficient_mean=("clustering_coefficient", "mean"),
        )
        .reset_index()
    )


def main() -> None:
    os.makedirs("results", exist_ok=True)

    # Cheap but defensible final setting.
    seeds = range(5)
    sizes = [40, 80, 160]
    budgets = [300, 1000]

    rows = []

    print("\n=== Size and training-budget sweep ===")

    for num_nodes in sizes:
        for num_steps in budgets:
            print(f"\nN={num_nodes}, steps={num_steps}")

            for seed in seeds:
                print(f"  seed={seed}")

                row = run_one(
                    seed=seed,
                    num_nodes=num_nodes,
                    num_steps=num_steps,
                )

                rows.append(row)

                print(
                    f"    edges={row['num_edges']}, "
                    f"L_A_before={row['loss_a_before']:.4f}, "
                    f"forgetting={row['forgetting']:.4f}, "
                    f"TaskB={row['loss_b_after']:.4f}, "
                    f"learned_A={row['task_a_learned']}"
                )

    df = pd.DataFrame(rows)

    out_path = "results/revision_size_budget_sweep.csv"
    df.to_csv(out_path, index=False)

    df_ok = df[df["task_a_learned"]].copy()
    filtered_path = "results/revision_size_budget_sweep_filtered.csv"
    df_ok.to_csv(filtered_path, index=False)

    summary_all = summarise(df)
    summary_ok = summarise(df_ok)

    summary_path = "results/revision_size_budget_sweep_summary.csv"
    summary_filtered_path = "results/revision_size_budget_sweep_filtered_summary.csv"

    summary_all.to_csv(summary_path, index=False)
    summary_ok.to_csv(summary_filtered_path, index=False)

    print("\nSaved:")
    print(out_path)
    print(filtered_path)
    print(summary_path)
    print(summary_filtered_path)

    print("\nFiltered summary:")
    print(summary_ok)


if __name__ == "__main__":
    main()