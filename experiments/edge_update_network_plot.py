import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import os

import jax.numpy as jnp
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from src.network import generate_connected_random_network
from src.tasks import get_task_a, get_task_b
from src.train import train_on_task


def main():
    seed = 0
    num_nodes = 40
    top_k = 10

    network = generate_connected_random_network(
        num_nodes=num_nodes,
        edge_prob=0.15,
        seed=seed,
        input_nodes=(0, 1),
        output_node=39,
    )

    theta0 = jnp.zeros(len(network.edge_i))
    task_a = get_task_a()
    task_b = get_task_b()

    # Train on Task A
    theta_a, _ = train_on_task(
        theta_init=theta0,
        network=network,
        task=task_a,
        learning_rate=0.1,
        num_steps=300,
    )

    # Train on Task B starting from theta_a
    theta_b, _ = train_on_task(
        theta_init=theta_a,
        network=network,
        task=task_b,
        learning_rate=0.1,
        num_steps=300,
    )

    delta_theta = np.abs(np.array(theta_b - theta_a, dtype=float))

    # Build graph
    G = nx.Graph()
    G.add_nodes_from(range(network.num_nodes))

    edges = list(zip(np.array(network.edge_i), np.array(network.edge_j)))
    for idx, (u, v) in enumerate(edges):
        G.add_edge(int(u), int(v), weight=delta_theta[idx], edge_index=idx)

    # Find top-k updated edges
    top_indices = np.argsort(delta_theta)[-top_k:]
    top_edge_set = set(int(i) for i in top_indices)

    # Layout
    pos = nx.spring_layout(G, seed=seed)

    # Separate edges into background vs highlighted
    background_edges = []
    highlighted_edges = []
    for u, v, data in G.edges(data=True):
        if data["edge_index"] in top_edge_set:
            highlighted_edges.append((u, v))
        else:
            background_edges.append((u, v))

    os.makedirs("plots", exist_ok=True)
    plot_path = "plots/edge_update_network_highlight_seed0.png"

    plt.figure(figsize=(9, 7))

    # Draw background edges
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=background_edges,
        width=1.0,
        alpha=0.25,
    )

    # Draw highlighted edges
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=highlighted_edges,
        width=2.5,
        alpha=0.9,
        edge_color="red",
    )

    # Draw nodes
    input_nodes = list(network.input_nodes)
    output_node = network.output_node
    other_nodes = [n for n in G.nodes if n not in input_nodes + [output_node]]

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=other_nodes,
        node_size=120,
        alpha=0.8,
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=input_nodes,
        node_size=180,
        node_color="green",
        alpha=0.95,
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=[output_node],
        node_size=220,
        node_color="orange",
        alpha=0.95,
    )

    # Labels only for special nodes
    labels = {
        input_nodes[0]: "In 1",
        input_nodes[1]: "In 2",
        output_node: "Out",
    }
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9)

    plt.title(f"Top {top_k} updated edges after Task B training")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"Saved plot to: {plot_path}")


if __name__ == "__main__":
    main()