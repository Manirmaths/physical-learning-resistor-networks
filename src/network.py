from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import jax.numpy as jnp
import networkx as nx


@dataclass
class ResistorNetwork:
    num_nodes: int
    edge_i: jnp.ndarray
    edge_j: jnp.ndarray
    input_nodes: Tuple[int, int]
    output_node: int


def generate_connected_random_network(
    num_nodes: int = 40,
    edge_prob: float = 0.15,
    seed: int = 0,
    input_nodes: Tuple[int, int] = (0, 1),
    output_node: int = 39,
) -> ResistorNetwork:
    rng_seed = seed
    while True:
        graph = nx.erdos_renyi_graph(n=num_nodes, p=edge_prob, seed=rng_seed)
        if nx.is_connected(graph):
            break
        rng_seed += 1

    edges = sorted((min(u, v), max(u, v)) for u, v in graph.edges())
    edge_i = jnp.array([e[0] for e in edges], dtype=jnp.int32)
    edge_j = jnp.array([e[1] for e in edges], dtype=jnp.int32)

    return ResistorNetwork(
        num_nodes=num_nodes,
        edge_i=edge_i,
        edge_j=edge_j,
        input_nodes=input_nodes,
        output_node=output_node,
    )


def softplus_conductances(theta: jnp.ndarray, eps: float = 1e-4) -> jnp.ndarray:
    return jnp.log1p(jnp.exp(theta)) + eps


def build_laplacian(network: ResistorNetwork, conductances: jnp.ndarray) -> jnp.ndarray:
    n = network.num_nodes
    i = network.edge_i
    j = network.edge_j
    w = conductances

    L = jnp.zeros((n, n), dtype=jnp.float32)

    L = L.at[i, i].add(w)
    L = L.at[j, j].add(w)
    L = L.at[i, j].add(-w)
    L = L.at[j, i].add(-w)

    return L


def solve_voltages(
    network: ResistorNetwork,
    conductances: jnp.ndarray,
    input_values: Tuple[float, float],
) -> jnp.ndarray:
    L = build_laplacian(network, conductances)

    boundary_nodes = jnp.array(list(network.input_nodes), dtype=jnp.int32)
    all_nodes = jnp.arange(network.num_nodes)
    mask = jnp.ones(network.num_nodes, dtype=bool).at[boundary_nodes].set(False)
    interior_nodes = all_nodes[mask]

    v_boundary = jnp.array(input_values, dtype=jnp.float32)

    L_ii = L[jnp.ix_(interior_nodes, interior_nodes)]
    L_ib = L[jnp.ix_(interior_nodes, boundary_nodes)]

    rhs = -L_ib @ v_boundary
    v_interior = jnp.linalg.solve(L_ii, rhs)

    voltages = jnp.zeros(network.num_nodes, dtype=jnp.float32)
    voltages = voltages.at[boundary_nodes].set(v_boundary)
    voltages = voltages.at[interior_nodes].set(v_interior)

    return voltages


def predict_output_voltage(
    network: ResistorNetwork,
    theta: jnp.ndarray,
    input_values: Tuple[float, float],
) -> jnp.ndarray:
    conductances = softplus_conductances(theta)
    voltages = solve_voltages(network, conductances, input_values)
    return voltages[network.output_node]