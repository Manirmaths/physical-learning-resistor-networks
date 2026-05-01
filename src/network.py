from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import jax.numpy as jnp
import networkx as nx
import numpy as np


@dataclass
class ResistorNetwork:
    num_nodes: int
    edge_i: jnp.ndarray
    edge_j: jnp.ndarray
    input_nodes: Tuple[int, int]
    output_node: int


def graph_to_resistor_network(
    graph: nx.Graph,
    input_nodes: Tuple[int, int] = (0, 1),
    output_node: int | None = None,
) -> ResistorNetwork:
    """Convert a connected NetworkX graph into a ResistorNetwork."""
    num_nodes = graph.number_of_nodes()

    if output_node is None:
        output_node = num_nodes - 1

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


def generate_connected_network(
    num_nodes: int = 40,
    topology: str = "erdos_renyi",
    seed: int = 0,
    edge_prob: float = 0.15,
    small_world_k: int = 6,
    small_world_beta: float = 0.2,
    scale_free_m: int = 3,
    geometric_radius: float | None = None,
    input_nodes: Tuple[int, int] = (0, 1),
    output_node: int | None = None,
    max_tries: int = 10_000,
) -> ResistorNetwork:
    """Generate a connected graph with a chosen topology.

    Supported topologies:
    - "erdos_renyi"
    - "small_world"
    - "scale_free"
    - "random_geometric"

    This function is intended for the revision experiments on topology effects.
    """
    rng = np.random.default_rng(seed)

    if output_node is None:
        output_node = num_nodes - 1

    if output_node in input_nodes:
        raise ValueError("output_node must be different from input_nodes.")

    if any(node < 0 or node >= num_nodes for node in input_nodes):
        raise ValueError("input_nodes must lie in the node range.")

    if output_node < 0 or output_node >= num_nodes:
        raise ValueError("output_node must lie in the node range.")

    for _ in range(max_tries):
        graph_seed = int(rng.integers(0, 2**32 - 1))

        if topology == "erdos_renyi":
            graph = nx.erdos_renyi_graph(
                n=num_nodes,
                p=edge_prob,
                seed=graph_seed,
            )

        elif topology == "small_world":
            k = min(small_world_k, num_nodes - 1)
            if k % 2 == 1:
                k -= 1
            if k < 2:
                raise ValueError("small_world_k must be at least 2.")
            graph = nx.watts_strogatz_graph(
                n=num_nodes,
                k=k,
                p=small_world_beta,
                seed=graph_seed,
            )

        elif topology == "scale_free":
            m = min(scale_free_m, num_nodes - 1)
            if m < 1:
                raise ValueError("scale_free_m must be at least 1.")
            graph = nx.barabasi_albert_graph(
                n=num_nodes,
                m=m,
                seed=graph_seed,
            )

        elif topology == "random_geometric":
            if geometric_radius is None:
                # A practical default that usually gives connected graphs
                # for moderate N, while preserving spatial structure.
                radius = 2.5 * np.sqrt(np.log(num_nodes) / num_nodes)
            else:
                radius = geometric_radius

            graph = nx.random_geometric_graph(
                n=num_nodes,
                radius=radius,
                seed=graph_seed,
            )

        else:
            raise ValueError(f"Unknown topology: {topology}")

        if nx.is_connected(graph):
            return graph_to_resistor_network(
                graph=graph,
                input_nodes=input_nodes,
                output_node=output_node,
            )

    raise RuntimeError(
        f"Could not generate a connected {topology} graph after {max_tries} tries."
    )


def generate_connected_random_network(
    num_nodes: int = 40,
    edge_prob: float = 0.15,
    seed: int = 0,
    input_nodes: Tuple[int, int] = (0, 1),
    output_node: int = 39,
) -> ResistorNetwork:
    """Backward-compatible wrapper for the original Erdős-Rényi generator."""
    return generate_connected_network(
        num_nodes=num_nodes,
        topology="erdos_renyi",
        seed=seed,
        edge_prob=edge_prob,
        input_nodes=input_nodes,
        output_node=output_node,
    )


def resistor_network_to_networkx(network: ResistorNetwork) -> nx.Graph:
    """Convert a ResistorNetwork back to an unweighted NetworkX graph."""
    graph = nx.Graph()
    graph.add_nodes_from(range(network.num_nodes))

    edge_i = np.asarray(network.edge_i)
    edge_j = np.asarray(network.edge_j)

    for i, j in zip(edge_i, edge_j):
        graph.add_edge(int(i), int(j))

    return graph


def shortest_distances_to_output(network: ResistorNetwork) -> Dict[int, int]:
    """Return shortest-path distances from all nodes to the output node."""
    graph = resistor_network_to_networkx(network)
    lengths = nx.single_source_shortest_path_length(graph, network.output_node)
    return dict(lengths)


def input_output_distance(network: ResistorNetwork) -> int:
    """Return the minimum graph distance from either input node to the output."""
    distances = shortest_distances_to_output(network)
    return min(distances[network.input_nodes[0]], distances[network.input_nodes[1]])


def candidate_outputs_at_distance(
    network: ResistorNetwork,
    distance: int,
) -> List[int]:
    """Return nodes whose minimum distance to the two input nodes equals distance.

    This is useful for constructing source-target distance experiments.
    """
    graph = resistor_network_to_networkx(network)

    d0 = nx.single_source_shortest_path_length(graph, network.input_nodes[0])
    d1 = nx.single_source_shortest_path_length(graph, network.input_nodes[1])

    candidates = []
    for node in range(network.num_nodes):
        if node in network.input_nodes:
            continue
        min_distance = min(d0[node], d1[node])
        if min_distance == distance:
            candidates.append(node)

    return candidates


def with_output_node(
    network: ResistorNetwork,
    output_node: int,
) -> ResistorNetwork:
    """Return the same network with a different output node."""
    if output_node in network.input_nodes:
        raise ValueError("output_node must be different from input_nodes.")

    if output_node < 0 or output_node >= network.num_nodes:
        raise ValueError("output_node must lie in the node range.")

    return ResistorNetwork(
        num_nodes=network.num_nodes,
        edge_i=network.edge_i,
        edge_j=network.edge_j,
        input_nodes=network.input_nodes,
        output_node=output_node,
    )


def softplus_conductances(theta: jnp.ndarray, eps: float = 1e-4) -> jnp.ndarray:
    """Positive conductance parameterisation."""
    return jnp.log1p(jnp.exp(theta)) + eps


def build_laplacian(network: ResistorNetwork, conductances: jnp.ndarray) -> jnp.ndarray:
    """Build the weighted graph Laplacian."""
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
    """Solve the reduced Dirichlet problem for node voltages.

    The input nodes are held at prescribed voltages. The free-node voltages
    satisfy

        L_II v_I = - L_IB v_B.

    This is the mathematically precise form that should be used in the paper.
    """
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
    """Return output-node voltage for given trainable parameters and inputs."""
    conductances = softplus_conductances(theta)
    voltages = solve_voltages(network, conductances, input_values)
    return voltages[network.output_node]