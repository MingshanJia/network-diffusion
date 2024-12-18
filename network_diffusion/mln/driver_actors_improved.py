"""Script with functions for driver actor selections and local improvement."""

from copy import deepcopy
from typing import Any, List, Set
import random

from network_diffusion.mln.actor import MLNetworkActor
from network_diffusion.mln.mlnetwork import MultilayerNetwork


def compute_driver_actors(net: MultilayerNetwork) -> List[MLNetworkActor]:
    """Return driver actors for a given network using MDS and local improvement."""
    # Step 1: Compute initial Minimum Dominating Set
    initial_dominating_set: Set[Any] = set()
    for layer in net.layers:
        initial_dominating_set = minimum_dominating_set_with_initial(
            net, layer, initial_dominating_set
        )

    # Step 2: Apply Local Improvement to enhance the Dominating Set
    improved_dominating_set = local_improvement(net, initial_dominating_set)

    return [net.get_actor(actor_id) for actor_id in improved_dominating_set]


def minimum_dominating_set_with_initial(net: MultilayerNetwork, layer: str, initial_set: Set[Any]) -> Set[Any]:
    """
    Return a dominating set that includes the initial set.

    net: MultilayerNetwork
    layer: layer name
    initial_set: set of nodes
    """
    actor_ids = [x.actor_id for x in net.get_actors()]
    if not set(initial_set).issubset(set(actor_ids)):
        raise ValueError("Initial set must be a subset of net's actors")

    dominating_set = set(initial_set)

    net_layer = net.layers[layer]
    isolated = set(actor_ids) - set(net_layer.nodes())
    dominating_set |= isolated
    dominated = deepcopy(dominating_set)

    for node_u in dominating_set:
        if node_u in net_layer.nodes:
            dominated.update(net_layer[node_u])

    while len(dominated) < len(net):
        # Choose a node which dominates the maximum number of undominated nodes
        node_u = max(
            net_layer.nodes(),
            key=lambda x: len(set(net_layer[x]) - dominated),
            default=None
        )
        if node_u is None:
            break  # No further nodes can dominate new nodes
        dominating_set.add(node_u)
        dominated.add(node_u)
        dominated.update(net_layer[node_u])

    return dominating_set


def local_improvement(net: "MultilayerNetwork", initial_set: set) -> set:
    """
    Perform local improvement on the initial dominating set using the First Improvement strategy,
    including the checking procedure after each feasible exchange move.
    """
    dominating_set = set(initial_set)

    # Precompute domination for each node
    domination = compute_domination(net, dominating_set)

    improvement = True
    while improvement:
        improvement = False
        # Shuffle the dominating set to diversify search of neighbors
        current_solution = list(dominating_set)
        random.shuffle(current_solution)

        for u in current_solution:
            # Identify candidate replacements v not in D, but only those leading to a feasible solution
            candidates_v = find_replacement_candidates(net, u, dominating_set, domination)
            random.shuffle(candidates_v)

            for v in candidates_v:
                # Store old solution for rollback if no improvement after checking
                old_dominating_set = set(dominating_set)

                # Attempt the exchange move
                new_dominating_set = (dominating_set - {u}) | {v}
                if is_feasible(net, new_dominating_set):
                    # After a feasible exchange, perform the checking procedure to remove redundancies
                    reduced_set = remove_redundant_vertices(net, new_dominating_set)

                    # Check if we actually improved (reduced the size of the solution)
                    if len(reduced_set) < len(old_dominating_set):
                        # We have found an improvement, update domination and break
                        dominating_set = reduced_set
                        domination = compute_domination(net, dominating_set)
                        improvement = True
                        break
                    else:
                        # No improvement after redundancy removal, revert to old solution
                        dominating_set = old_dominating_set
                        # domination stays the same, no improvement here
                # If not feasible, just continue trying other candidates

            if improvement:
                # Restart the outer loop after finding the first improvement
                break

    return dominating_set


def compute_domination(net: MultilayerNetwork, dominating_set: Set[Any]) -> dict:
    """
    Compute the domination map for the current dominating set per layer.

    Returns a dictionary where keys are layer names and values are dictionaries
    mapping node IDs to sets of dominators in that layer.
    """
    domination_map = {
        layer: {actor.actor_id: set() for actor in net.get_actors()}
        for layer in net.layers
    }

    for layer, net_layer in net.layers.items():
        for actor_id in dominating_set:
            if actor_id in net_layer.nodes:
                domination_map[layer][actor_id].add(actor_id)  # A node dominates itself
                for neighbor in net_layer[actor_id]:
                    domination_map[layer][neighbor].add(actor_id)
    return domination_map


def find_replacement_candidates(net: MultilayerNetwork, u: Any, dominating_set: Set[Any], domination: dict) -> List[
    Any]:
    """
    Find candidate nodes v that can replace u in the dominating set,
    ensuring that all layers remain dominated.
    """
    exclusively_dominated = {}

    for layer, net_layer in net.layers.items():
        if u in net_layer:
            exclusively_dominated[layer] = {
                w for w in set(net_layer[u]) | {u}
                if domination[layer][w] == {u}
            }
        else:
            exclusively_dominated[layer] = set()  # No nodes exclusively dominated by u in this layer

    # Find valid replacement candidates
    candidates = []
    for v in net.get_actor_ids():
        if v in dominating_set:
            continue

        # Ensure v exists in all layers where exclusively dominated nodes are expected
        if all(
                v in net.layers[layer] and nodes.issubset(set(net.layers[layer][v]) | {v})
                for layer, nodes in exclusively_dominated.items()
        ):
            candidates.append(v)

    return candidates


def is_feasible(net: MultilayerNetwork, dominating_set: Set[Any]) -> bool:
    """
    Check if the dominating set is feasible across all layers.
    """
    for layer, net_layer in net.layers.items():
        dominated = set()
        for actor_id in dominating_set:
            if actor_id in net_layer.nodes:
                dominated.add(actor_id)
                dominated.update(net_layer[actor_id])
        if dominated != set(net_layer.nodes()):
            return False
    return True


def remove_redundant_vertices(net, dominating_set):
    """
    Try to remove redundant vertices from the dominating_set without losing feasibility.
    A vertex is redundant if removing it still leaves all nodes dominated.
    Returns a new dominating set with as many redundant vertices removed as possible.
    """
    # We'll attempt to remove vertices one by one.
    # A simple (although not necessarily minimum) approach is to try removing each vertex
    # and see if the set remains feasible. If yes, permanently remove it.
    improved = True
    improved_set = set(dominating_set)
    while improved:
        improved = False
        for d in list(improved_set):
            candidate_set = improved_set - {d}
            if is_feasible(net, candidate_set):
                improved_set = candidate_set
                improved = True
                # Break to re-check from scratch after every removal, ensuring first improvement strategy
                break
    return improved_set



