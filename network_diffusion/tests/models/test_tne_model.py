"""Tests for the network_diffusion.models.tne_model."""
import random
import unittest

import networkx as nx
import numpy as np

from network_diffusion import TemporalNetwork
from network_diffusion.models import TemporalNetworkEpistemologyModel
from network_diffusion.seeding import RandomSeedSelector
from network_diffusion.temporal_spreading import TemporalSpreading


random.seed(42)
np.random.seed(42)


def create_artificial_temporal_data():
    # Create a list of 10 nodes
    nodes = list(range(1, 11))

    # Function to create a snapshot of the temporal network
    def create_snapshot():
        G = nx.Graph()
        G.add_nodes_from(nodes)

        # Add 20 random edges for each snapshot
        edges = random.sample([(i, j) for i in nodes for j in nodes if i != j], 20)
        G.add_edges_from(edges)
        return G

    # Create the dictionary of snapshots
    snapshots = []
    for snapshot_id in range(10):
        snapshot_graph = create_snapshot()
        snapshots.append(snapshot_graph)

    return snapshots


class TestDSAAModel(unittest.TestCase):
    """Test class for MultilayerNetwork class."""

    # pylint: disable=W0212, C0206, C0201, R0914, R1721

    def setUp(self) -> None:
        """Set up most common testing parameters."""
        # init temporal network from nx predefined network
        temporal_snaps = create_artificial_temporal_data()
        network = TemporalNetwork.from_nx_layers(temporal_snaps)
        self.network = network

        self.model = TemporalNetworkEpistemologyModel(
            seeding_budget=(50, 50),
            seed_selector=RandomSeedSelector(),
            trials_nr=10,
            epsilon=0.05
        )

    def test_experiment_results(self) -> None:
        """Check if the experiment results are as expected."""
        expected_global_stats = [
            {'TPN': (('A', 5), ('B', 5))},
            {'TPN': (('A', 7), ('B', 3))},
            {'TPN': (('A', 6), ('B', 4))},
            {'TPN': (('A', 4), ('B', 6))},
            {'TPN': (('A', 3), ('B', 7))},
            {'TPN': (('A', 3), ('B', 7))},
            {'TPN': (('B', 8), ('A', 2))},
            {'TPN': (('B', 9), ('A', 1))},
            {'TPN': (('B', 9), ('A', 1))},
            {'TPN': (('B', 10),)}
        ]

        experiment = TemporalSpreading(self.model, self.network)

        logs = experiment.perform_propagation()

        # check weather course of the process goes as expected
        self.assertEqual(
            obtained_global_stats := logs._global_stats,
            expected_global_stats,
            f"Wrong course of the spreading process, expected {expected_global_stats} found {obtained_global_stats}",
        )
