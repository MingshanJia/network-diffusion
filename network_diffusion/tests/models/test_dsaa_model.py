"""Tests for the network_diffusion.multi_spreading."""
import unittest
from typing import Dict, Tuple

import networkx as nx

from network_diffusion import MultilayerNetwork
from network_diffusion.models import DSAAModel
from network_diffusion.models.utils.compartmental import CompartmentalGraph


def prepare_compartments() -> Tuple[CompartmentalGraph, Dict]:
    """Set up compartments needed to perform tests."""
    cmprt = CompartmentalGraph()
    phenomena = {
        "ill": ["S", "I", "R"],
        "aware": ["UA", "A"],
        "vacc": ["UV", "V"],
    }
    for layer, phenomenon in phenomena.items():
        cmprt.add(layer, phenomenon)
    cmprt.compile(background_weight=0)

    cmprt.set_transition_fast("ill.S", "ill.I", ("vacc.UV", "aware.UA"), 0.9)
    cmprt.set_transition_fast("ill.S", "ill.I", ("vacc.V", "aware.A"), 0.05)
    cmprt.set_transition_fast("ill.S", "ill.I", ("vacc.UV", "aware.A"), 0.2)
    cmprt.set_transition_fast("ill.I", "ill.R", ("vacc.UV", "aware.UA"), 0.1)
    cmprt.set_transition_fast("ill.I", "ill.R", ("vacc.V", "aware.A"), 0.7)
    cmprt.set_transition_fast("ill.I", "ill.R", ("vacc.UV", "aware.A"), 0.3)
    cmprt.set_transition_fast("vacc.UV", "vacc.V", ("aware.A", "ill.S"), 0.03)
    cmprt.set_transition_fast("vacc.UV", "vacc.V", ("aware.A", "ill.I"), 0.01)
    cmprt.set_transition_fast(
        "aware.UA", "aware.A", ("vacc.UV", "ill.S"), 0.05
    )
    cmprt.set_transition_fast("aware.UA", "aware.A", ("vacc.V", "ill.S"), 1)
    cmprt.set_transition_fast("aware.UA", "aware.A", ("vacc.UV", "ill.I"), 0.2)

    return cmprt, phenomena


class TestDSAAModel(unittest.TestCase):
    """Test class for MultilayerNetwork class."""

    # pylint: disable=W0212, C0206, C0201, R0914, R1721

    def setUp(self) -> None:
        """Set up most common testing parameters."""
        compartments, phenomena = prepare_compartments()
        self.phenomena = phenomena

        # init multilayer network from nx predefined network
        network = MultilayerNetwork.from_nx_layer(
            nx.les_miserables_graph(), [*phenomena.keys()]
        )
        self.network = network

        # init model
        self.model = DSAAModel(compartments)
        self.model.compartments.seeding_budget = {
            "ill": (84, 13, 3),
            "aware": (77, 23),
            "vacc": (90, 10),
        }

    def test_set_initial_states(self) -> None:
        """Check if set_initial_states appends good statuses to nodes."""
        expected_nodes_states = (
            self.model._compartmental_graph.get_seeding_budget_for_network(
                self.network
            )
        )

        self.model.set_initial_states(net=self.network)

        # obtain info about numbers of nodes
        real_nodes_states = {}
        for l_name, l_graph in self.network.layers.items():
            l_states: Dict[str, int] = {}
            for node in l_graph.nodes():
                state = l_graph.nodes[node]["status"]
                if not l_states.get(state):
                    l_states[state] = 1
                else:
                    l_states[state] += 1
            real_nodes_states[l_name] = l_states

        # check wether layer names are the same
        self.assertEqual(
            rl_names := set(real_nodes_states.keys()),
            el_names := set(expected_nodes_states.keys()),
            f"Wrong number of processes, expected {rl_names} found {el_names}",
        )

        # for each layer do comparison
        for l_name in rl_names:

            # check wether states names are the same
            self.assertEqual(
                rs_names := real_nodes_states[l_name].keys(),
                es_names := expected_nodes_states[l_name].keys(),
                f"Wrong number of states in process {l_name} expected "
                f"{rs_names} found {es_names}",
            )

            # check wether numbers of nodes in given state are the same
            for state in rs_names:
                self.assertEqual(
                    rn_num := real_nodes_states[l_name][state],
                    en_num := expected_nodes_states[l_name][state],
                    f"Wrong number of nodes in state {state} expected "
                    f"{en_num} found {rn_num}",
                )
