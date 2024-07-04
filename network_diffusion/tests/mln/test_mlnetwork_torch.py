import networkx as nx
import pytest
import torch
from bidict import bidict

from network_diffusion import utils
from network_diffusion.mln import (
    MultilayerNetwork,
    MultilayerNetworkTorch,
    functions,
    mlnetwork_torch,
)


@pytest.fixture
def set_up():
    utils.fix_random_seed(42)


def network_florentine():
    return MultilayerNetwork.from_mpx(
        f"{utils.get_absolute_path()}/tests/data/florentine.mpx"
    )


def network_t1():
    return MultilayerNetwork(
        {
            "l1": nx.Graph([(0, 4), (1, 2), (1, 3), (1, 4), (2, 3)]),
            "l2": nx.Graph([(0, 2), (1, 4), (2, 3)]),
        }
    )


def network_t2():
    net_base = network_t1()
    net_base["l1"].add_node(5)
    net_base["l2"].add_edge(0, 5)
    return net_base


@pytest.mark.parametrize(
    "net_in,exp_ac_map,exp_ad_nodes",
    [
        (
            functions.get_toy_network_piotr(),
            bidict(
                {
                    1: 0,
                    4: 1,
                    2: 2,
                    3: 3,
                    5: 4,
                    6: 5,
                    7: 6,
                    9: 7,
                    8: 8,
                    10: 9,
                    11: 10,
                }
            ),
            {"l1": {11}, "l2": {3}, "l3": {8}},
        ),
        (
            functions.get_toy_network_cim(),
            bidict(
                {
                    1: 0,
                    2: 1,
                    3: 2,
                    4: 3,
                    6: 4,
                    7: 5,
                    5: 6,
                    11: 7,
                    8: 8,
                    9: 9,
                    10: 10,
                    12: 11,
                }
            ),
            None,
        ),
        (
            network_florentine(),
            bidict(
                {
                    "Ridolfi": 0,
                    "Tornabuoni": 1,
                    "Strozzi": 2,
                    "Peruzzi": 3,
                    "Pazzi": 4,
                    "Salviati": 5,
                    "Medici": 6,
                    "Guadagni": 7,
                    "Lamberteschi": 8,
                    "Castellani": 9,
                    "Bischeri": 10,
                    "Barbadori": 11,
                    "Albizzi": 12,
                    "Ginori": 13,
                    "Acciaiuoli": 14,
                }
            ),
            {
                "marriage": set(),
                "business": {"Strozzi", "Acciaiuoli", "Ridolfi", "Albizzi"},
            },
        ),
    ],
)
def test__prepare_mln_for_conversion(net_in, exp_ac_map, exp_ad_nodes, set_up):
    obt_net, obt_actors, obt_added_nodes = (
        mlnetwork_torch._prepare_mln_for_conversion(net_in)
    )
    assert obt_net.is_multiplex()
    assert obt_added_nodes == exp_ad_nodes
    assert obt_actors == exp_ac_map


@pytest.mark.parametrize(
    "net_in,ac_order,net_exp,l_exp",
    [
        (
            network_t1(),
            [0, 1, 2, 3, 4],
            torch.sparse_coo_tensor(
                indices=torch.Tensor(
                    [
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                        [0, 1, 1, 1, 2, 2, 3, 3, 4, 4, 0, 1, 2, 2, 3, 4],
                        [4, 2, 3, 4, 1, 3, 1, 2, 0, 1, 2, 4, 0, 3, 2, 1],
                    ]
                ),
                values=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ),
            ["l1", "l2"],
        ),
        (
            network_t2(),
            [0, 1, 2, 3, 4, 5],
            torch.sparse_coo_tensor(
                indices=torch.Tensor(
                    [
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                        [0, 1, 1, 1, 2, 2, 3, 3, 4, 4, 0, 0, 1, 2, 2, 3, 4, 5],
                        [4, 2, 3, 4, 1, 3, 1, 2, 0, 1, 2, 5, 4, 0, 3, 2, 1, 0],
                    ]
                ),
                values=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ),
            ["l1", "l2"],
        ),
        (
            network_t2(),
            [5, 3, 1, 0, 2, 4],
            torch.sparse_coo_tensor(
                indices=torch.Tensor(
                    [
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 2, 2, 2, 3, 4, 4, 5, 5, 0, 1, 2, 3, 3, 4, 4, 5],
                        [2, 4, 1, 4, 5, 5, 1, 2, 2, 3, 3, 4, 5, 0, 4, 1, 3, 2],
                    ]
                ),
                values=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ),
            ["l1", "l2"],
        ),
    ],
)
def test__mln_to_sparse(net_in, ac_order, net_exp, l_exp):
    net_obtained, l_names = mlnetwork_torch._mln_to_sparse(net_in, ac_order)
    assert l_exp == l_names
    assert torch.all(net_exp._indices() == net_obtained._indices())
    assert torch.all(net_exp._values() == net_obtained._values())
    assert net_exp.size() == net_obtained.size()
    assert net_exp._nnz() == net_obtained._nnz()
    assert net_exp.layout == net_obtained.layout


@pytest.mark.parametrize(
    "l_order,ac_map,nodes_added,exp_output",
    [
        (
            ["l1", "l2"],
            bidict({1: 0, 2: 1, 3: 2, 4: 3}),
            {"l1": {}, "l2": {1, 2}},
            torch.Tensor([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]]),
        ),
        (
            ["l_1"],
            bidict({0: 0, 1: 1, 2: 2}),
            {"l_1": {}},
            torch.Tensor([[0.0, 0.0, 0.0]]),
        ),
        (
            ["la_1", "la_2", "la_3"],
            bidict({0: 0, 1: 2, 2: 1, 3: 3, 4: 5, 5: 4}),
            {"la_1": {0, 1, 2}, "la_2": {}, "la_3": {5}},
            torch.Tensor(
                [
                    [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                ]
            ),
        ),
    ],
)
def test__create_nodes_mask(l_order, ac_map, nodes_added, exp_output, set_up):
    obtained_mask = mlnetwork_torch._create_nodes_mask(
        l_order, ac_map, nodes_added, True
    )
    assert torch.equal(exp_output, obtained_mask)


# class TestMultilayerNetworkTorch:

#     @pytest.mark.parametrize(
#         "network,exp_output",
#         [
#             (
#                 functions.get_toy_network_piotr(),
#                 # bidict({1:0, 2:1, 3:2, 4:3}),
#                 # {"l1": {}, "l2": {1, 2}},
#                 torch.Tensor([[0., 0., 0., 0.], [1., 1., 0., 0.]])
#             ),
#         ]
#     )
#     def test_from_mln(self, network, exp_output):
#         net_tensor = MultilayerNetworkTorch.from_mln(network)
#         assert True

#     def test___repr__(self):
#         assert True
