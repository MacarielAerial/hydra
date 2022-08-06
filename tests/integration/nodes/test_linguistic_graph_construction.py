from logging import Logger

from src.hydra.nodes.linguistic_graph_construction import (
    build_graph_from_node_tuples_and_edge_tuples,
    collect_sent_graph_elements_from_spacy,
    contract_ntype_nodes_by_identical_text,
)
from src.hydra.nodes.linguistic_graph_nodes import NodeType
from tests.conftest import TestFixture


def test_spacy_networkx_graph_contraction(
    test_fixture: TestFixture, test_logger: Logger
) -> None:
    node_tuples, edge_tuples = collect_sent_graph_elements_from_spacy(
        sent=test_fixture.example_sentence_span, logger=test_logger
    )

    nx_g = build_graph_from_node_tuples_and_edge_tuples(
        node_tuples=node_tuples, edge_tuples=edge_tuples, logger=test_logger
    )

    nx_g = contract_ntype_nodes_by_identical_text(
        nx_g=nx_g, ntype=NodeType.token, logger=test_logger
    )

    assert nx_g.number_of_nodes() < len(node_tuples.list_node_tuple)
    assert nx_g.number_of_edges() < len(edge_tuples.list_edge_tuple)
