from logging import Logger

from src.hydra.nodes.linguistic_graph_construction import (
    build_graph_from_node_tuples_and_edge_tuples,
    collect_sent_graph_elements_from_spacy,
    contract_ntype_nodes_by_identical_text,
)
from src.hydra.nodes.linguistic_graph_nodes import NodeType
from tests.conftest import TestFixture


def test_collect_sent_graph_from_spacy(
    test_logger: Logger, test_fixture: TestFixture
) -> None:
    node_tuples, edge_tuples = collect_sent_graph_elements_from_spacy(
        sent=test_fixture.example_sentence_span, logger=test_logger
    )

    assert len(node_tuples.list_node_tuple) > 0
    assert len(edge_tuples.list_edge_tuple) > 0


def test_build_graph_from_node_tuples_and_edge_tuples(
    test_logger: Logger, test_fixture: TestFixture
) -> None:
    nx_g = build_graph_from_node_tuples_and_edge_tuples(
        node_tuples=test_fixture.example_node_tuples,
        edge_tuples=test_fixture.example_edge_tuples,
        logger=test_logger,
    )

    assert len(nx_g.nodes) == 6
    assert len(nx_g.edges) == 2


def test_contract_ntype_nodes_by_identical_text(
    test_logger: Logger, test_fixture: TestFixture
) -> None:
    nx_g = contract_ntype_nodes_by_identical_text(
        nx_g=test_fixture.example_nx_g_for_contraction_by_text,
        ntype=NodeType.token,
        logger=test_logger,
    )

    assert len(nx_g.nodes) == 2
    assert len(nx_g.edges) == 1
