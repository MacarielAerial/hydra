from logging import Logger

from src.hydra.nodes.linguistic_graph_construction import collect_sent_graph_from_spacy
from tests.conftest import TestFixture


def test_collect_sent_graph_from_spacy(
    test_logger: Logger, test_fixture: TestFixture
) -> None:
    node_tuples, edge_tuples = collect_sent_graph_from_spacy(
        sent=test_fixture.example_sentence_span, logger=test_logger
    )

    assert len(node_tuples.list_node_tuple) > 0
    assert len(edge_tuples.list_edge_tuple) > 0
