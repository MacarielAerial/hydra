from __future__ import annotations

from logging import Logger
from pathlib import Path

import en_core_web_sm
from networkx import DiGraph
from pytest import fixture
from spacy.language import Language
from spacy.tokens import Doc, Span

from src.hydra.nodes.base_logger import get_base_logger
from src.hydra.nodes.linguistic_graph_edges import (
    BaseEdgeFeats,
    EdgeTuple,
    EdgeTuples,
    EdgeType,
)
from src.hydra.nodes.linguistic_graph_nodes import (
    BaseNodeFeats,
    NodeTuple,
    NodeTuples,
    NodeType,
    TokenNodeFeats,
)


@fixture
def test_logger() -> Logger:
    logger = get_base_logger()

    return logger


@fixture
def test_fixture() -> TestFixture:
    return TestFixture()


class TestFixture:
    @property
    def path_own_file(self) -> Path:
        return Path(__file__)

    @property
    def example_spacy_model(self) -> Language:
        nlp: Language = en_core_web_sm.load()

        return nlp

    @property
    def example_paragraph(self) -> str:
        paragraph: str = "In 'To Have and Have Not,' an article in American Theatre "
        "about the need for artists to empower themselves, "
        "arts advocate and activist Jaan Whitehead warns: "
        "'The relationship of language to identity is one of our "
        "least appreciated issues. Language is always more powerful than "
        "it seems in everyday life. It expresses our view of ourselves, "
        "but it also constitutes that view. We can only talk about ourselves "
        "in the language we have available. If that language is rich, it "
        "illuminates us. But if it is narrow or restricted, it represses "
        "and conceals us. If we do not have language that describes "
        "what we believe ourselves to be or what we want to be, we risk "
        "being defined in someone else’s terms.'"

        return paragraph

    @property
    def example_paragraph_doc(self) -> Doc:
        nlp = self.example_spacy_model
        doc: Doc = nlp(self.example_paragraph)

        return doc

    @property
    def example_sentence_span(self) -> Span:
        sent = list(self.example_paragraph_doc.sents)[0]

        return sent

    @property
    def example_node_tuples(self) -> NodeTuples:
        node_tuples = NodeTuples(
            list_node_tuple=[
                NodeTuple(
                    node_id=0,
                    node_feats=TokenNodeFeats(
                        ntype=NodeType.token, text="New York", position_id=0
                    ),
                ),
                NodeTuple(
                    node_id=1,
                    node_feats=TokenNodeFeats(
                        ntype=NodeType.token, text="City", position_id=1
                    ),
                ),
                NodeTuple(
                    node_id=2,
                    node_feats=TokenNodeFeats(
                        ntype=NodeType.token, text="is", position_id=2
                    ),
                ),
                NodeTuple(
                    node_id=3,
                    node_feats=TokenNodeFeats(
                        ntype=NodeType.token, text="in", position_id=3
                    ),
                ),
                NodeTuple(
                    node_id=4,
                    node_feats=TokenNodeFeats(
                        ntype=NodeType.token, text="New York", position_id=4
                    ),
                ),
                NodeTuple(
                    node_id=5,
                    node_feats=BaseNodeFeats(
                        ntype=NodeType.sentence,
                        text="New York City is in New York",
                    ),
                ),
            ]
        )

        return node_tuples

    @property
    def example_edge_tuples(self) -> EdgeTuples:
        edge_tuples = EdgeTuples(
            list_edge_tuple=[
                EdgeTuple(
                    src_id=0,
                    dst_id=5,
                    edge_feats=BaseEdgeFeats(etype=EdgeType.token_to_sent, text=""),
                ),
                EdgeTuple(
                    src_id=4,
                    dst_id=5,
                    edge_feats=BaseEdgeFeats(etype=EdgeType.token_to_sent, text=""),
                ),
            ]
        )

        return edge_tuples

    @property
    def example_nx_g_for_contraction_by_text(  # type: ignore[no-any-unimported]
        self,
    ) -> DiGraph:
        nx_g = DiGraph()
        nx_g.add_nodes_from(
            [
                (0, {"ntype": "TOKEN", "text": "New York"}),
                (1, {"ntype": "TOKEN", "text": "New York"}),
                (2, {"ntype": "SENTENCE", "text": "New York New York"}),
            ]
        )
        nx_g.add_edges_from(
            [(0, 2, {"etype": "TokenToSent"}), (1, 2, {"etype": "TokenToSent"})]
        )

        return nx_g
