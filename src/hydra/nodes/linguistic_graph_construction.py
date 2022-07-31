from logging import Logger
from pydoc import Doc
from typing import Dict, Set, Tuple

from spacy.tokens import Span

from .linguistic_graph_edges import (
    DependencyLabel,
    EdgeFeats,
    EdgeTuple,
    EdgeTuples,
    EdgeType,
)
from .linguistic_graph_nodes import (
    NamedEntityLabel,
    NodeFeats,
    NodeTuple,
    NodeTuples,
    NodeType,
    UniversalPOSTag,
)


def collect_sent_graph_from_spacy(
    sent: Span, logger: Logger
) -> Tuple[NodeTuples, EdgeTuples]:
    # Initiate result variable
    node_tuples = NodeTuples(list_node_tuple=[])
    edge_tuples = EdgeTuples(list_edge_tuple=[])

    # Initiate mappings from token indices in the sentence to node ids
    mapping_token_i_to_token_nid: Dict[int, int] = {}
    mapping_ent_id_to_ner_nid: Dict[int, int] = {}
    mapping_token_i_to_uni_pos_nid: Dict[int, int] = {}

    # Initiate a counter for node ids
    curr_nid: int = 0

    #
    # Parse sentence level spacy output into nodes

    # Collect a sentence node
    sent_node_feats = NodeFeats(ntype=NodeType.sentence, text=sent.text)
    sent_node_tuple = NodeTuple(node_id=curr_nid, node_feats=sent_node_feats)

    curr_nid += 1

    # Collect ner nodes
    ner_node_tuples = NodeTuples(list_node_tuple=[])
    set_ner: Set[str] = set()
    for ent in sent.ents:
        # Populate mapping from spacy ner id to ner node id
        mapping_ent_id_to_ner_nid.update({ent.ent_id: curr_nid})

        ner_node_feats = NodeFeats(
            ntype=NodeType.ner, text=NamedEntityLabel(ent.label_).value
        )
        ner_node_tuple = NodeTuple(node_id=curr_nid, node_feats=ner_node_feats)

        ner_node_tuples.list_node_tuple.append(ner_node_tuple)
        set_ner.add(ent.label_)
        curr_nid += 1

    #
    # Parse token level spacy output into nodes
    #

    token_node_tuples = NodeTuples(list_node_tuple=[])
    uni_pos_node_tuples = NodeTuples(list_node_tuple=[])
    set_uni_pos: Set[str] = set()
    for token in sent:
        mapping_token_i_to_token_nid.update({token.i: curr_nid})

        # Collect a token node
        token_node_feats = NodeFeats(ntype=NodeType.token, text=token.text)
        token_node_tuple = NodeTuple(node_id=curr_nid, node_feats=token_node_feats)

        token_node_tuples.list_node_tuple.append(token_node_tuple)

        curr_nid += 1

        mapping_token_i_to_uni_pos_nid.update({token.i: curr_nid})

        # Collect the token node's universal part-of-speech node
        uni_pos_feats = NodeFeats(
            ntype=NodeType.uni_pos, text=UniversalPOSTag(token.pos_.upper()).value
        )
        uni_pos_tuple = NodeTuple(node_id=curr_nid, node_feats=uni_pos_feats)

        uni_pos_node_tuples.list_node_tuple.append(uni_pos_tuple)
        set_uni_pos.add(token.pos_)
        curr_nid += 1

    #
    # Parse sentence level spacy output into edges
    #

    token_to_ner_tuples = EdgeTuples(list_edge_tuple=[])
    for ent in sent.ents:
        for token in ent:
            # Collect a token-to-ner edge
            token_to_ner_feats = EdgeFeats(etype=EdgeType.token_to_ner, text="")
            token_to_ner_tuple = EdgeTuple(
                src_id=mapping_token_i_to_token_nid[token.i],
                dst_id=mapping_ent_id_to_ner_nid[ent.ent_id],
                edge_feats=token_to_ner_feats,
            )
            token_to_ner_tuples.list_edge_tuple.append(token_to_ner_tuple)

    #
    # Parse token level spacy output into edges
    #

    token_to_sent_tuples = EdgeTuples(list_edge_tuple=[])
    token_to_uni_pos_tuples = EdgeTuples(list_edge_tuple=[])
    dependency_arc_tuples = EdgeTuples(list_edge_tuple=[])
    for token in sent:
        # Collect a token-to-sent edge
        token_to_sent_feats = EdgeFeats(etype=EdgeType.token_to_sent, text="")
        token_to_sent_tuple = EdgeTuple(
            src_id=mapping_token_i_to_token_nid[token.i],
            dst_id=sent_node_tuple.node_id,
            edge_feats=token_to_sent_feats,
        )

        token_to_sent_tuples.list_edge_tuple.append(token_to_sent_tuple)

        # Collect a token-to-uni-pos edge
        token_to_uni_pos_feats = EdgeFeats(etype=EdgeType.token_to_uni_pos, text="")
        token_to_uni_pos_tuple = EdgeTuple(
            src_id=mapping_token_i_to_token_nid[token.i],
            dst_id=mapping_token_i_to_uni_pos_nid[token.i],
            edge_feats=token_to_uni_pos_feats,
        )

        token_to_uni_pos_tuples.list_edge_tuple.append(token_to_uni_pos_tuple)

        for child in token.children:
            # Collect a dependency-arc edge
            dependency_arc_feats = EdgeFeats(
                etype=EdgeType.dependency_arc,
                text=DependencyLabel(child.dep_.upper()).value,
            )
            dependency_arc_tuple = EdgeTuple(
                src_id=mapping_token_i_to_token_nid[token.i],
                dst_id=mapping_token_i_to_token_nid[child.i],
                edge_feats=dependency_arc_feats,
            )

            dependency_arc_tuples.list_edge_tuple.append(dependency_arc_tuple)

    #
    # Collect all node tuples and edge tuples
    #

    node_tuples.list_node_tuple.append(sent_node_tuple)
    node_tuples.list_node_tuple.extend(ner_node_tuples.list_node_tuple)
    node_tuples.list_node_tuple.extend(token_node_tuples.list_node_tuple)
    node_tuples.list_node_tuple.extend(uni_pos_node_tuples.list_node_tuple)

    edge_tuples.list_edge_tuple.extend(token_to_ner_tuples.list_edge_tuple)
    edge_tuples.list_edge_tuple.extend(token_to_sent_tuples.list_edge_tuple)
    edge_tuples.list_edge_tuple.extend(token_to_uni_pos_tuples.list_edge_tuple)
    edge_tuples.list_edge_tuple.extend(dependency_arc_tuples.list_edge_tuple)

    logger.debug(
        f"Parsed {NodeTuples} and {EdgeTuples} from {Span} "
        f"from span id {sent.id} of parent {Doc} whose text field "
        f"is of size {len(sent.doc.text)}"
    )

    return node_tuples, edge_tuples
