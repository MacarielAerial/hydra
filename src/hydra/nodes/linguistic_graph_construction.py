from __future__ import annotations

from logging import Logger
from typing import Dict, List, Set, Tuple

import networkx as nx
from networkx import DiGraph, Graph
from networkx_query import search_nodes
from spacy.tokens import Span

from .linguistic_graph_config import NetworkXGraphType
from .linguistic_graph_edges import (
    BaseEdgeFeats,
    DependencyLabel,
    EdgeTuple,
    EdgeTuples,
    EdgeType,
)
from .linguistic_graph_nodes import (
    BaseNodeFeats,
    NamedEntityLabel,
    NodeTuple,
    NodeTuples,
    NodeType,
    TokenNodeFeats,
    UniversalPOSTag,
)
from .utils import group_dict_key_by_value


def collect_sent_graph_elements_from_spacy(
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
    sent_node_feats = BaseNodeFeats(ntype=NodeType.sentence, text=sent.text)
    sent_node_tuple = NodeTuple(node_id=curr_nid, node_feats=sent_node_feats)

    curr_nid += 1

    # Collect ner nodes
    ner_node_tuples = NodeTuples(list_node_tuple=[])
    set_ner: Set[str] = set()
    for ent in sent.ents:
        # Populate mapping from spacy ner id to ner node id
        mapping_ent_id_to_ner_nid.update({ent.ent_id: curr_nid})

        ner_node_feats = BaseNodeFeats(
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
        token_node_feats = TokenNodeFeats(
            ntype=NodeType.token, text=token.text, position_id=token.i
        )
        token_node_tuple = NodeTuple(node_id=curr_nid, node_feats=token_node_feats)

        token_node_tuples.list_node_tuple.append(token_node_tuple)

        curr_nid += 1

        mapping_token_i_to_uni_pos_nid.update({token.i: curr_nid})

        # Collect the token node's universal part-of-speech node
        uni_pos_feats = BaseNodeFeats(
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
            token_to_ner_feats = BaseEdgeFeats(etype=EdgeType.token_to_ner, text="")
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
        token_to_sent_feats = BaseEdgeFeats(etype=EdgeType.token_to_sent, text="")
        token_to_sent_tuple = EdgeTuple(
            src_id=mapping_token_i_to_token_nid[token.i],
            dst_id=sent_node_tuple.node_id,
            edge_feats=token_to_sent_feats,
        )

        token_to_sent_tuples.list_edge_tuple.append(token_to_sent_tuple)

        # Collect a token-to-uni-pos edge
        token_to_uni_pos_feats = BaseEdgeFeats(etype=EdgeType.token_to_uni_pos, text="")
        token_to_uni_pos_tuple = EdgeTuple(
            src_id=mapping_token_i_to_token_nid[token.i],
            dst_id=mapping_token_i_to_uni_pos_nid[token.i],
            edge_feats=token_to_uni_pos_feats,
        )

        token_to_uni_pos_tuples.list_edge_tuple.append(token_to_uni_pos_tuple)

        for child in token.children:
            # Collect a dependency-arc edge
            dependency_arc_feats = BaseEdgeFeats(
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
        f"Parsed {len(node_tuples.list_node_tuple)} node tuples "
        f"and {len(edge_tuples.list_edge_tuple)} edge tuples "
        f" from span of length {len(sent.text)} "
        f"whose span id is {sent.id} and whose parent "
        f"doc object's text field is of size {len(sent.doc.text)}"
    )

    return node_tuples, edge_tuples


def build_graph_from_node_tuples_and_edge_tuples(  # type: ignore[no-any-unimported]
    node_tuples: NodeTuples,
    edge_tuples: EdgeTuples,
    logger: Logger,
    graph_type: NetworkXGraphType = NetworkXGraphType.digraph,
) -> Graph:
    # Initiate networkx graph
    if graph_type == NetworkXGraphType.digraph:
        nx_g = DiGraph()
    else:
        raise NotImplementedError(f"{graph_type} is not defined in its enum class")

    # Populate the initialised graph
    nx_g.add_nodes_from(nodes_for_adding=node_tuples.to_list_node_tuple())
    nx_g.add_edges_from(ebunch_to_add=edge_tuples.to_list_edge_tuple())

    logger.debug(
        f"Constructed networkx graph of type {nx_g.__class__} "
        f"with {len(nx_g.nodes)} nodes and {len(nx_g.edges)} edges"
    )

    return nx_g


def contract_ntype_nodes_by_identical_text(  # type: ignore[no-any-unimported]
    nx_g: Graph,
    ntype: NodeType,
    logger: Logger,
    nfeat_ntype: str = "ntype",
    nfeat_text: str = "text",
    drop_nfeat_contraction: bool = True,
) -> Graph:
    logger.debug(
        f"Pre contraction graph of type {nx_g.__class__} has "
        f"{len(nx_g.nodes)} nodes and {len(nx_g.edges)} edges"
    )

    # Gather node ids of all nodes of a specific type
    list_ntype_nid: List[int] = list(
        search_nodes(graph=nx_g, query={"==": [(nfeat_ntype,), ntype.value]})
    )

    logger.debug(
        f"Identified {len(list_ntype_nid)} nodes of type {ntype.value} "
        f"as candidates for contraction"
    )

    # Identify sets of node ids with identical text value
    dict_nid_text: Dict[str, int] = nx.get_node_attributes(G=nx_g, name=nfeat_text)
    dict_text_list_nid = group_dict_key_by_value(d_input=dict_nid_text)

    # Iteratively contract nodes based on the mapping above
    n_contraction_group: int = 0
    for list_nid in dict_text_list_nid.values():
        if len(list_nid) == 1:
            # No contraction for nodes with unique text
            continue
        else:
            n_contraction_group += 1

        for i in range(1, len(list_nid)):
            nx_g = nx.contracted_nodes(
                G=nx_g,
                u=list_nid[0],
                v=list_nid[i],
                # the below argument needs to be false to avoid copying
                copy=False,
            )

    logger.debug(f"Contracted {n_contraction_group} groups of nodes by identical text")

    if drop_nfeat_contraction:
        n_nfeat_contraction: int = 0
        n_efeat_contraction: int = 0
        # Remove "contraction" as a node attribute added
        for nid, feats in nx_g.nodes.data():
            if "contraction" in feats.keys():
                del nx_g.nodes[nid]["contraction"]
                n_nfeat_contraction += 1

        # Remove "contraction" as an edge attribute added
        for u, v, efeats in nx_g.edges.data():
            if "contraction" in efeats.keys():
                del nx_g.edges[u, v]["contraction"]
                n_efeat_contraction += 1

        logger.debug(
            f"Removed {n_nfeat_contraction} 'contraction' node attributes "
            f"and {n_efeat_contraction} 'contraction' edge attributes "
            f"from a graph with {len(nx_g.nodes)} nodes and {len(nx_g.edges)} edges"
        )

    return nx_g
