from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Tuple

from dataclasses_json import dataclass_json

from .utils import enum_dict_factory


class DependencyLabel(Enum):
    root: str = "ROOT"
    adjectival_clause: str = "ACL"
    adjectival_complement: str = "ACOMP"
    adverbial_clause_modifier: str = "ADVCL"
    adverbial_modifier: str = "ADVMOD"
    agent: str = "AGENT"
    adjectival_modifier: str = "AMOD"
    appositional_modifier: str = "APPOS"
    attribute: str = "ATTR"
    auxiliary: str = "AUX"
    passive_auxiliary: str = "AUXPASS"
    case_marking: str = "CASE"
    coordinating_conjunction: str = "CC"
    clausal_complement: str = "CCOMP"
    compound: str = "COMPOUND"
    conjunct: str = "CONJ"
    clausal_subject: str = "CSUBJ"
    passive_clausal_subject: str = "CSUBJPASS"
    dative: str = "DATIVE"
    unclassified_dependent: str = "DEP"
    determiner: str = "DET"
    direct_object: str = "DOBJ"
    expletive: str = "EXPL"
    interjection: str = "INTJ"
    marker: str = "MARKER"
    meta_modifier: str = "META"
    negation_modifier: str = "NEG"
    modifier_of_nominal: str = "NMOD"
    noun_phrase_as_adverbial_modifier: str = "NPADVMOD"
    nominal_subject: str = "NSUBJ"
    passive_nominal_subject: str = "NSUBJPASS"
    numeric_modifier: str = "NUMMOD"
    object_predicate: str = "OPRD"
    parataxis: str = "PARATAXIS"
    complement_of_preposition: str = "PCOMP"
    object_of_preposition: str = "POBJ"
    possession_modifier: str = "POSS"
    pre_correlative_conjunction: str = "PRECONJ"
    pre_determiner: str = "PREDET"
    prepositional_modifier: str = "PREP"
    particle: str = "PRT"
    punctuation: str = "PUNCT"
    modifier_of_quantifier: str = "QUANTMOD"
    relative_clause_modifier: str = "RELCL"
    open_clausal_complement: str = "XCOMP"


class EdgeType(Enum):
    sent_to_para: str = "SentToPara"
    token_to_ner: str = "TokenToNER"
    token_to_sent: str = "TokenToSent"
    token_to_uni_pos: str = "TokenToUniPOS"
    dependency_arc: str = "DependencyArc"


@dataclass_json
@dataclass
class BaseEdgeFeats:
    etype: EdgeType
    text: str


@dataclass_json
@dataclass
class EdgeTuples:
    list_edge_tuple: List[EdgeTuple]

    def to_list_edge_tuple(self) -> List[Tuple[int, int, Dict[str, Any]]]:
        untyped_list_edge_tuple: List[Tuple[int, int, Dict[str, Any]]] = [
            edge_tuple.to_tuple() for edge_tuple in self.list_edge_tuple
        ]

        return untyped_list_edge_tuple


@dataclass_json
@dataclass
class EdgeTuple:
    src_id: int
    dst_id: int
    edge_feats: BaseEdgeFeats

    def to_tuple(self) -> Tuple[int, int, Dict[str, Any]]:
        untyped_edge_tuple: Tuple[int, int, Dict[str, Any]] = (
            self.src_id,
            self.dst_id,
            asdict(self.edge_feats, dict_factory=enum_dict_factory),
        )

        return untyped_edge_tuple
