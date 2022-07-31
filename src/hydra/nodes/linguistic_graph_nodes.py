from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Tuple

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class UniversalPOSTag(Enum):
    adjective: str = "ADJ"
    adposition: str = "ADP"
    adverb: str = "ADV"
    auxiliary: str = "AUX"
    coordinating_conjunction: str = "CCONJ"
    determiner: str = "DET"
    interjection: str = "INTJ"
    noun: str = "NOUN"
    numeral: str = "NUM"
    particle: str = "PART"
    pronoun: str = "PRON"
    proper_noun: str = "PROPN"
    punctuation: str = "PUNCT"
    subordinating_conjunction: str = "SCONJ"
    symbol: str = "SYM"
    ver: str = "VERB"
    other: str = "X"


@dataclass_json
@dataclass
class NamedEntityLabel(Enum):
    cardinal: str = "CARDINAL"
    date: str = "DATE"
    event: str = "EVENT"
    fac: str = "FAC"
    gpe: str = "GPE"
    language: str = "LANGUAGE"
    law: str = "LAW"
    loc: str = "LOC"
    money: str = "MONEY"
    norp: str = "NORP"
    ordinal: str = "ORDINAL"
    org: str = "ORG"
    percent: str = "PERCENT"
    person: str = "PERSON"
    product: str = "PRODUCT"
    quantity: str = "QUANTITY"
    time: str = "TIME"
    work_of_art: str = "WORK_OF_ART"


@dataclass_json
@dataclass
class NodeFeats:
    ntype: NodeType
    text: str


@dataclass_json
@dataclass
class NodeType(Enum):
    paragraph: str = "PARAGRAPH"
    sentence: str = "SENTENCE"
    token: str = "TOKEN"
    uni_pos: str = "UNIVERSALPOS"
    ner: str = "NER"


@dataclass_json
@dataclass
class NodeTuples:
    list_node_tuple: List[NodeTuple]

    def to_list_tuple(self) -> List[Tuple[int, Dict[str, Any]]]:
        untyped_list_node_tuple: List[Tuple[int, Dict[str, Any]]] = [
            node_tuple.to_tuple() for node_tuple in self.list_node_tuple
        ]

        return untyped_list_node_tuple


@dataclass_json
@dataclass
class NodeTuple:
    node_id: int
    node_feats: NodeFeats

    def to_tuple(self) -> Tuple[int, Dict[str, Any]]:
        untyped_node_tuple: Tuple[int, Dict[str, Any]] = (
            self.node_id,
            asdict(self.node_feats),
        )

        return untyped_node_tuple
