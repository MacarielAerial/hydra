from __future__ import annotations

from logging import Logger
from pathlib import Path

import en_core_web_sm
from pytest import fixture
from spacy.language import Language
from spacy.tokens import Doc, Span

from src.hydra.nodes.base_logger import get_base_logger


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
