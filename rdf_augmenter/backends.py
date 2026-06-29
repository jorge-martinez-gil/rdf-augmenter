"""Augmentation backends.

A *backend* answers a single question: given a term and a relation type, what
are the most plausible related terms? All backends share the :class:`Backend`
interface so they are interchangeable.

============  ===========================  ==================================
Backend       Method                       Extra dependencies
============  ===========================  ==================================
Lexicon       Curated thesaurus lookup     none (ships with the package)
WordNet       Princeton WordNet relations  ``nltk`` + WordNet corpus
Bert          BERT fill-mask + embeddings  ``transformers`` + ``torch``
============  ===========================  ==================================

The lexicon backend is the default because it is offline, deterministic and
instant -- ideal for teaching, reproducible experiments and CI. The other two
trade reproducibility and install weight for broader vocabulary coverage.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

#: Relation types every backend understands.
#:
#: * ``"synonym"``  -- an alternative label for the same concept
#: * ``"hypernym"`` -- a broader concept / type ("is a kind of")
#: * ``"related"``  -- an associated but distinct concept
RELATION_TYPES = ("synonym", "hypernym", "related")

_KEY = {"synonym": "synonyms", "hypernym": "hypernyms", "related": "related"}


class Backend(ABC):
    """Interface implemented by every augmentation backend."""

    #: Short identifier recorded in provenance and manifests.
    name = "base"

    @abstractmethod
    def suggest(self, term: str, relation_type: str, top_k: int = 5) -> List[str]:
        """Return up to ``top_k`` related terms for ``term``.

        Implementations must never return ``term`` itself and should return an
        empty list when they have nothing to offer rather than raising.
        """

    def is_available(self) -> bool:
        """Whether the backend's dependencies/data are present."""
        return True

    def describe(self) -> Dict[str, str]:
        """Human-readable description, embedded in reproducibility manifests."""
        return {"name": self.name, "class": type(self).__name__}


class LexiconBackend(Backend):
    """Deterministic, offline backend backed by a curated thesaurus.

    The thesaurus is a JSON mapping ``term -> {synonyms, hypernyms, related}``.
    A small but useful default lexicon ships with the package; pass your own
    with ``lexicon=`` (a dict) or ``path=`` (a JSON file) to cover your domain.

    Because lookups are pure dictionary access with a stable order, results are
    fully reproducible and require no network, model download or GPU.
    """

    name = "lexicon"

    def __init__(
        self,
        lexicon: Optional[Dict[str, dict]] = None,
        path: Optional[str] = None,
    ):
        if lexicon is not None:
            self._lex = {k.lower(): v for k, v in lexicon.items()}
        else:
            self._lex = self._load_default(path)
        self._source = path or "packaged default lexicon"

    @staticmethod
    def _load_default(path: Optional[str]) -> Dict[str, dict]:
        if path:
            with open(path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
        else:
            try:
                from importlib.resources import files

                text = (
                    files("rdf_augmenter.data")
                    .joinpath("lexicon.json")
                    .read_text(encoding="utf-8")
                )
            except Exception:  # pragma: no cover - fallback for odd layouts
                import os

                here = os.path.join(os.path.dirname(__file__), "data", "lexicon.json")
                with open(here, "r", encoding="utf-8") as fh:
                    text = fh.read()
            raw = json.loads(text)
        return {k.lower(): v for k, v in raw.items()}

    def suggest(self, term: str, relation_type: str, top_k: int = 5) -> List[str]:
        entry = self._lex.get(str(term).lower())
        if not entry:
            return []
        values = entry.get(_KEY.get(relation_type, relation_type), [])
        out: List[str] = []
        seen = set()
        for value in values:
            v = str(value).strip()
            low = v.lower()
            if v and low != str(term).lower() and low not in seen:
                out.append(v)
                seen.add(low)
            if len(out) >= top_k:
                break
        return out

    def terms(self) -> List[str]:
        """Sorted list of terms covered by the lexicon."""
        return sorted(self._lex.keys())

    def describe(self) -> Dict[str, str]:
        d = super().describe()
        d.update(source=self._source, n_terms=str(len(self._lex)))
        return d


class WordNetBackend(Backend):
    """Backend backed by Princeton WordNet via NLTK (optional dependency).

    Synonyms come from the lemmas of a term's synsets, hypernyms from the
    WordNet hypernym hierarchy, and related terms from sister terms (other
    hyponyms of a shared hypernym). Requires ``nltk`` and the ``wordnet``
    corpus (``python -m nltk.downloader wordnet omw-1.4``).
    """

    name = "wordnet"

    def __init__(self, pos: Optional[str] = None):
        self.pos = pos  # e.g. "n" to restrict to nouns
        self._wn = None

    def is_available(self) -> bool:
        try:
            from nltk.corpus import wordnet as wn

            wn.synsets("test")
            return True
        except Exception:
            return False

    def _wordnet(self):
        if self._wn is None:
            from nltk.corpus import wordnet as wn

            self._wn = wn
        return self._wn

    @staticmethod
    def _clean(name: str) -> str:
        return name.replace("_", " ").strip()

    def suggest(self, term: str, relation_type: str, top_k: int = 5) -> List[str]:
        wn = self._wordnet()
        synsets = wn.synsets(str(term).replace(" ", "_"), pos=self.pos)
        if not synsets:
            return []
        primary = synsets[0]
        out: List[str] = []
        seen = {str(term).lower()}

        def push(name: str):
            cleaned = self._clean(name)
            if cleaned and cleaned.lower() not in seen:
                out.append(cleaned)
                seen.add(cleaned.lower())

        if relation_type == "synonym":
            for syn in synsets:
                for lemma in syn.lemmas():
                    push(lemma.name())
        elif relation_type == "hypernym":
            for hyper in primary.hypernyms():
                for lemma in hyper.lemmas():
                    push(lemma.name())
        elif relation_type == "related":
            for hyper in primary.hypernyms():
                for sister in hyper.hyponyms():
                    if sister != primary:
                        for lemma in sister.lemmas():
                            push(lemma.name())
        return out[:top_k]


class BertBackend(Backend):
    """Contextual backend using BERT fill-mask + embedding filtering.

    This reproduces the method from Martinez-Gil et al. (2022): masked-language
    templates generate candidate terms, which are then filtered by cosine
    similarity of their BERT embeddings to the source term. It is contextual
    and broad but **non-deterministic across environments** and heavy to
    install (``transformers`` + ``torch``), so it is opt-in rather than the
    default.

    Note: this backend's similarity-threshold filtering makes ``top_k`` an
    upper bound on the *candidates considered*, not a guaranteed output size.
    """

    name = "bert"

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        similarity_threshold: float = 0.85,
    ):
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self._nlp = None
        self._embed_model = None
        self._embed_tok = None

    def is_available(self) -> bool:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401

            return True
        except Exception:
            return False

    def _ensure(self):
        if self._nlp is None:
            from transformers import (
                AutoModelForMaskedLM,
                AutoTokenizer,
                BertModel,
                BertTokenizer,
                pipeline,
            )

            tok = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForMaskedLM.from_pretrained(self.model_name)
            self._nlp = pipeline("fill-mask", model=model, tokenizer=tok)
            self._embed_model = BertModel.from_pretrained(self.model_name)
            self._embed_tok = BertTokenizer.from_pretrained(self.model_name)

    def _embedding(self, text: str):
        inputs = self._embed_tok(text, return_tensors="pt")
        outputs = self._embed_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).detach().numpy()

    def suggest(self, term: str, relation_type: str, top_k: int = 5) -> List[str]:
        self._ensure()
        from sklearn.metrics.pairwise import cosine_similarity

        if relation_type == "hypernym":
            masked = f"{term} is a type of [MASK] in a professional context."
        elif relation_type == "related":
            masked = f"{term} is closely related to [MASK] in the field of {term.lower()}."
        else:  # synonym
            masked = f"{term} can also be called a [MASK]."

        results = self._nlp(masked, top_k=top_k)
        candidates = [
            r["token_str"].strip().lower()
            for r in results
            if r["token_str"].strip().lower() != str(term).lower()
        ]

        term_vec = self._embedding(term)
        kept: List[str] = []
        for cand in candidates:
            sim = cosine_similarity(term_vec, self._embedding(cand))[0][0]
            if sim > self.similarity_threshold:
                kept.append(cand)
        return kept

    def describe(self) -> Dict[str, str]:
        d = super().describe()
        d.update(
            model_name=self.model_name,
            similarity_threshold=str(self.similarity_threshold),
        )
        return d
