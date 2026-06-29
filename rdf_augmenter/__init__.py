"""rdf-augmenter: ontology-aware augmentation of RDF knowledge graphs.

The package turns a small RDF graph into a richer one by proposing
semantically related terms (synonyms, hypernyms / types, and related concepts)
for selected literals, and adding them back as new triples with full
provenance.

Design goals
------------
* **Runs out of the box.** The default :class:`LexiconBackend` is offline and
  deterministic, so tutorials, notebooks and CI execute in seconds with no GPU
  and no model downloads.
* **Pluggable.** Heavier, richer backends (:class:`WordNetBackend`,
  :class:`BertBackend`) implement the same interface and can be swapped in with
  one line.
* **Reproducible.** Every run takes an explicit ``seed`` and can emit a JSON
  reproducibility manifest describing inputs, parameters and library versions.
* **Traceable.** Generated triples can be annotated with W3C PROV-O
  provenance so you always know which triples were synthetic.

Quick start
-----------
>>> from rdf_augmenter import RDFAugmenter
>>> aug = RDFAugmenter(seed=42)                      # offline lexicon backend
>>> aug.load("examples/sample_people.ttl")           # doctest: +SKIP
>>> report = aug.augment(predicates=["http://example.org/occupation"])
>>> aug.export("augmented.ttl")                       # doctest: +SKIP
>>> print(report.added)                               # doctest: +SKIP
"""

from .backends import (
    Backend,
    BertBackend,
    LexiconBackend,
    WordNetBackend,
    RELATION_TYPES,
)
from .augmenter import RDFAugmenter, AugmentationReport, AUG

__all__ = [
    "RDFAugmenter",
    "AugmentationReport",
    "Backend",
    "LexiconBackend",
    "WordNetBackend",
    "BertBackend",
    "RELATION_TYPES",
    "AUG",
    "__version__",
]

__version__ = "0.2.0"
