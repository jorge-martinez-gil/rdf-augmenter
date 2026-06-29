# rdf-augmenter — RDF Knowledge Graph Augmentation Toolkit

**rdf-augmenter** is an open-source toolkit for **RDF augmentation** and
**knowledge graph augmentation**. It enriches sparse RDF knowledge graphs with
semantically related terms — synonyms, hypernyms (types), and related concepts —
so that downstream tasks such as question answering, semantic search, link
prediction, and graph machine learning have richer signal to work with.

It is ontology-aware, reproducible, and produces SHACL-validatable output, making
it suitable both as a practical **knowledge graph preprocessing** tool and as
**semantic web benchmark** infrastructure for research.

## Install

```bash
pip install rdf-augmenter
```

## 30-second example

```python
from rdf_augmenter import RDFAugmenter

aug = RDFAugmenter(seed=42)
aug.load("examples/sample_people.ttl")
report = aug.augment(predicates=["http://example.org/occupation"])
aug.export("augmented.ttl")
print(report.summary())
```

![Graph growth](assets/graph_growth.png)

## Why use it?

- **Ontology-aware augmentation** that respects your vocabulary.
- **Deterministic and reproducible** — every run is seeded and emits a manifest.
- **Provenance-tracked** — synthetic triples are always distinguishable (PROV-O).
- **Standards-friendly** — SKOS output, SHACL validation, RDFS/OWL reasoning.
- **Pluggable backends** — offline lexicon (default), WordNet, or BERT.
- **Interoperable** — export Turtle, N-Triples, RDF/XML, JSON-LD, TriG, N3.

## Where to start

- New to RDF? Begin with the
  [beginner tutorial](tutorials/01-beginner-rdf-augmentation.md).
- Building research-grade pipelines? See the
  [advanced tutorial](tutorials/02-advanced-ontology-aware-augmentation.md).
- Need definitions? The [concepts glossary](concepts.md) explains every term.
- Teaching a course? The [exercises](exercises.md) are runnable and graded.

## Citing

Please cite the associated paper, *Knowledge Graph Augmentation for Increased
Question Answering Accuracy* (Martinez-Gil et al., 2022),
[doi:10.1007/978-3-662-66146-8_3](https://doi.org/10.1007/978-3-662-66146-8_3),
and the software via [`CITATION.cff`](https://github.com/jorge-martinez-gil/rdf-augmenter/blob/main/CITATION.cff).
