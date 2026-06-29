# Advanced Tutorial: Ontology-Aware Augmentation, Validation & Provenance

**Audience:** Semantic Web researchers and knowledge-graph practitioners who
already understand RDF and have completed the beginner tutorial.
**Time:** ~40 minutes.

This tutorial covers the capabilities that make augmentation *trustworthy* for
research: controlling what gets added, validating semantic correctness with
SHACL, building a SKOS thesaurus, swapping in richer backends, and tracking
provenance so synthetic triples are always distinguishable from original ones.

---

## 1. Controlling augmentation

`RDFAugmenter.augment()` exposes the knobs you need for controlled experiments:

```python
from rdf_augmenter import RDFAugmenter

aug = RDFAugmenter(seed=42)
aug.load("examples/sample_people.ttl")

report = aug.augment(
    predicates=["http://example.org/occupation"],  # which literals are eligible
    relations=("synonym", "hypernym", "related"),    # which relation types
    top_k=4,            # max suggestions requested per (term, relation)
    ratio=0.5,          # augment only 50% of eligible literals (seeded sample)
    attach="subject",   # "subject" (paper-faithful) or "concept" (SKOS thesaurus)
    provenance=True,    # record a PROV-O activity describing the run
    full_provenance=False,  # also reify every generated triple (verbose)
)
```

`ratio` is useful for **ablation studies**: sweep it from `0.1` to `1.0` and
measure how downstream performance changes with augmentation budget. Because the
sample is seeded, each ratio is reproducible.

### Custom predicates

By default, synonyms reuse the source predicate, hypernyms use
`aug:hypernym`, and related terms use `aug:relatedTerm`. Override any of these:

```python
from rdflib import URIRef
report = aug.augment(
    predicates=["http://example.org/occupation"],
    predicate_map={
        "hypernym": URIRef("http://www.w3.org/2004/02/skos/core#broader"),
        "related":  URIRef("http://www.w3.org/2004/02/skos/core#related"),
    },
)
```

## 2. Ontology-aware augmentation with a domain lexicon

The default lexicon covers common occupations. For *your* domain, supply your
own thesaurus so augmentation respects your ontology's vocabulary:

```python
from rdf_augmenter import RDFAugmenter, LexiconBackend

my_lexicon = {
    "myocardial infarction": {
        "synonyms":  ["heart attack"],
        "hypernyms": ["cardiovascular disease", "medical condition"],
        "related":   ["coronary artery", "ischemia", "troponin"],
    },
}
backend = LexiconBackend(lexicon=my_lexicon)        # or LexiconBackend(path="lex.json")
aug = RDFAugmenter(backend=backend, seed=42)
```

Because the lexicon *is* your controlled vocabulary, every term added is one your
ontology already sanctions — this is the simplest form of ontology-aware
augmentation, and it is fully deterministic.

## 3. Building a SKOS thesaurus (`attach="concept"`)

Attaching hypernyms directly to a person (`Person2 aug:hypernym "physician"`) is
convenient but semantically loose. **Concept mode** instead mints a
`skos:Concept` for each distinct literal and connects the concepts to each
other, producing a proper [SKOS](https://www.w3.org/TR/skos-reference/)
thesaurus:

```python
aug = RDFAugmenter(seed=42)
aug.load("examples/sample_people.ttl")
aug.augment(predicates=["http://example.org/occupation"], attach="concept", top_k=3)
print(aug.export(fmt="turtle"))
```

produces, for each occupation, structure like:

```turtle
aug:concept/surgeon a skos:Concept ;
    skos:prefLabel "Surgeon" ;
    skos:altLabel  "operating physician" ;       # synonym
    skos:broader   aug:concept/physician ;        # hypernym
    skos:related   aug:concept/surgery .          # related

ex:Person2 ex:occupation "Surgeon" ;
    aug:concept aug:concept/surgeon .              # link person -> concept
```

This is the representation you want when the augmented graph will feed an
ontology-learning or taxonomy-induction pipeline.

## 4. Validating semantic correctness with SHACL

Augmentation must not silently produce nonsense. [SHACL](https://www.w3.org/TR/shacl/)
(Shapes Constraint Language) lets you *declare* what a valid graph looks like and
check the augmented output against it. Install the validator:

```bash
pip install pyshacl
```

Define a shape — e.g. "every `skos:Concept` must have exactly one
`skos:prefLabel`, and every `skos:broader` value must itself be a `skos:Concept`":

```python
from pyshacl import validate
from rdflib import Graph

shapes = Graph().parse(data="""
@prefix sh:   <http://www.w3.org/ns/shacl#> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .

skos:ConceptShape a sh:NodeShape ;
    sh:targetClass skos:Concept ;
    sh:property [ sh:path skos:prefLabel ; sh:minCount 1 ; sh:maxCount 1 ] ;
    sh:property [ sh:path skos:broader   ; sh:class skos:Concept ] .
""", format="turtle")

conforms, _, text = validate(aug.graph, shacl_graph=shapes, inference="rdfs")
print("conforms:", conforms)
print(text)
```

If augmentation ever violates a shape, `conforms` is `False` and the report
pinpoints the offending triples. Wire this into CI to **guarantee** every
augmented graph you publish is valid against your shapes.

### RDFS / OWL reasoning

You can also check that augmentation *preserves reasoning*. Materialize the
RDFS (or OWL-RL) closure before and after augmentation and confirm no
contradictions were introduced:

```python
import owlrl
owlrl.DeductiveClosure(owlrl.RDFS_Semantics).expand(aug.graph)
```

## 5. Provenance: knowing what is synthetic

Research-grade augmentation must be **traceable**. With `provenance=True`,
every run records a [PROV-O](https://www.w3.org/TR/prov-o/) activity:

```turtle
aug:run-42 a prov:Activity ;
    prov:wasAssociatedWith aug:agent/rdf-augmenter ;
    prov:startedAtTime "2026-..."^^xsd:dateTime ;
    aug:backend "lexicon" ; aug:seed 42 ; aug:topK 4 ;
    aug:triplesAdded 78 .
```

With `full_provenance=True`, *each* generated triple is reified and linked to the
run, so you can filter synthetic triples out at any time:

```python
aug.augment(predicates=["http://example.org/occupation"], full_provenance=True)

# SPARQL: list every triple this tool generated, with its source term
q = """
PREFIX prov: <http://www.w3.org/ns/prov#>
PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX aug:  <https://w3id.org/rdf-augmenter/ns#>
SELECT ?s ?p ?o ?sourceTerm WHERE {
  ?stmt a rdf:Statement ;
        rdf:subject ?s ; rdf:predicate ?p ; rdf:object ?o ;
        prov:wasGeneratedBy ?run ; aug:sourceTerm ?sourceTerm .
}
"""
for row in aug.graph.query(q):
    print(row.s, row.p, row.o, "<-", row.sourceTerm)
```

## 6. Swapping in richer backends

The lexicon backend is deterministic but limited to its vocabulary. For broader
coverage, swap in WordNet or the original BERT method — same interface:

```python
from rdf_augmenter import WordNetBackend, BertBackend

# WordNet: pip install nltk && python -m nltk.downloader wordnet omw-1.4
aug = RDFAugmenter(backend=WordNetBackend(), seed=42)

# BERT (the published method): pip install "rdf-augmenter[bert]"
aug = RDFAugmenter(backend=BertBackend(model_name="bert-base-uncased"), seed=42)
```

Always check availability before relying on an optional backend:

```python
b = WordNetBackend()
print("WordNet ready:", b.is_available())
```

A practical pattern for reproducible experiments: **report results per backend**,
and use the lexicon backend as the deterministic baseline everyone can rerun.

## 7. A reproducible experiment template

```python
from rdf_augmenter import RDFAugmenter

for ratio in (0.25, 0.5, 0.75, 1.0):
    aug = RDFAugmenter(seed=42)
    aug.load("examples/sample_people.ttl")
    rep = aug.augment(predicates=["http://example.org/occupation"], ratio=ratio)
    aug.export(f"aug_ratio_{ratio}.ttl")
    rep.save_manifest(f"manifest_ratio_{ratio}.json")
    print(ratio, "->", rep.added, "triples added")
```

Commit the manifests alongside your results and your experiment is fully
reproducible by anyone.

---

### Recap

You controlled augmentation with predicates, relations, `top_k` and `ratio`;
built a SKOS thesaurus; validated the output with SHACL and RDFS reasoning;
traced every synthetic triple with PROV-O; and swapped backends behind one
interface. These are the building blocks for benchmark-quality, citable
augmentation experiments.
