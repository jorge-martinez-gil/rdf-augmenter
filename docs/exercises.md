# Practical Exercises

A set of graded exercises for self-study or classroom use. Each builds on the
tutorials. Solutions are collapsed at the end of each exercise — try first, then
check. All exercises run on the offline lexicon backend (no GPU, no downloads).

Setup once:

```bash
pip install rdf-augmenter
```

```python
from rdf_augmenter import RDFAugmenter, LexiconBackend
SAMPLE = "examples/sample_people.ttl"
```

---

## Exercise 1 — Count before and after (warm-up)

Load the sample graph, augment only the `occupation` predicate with **hypernyms
only**, and print how many triples the graph had before and after.

<details><summary>Solution</summary>

```python
aug = RDFAugmenter(seed=42)
aug.load(SAMPLE)
report = aug.augment(predicates=["http://example.org/occupation"],
                     relations=("hypernym",))
print("before:", report.triples_before, "after:", report.triples_after)
print("added :", report.added)
```
</details>

## Exercise 2 — Reproducibility

Show that two runs with the same seed produce byte-identical N-Triples output
(turn provenance off so timestamps don't differ), but different seeds with
`ratio=0.5` can differ.

<details><summary>Solution</summary>

```python
def run(seed, ratio=0.5):
    a = RDFAugmenter(seed=seed); a.load(SAMPLE)
    a.augment(predicates=["http://example.org/occupation"],
              ratio=ratio, provenance=False)
    return a.export(fmt="nt")

assert run(42) == run(42)          # identical
print("seed 42 reproducible:", run(42) == run(42))
print("42 vs 7 differ:", run(42) != run(7))
```
</details>

## Exercise 3 — A domain lexicon

Write a small lexicon for two terms in *your* field (e.g. "neural network",
"transformer"), build a `LexiconBackend` from it, augment a one-triple graph,
and inspect the output.

<details><summary>Solution</summary>

```python
lex = {
  "neural network": {"synonyms": ["artificial neural network", "neural net"],
                      "hypernyms": ["model", "function approximator"],
                      "related": ["backpropagation", "neurons", "deep learning"]},
}
aug = RDFAugmenter(backend=LexiconBackend(lexicon=lex), seed=1)
aug.load(data='@prefix ex: <http://example.org/> . '
              'ex:m1 ex:method "neural network" .', fmt="turtle")
aug.augment(predicates=["http://example.org/method"])
print(aug.export(fmt="turtle"))
```
</details>

## Exercise 4 — Build and inspect a SKOS thesaurus

Use `attach="concept"` to turn the occupations into a SKOS thesaurus. Then write
a SPARQL query that lists every `skos:broader` pair (concept → its broader
concept) by their `skos:prefLabel`.

<details><summary>Solution</summary>

```python
aug = RDFAugmenter(seed=42); aug.load(SAMPLE)
aug.augment(predicates=["http://example.org/occupation"], attach="concept", top_k=3)

q = """
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
SELECT ?narrow ?broad WHERE {
  ?c  skos:prefLabel ?narrow ; skos:broader ?b .
  ?b  skos:prefLabel ?broad .
}
"""
for row in aug.graph.query(q):
    print(f"{row.narrow}  ->  {row.broad}")
```
</details>

## Exercise 5 — Validate with SHACL

Install `pyshacl`. Write a shape asserting that every `skos:Concept` has at least
one `skos:prefLabel`, validate the concept-mode graph, and confirm it conforms.

<details><summary>Solution</summary>

```python
from pyshacl import validate
from rdflib import Graph

aug = RDFAugmenter(seed=42); aug.load(SAMPLE)
aug.augment(predicates=["http://example.org/occupation"], attach="concept")

shapes = Graph().parse(data="""
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .
[] a sh:NodeShape ; sh:targetClass skos:Concept ;
   sh:property [ sh:path skos:prefLabel ; sh:minCount 1 ] .
""", format="turtle")

conforms, _, _ = validate(aug.graph, shacl_graph=shapes)
print("conforms:", conforms)   # expect True
```
</details>

## Exercise 6 — A `top_k` ablation (mini-experiment)

For `top_k` in 1..5, record the number of triples added, save a manifest per
run, and print a small table. (This reproduces the figure in
`notebooks/02_visualizing_augmentation.ipynb`.)

<details><summary>Solution</summary>

```python
print(f"{'top_k':>5} {'added':>6}")
for k in range(1, 6):
    a = RDFAugmenter(seed=42); a.load(SAMPLE)
    r = a.augment(predicates=["http://example.org/occupation"],
                  top_k=k, provenance=False)
    r.save_manifest(f"manifest_topk_{k}.json")
    print(f"{k:>5} {r.added:>6}")
```
</details>

## Exercise 7 — Trace provenance (advanced)

Run with `full_provenance=True`, then write a SPARQL query that returns only the
triples this tool generated, together with the source term each came from.

<details><summary>Solution</summary>

```python
aug = RDFAugmenter(seed=42); aug.load(SAMPLE)
aug.augment(predicates=["http://example.org/occupation"], full_provenance=True)

q = """
PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX prov: <http://www.w3.org/ns/prov#>
PREFIX aug:  <https://w3id.org/rdf-augmenter/ns#>
SELECT ?o ?sourceTerm WHERE {
  ?stmt a rdf:Statement ; rdf:object ?o ;
        prov:wasGeneratedBy ?run ; aug:sourceTerm ?sourceTerm .
} ORDER BY ?sourceTerm LIMIT 15
"""
for row in aug.graph.query(q):
    print(f'{row.sourceTerm:<16} -> {row.o}')
```
</details>

---

### For instructors

These exercises map onto a typical Semantic Web / knowledge-engineering syllabus:
exercises 1–3 cover RDF basics and reproducibility; exercise 4 covers SKOS and
SPARQL; exercise 5 covers SHACL validation; exercises 6–7 introduce reproducible
experimentation and provenance. Each is self-contained and runs in seconds.
