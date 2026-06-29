# Beginner Tutorial: Augment Your First RDF Knowledge Graph

**Audience:** anyone new to RDF, knowledge graphs, or data augmentation.
**Time:** ~20 minutes.
**Prerequisites:** Python 3.9+ and `pip install rdf-augmenter`. No GPU, no model
downloads — this tutorial runs entirely on the offline lexicon backend.

By the end you will understand what an RDF knowledge graph is, why you might
want to *augment* one, and how to do it reproducibly in a few lines of Python.

---

## 1. What is an RDF knowledge graph?

A **knowledge graph** stores facts as a network of connected things. In RDF
(the Resource Description Framework, a W3C standard), every fact is a **triple**
made of three parts:

```
subject     predicate      object
ex:Person2  ex:occupation  "Surgeon"
```

Read it like a tiny sentence: *Person2 has the occupation "Surgeon".* A whole
knowledge graph is just a set of these triples. Subjects and predicates are
identified by URIs (web-style identifiers), and objects are either other
resources (URIs) or **literals** — plain values such as strings, numbers, or
dates (`"Surgeon"` is a literal).

Here is the graph we will use, written in **Turtle**, the most human-readable
RDF syntax:

```turtle
@prefix ex: <http://example.org/> .

ex:Person2 ex:name "Alice" ;
           ex:occupation "Surgeon" .
```

The `@prefix` line lets us abbreviate `http://example.org/Person2` as
`ex:Person2`. The semicolon means "same subject, another predicate".

## 2. Why augment a knowledge graph?

Real knowledge graphs are almost always **incomplete and sparse**. A graph might
know that Alice is a "Surgeon" but not that a surgeon *is a kind of* physician,
*is also called* an operating physician, or *is related to* surgery and
hospitals. That missing context hurts downstream tasks:

- A **question-answering** system that is asked "Which people are doctors?"
  cannot connect "Surgeon" to "doctor".
- A **semantic search** engine misses documents that say "physician" when the
  graph only contains "Surgeon".
- A **graph machine-learning** model has too few examples of rare relations to
  learn from.

**Augmentation** adds new, plausible triples to fill these gaps — synonyms,
broader types (hypernyms), and related concepts — so the graph becomes denser
and more useful, without you hand-writing every fact. This is the idea behind
Martinez-Gil et al. (2022), *Knowledge Graph Augmentation for Increased Question
Answering Accuracy*, which this toolkit grew out of.

## 3. Your first augmentation

Create a file `people.ttl`:

```turtle
@prefix ex: <http://example.org/> .

ex:Person1 ex:name "John"    ; ex:occupation "Data Scientist" .
ex:Person2 ex:name "Alice"   ; ex:occupation "Surgeon" .
ex:Person3 ex:name "Michael" ; ex:occupation "Architect" .
```

Then run:

```python
from rdf_augmenter import RDFAugmenter

aug = RDFAugmenter(seed=42)              # a seed makes the run reproducible
aug.load("people.ttl")

report = aug.augment(
    predicates=["http://example.org/occupation"],   # only augment occupations
    relations=("synonym", "hypernym", "related"),     # the three relation types
    top_k=4,                                           # up to 4 suggestions each
)

print(report.summary())
aug.export("people_augmented.ttl")
```

You will see something like:

```
RDF augmentation report
-----------------------
backend            : lexicon
seed               : 42
triples before     : 6
triples after      : 36
triples added      : 30
  - hypernym      : 9
  - related       : 12
  - synonym       : 9
```

Open `people_augmented.ttl` and you will find new facts such as:

```turtle
ex:Person2 ex:name "Alice" ;
    ex:occupation "Surgeon", "operating physician" ;   # synonym
    aug:hypernym "physician", "doctor" ;                # broader type
    aug:relatedTerm "surgery", "anatomy", "hospital" .  # related concepts
```

## 4. Understanding the three relation types

| Relation   | Question it answers              | Example for "Surgeon"          |
|------------|----------------------------------|--------------------------------|
| `synonym`  | What else is this called?        | operating physician            |
| `hypernym` | What broader type is this?       | physician, doctor              |
| `related`  | What concepts go with this?      | surgery, anatomy, hospital     |

You can ask for any subset. For example, to add only broader types:

```python
report = aug.augment(predicates=["http://example.org/occupation"],
                     relations=("hypernym",))
```

## 5. Reproducibility: the seed and the manifest

Science needs runs that other people can repeat. Two things make that easy here:

1. **The `seed`.** Re-running with the same seed and parameters produces the
   same graph (when provenance timestamps are off):

   ```python
   a = RDFAugmenter(seed=42); a.load("people.ttl")
   a.augment(predicates=["http://example.org/occupation"], provenance=False)
   ```

2. **The manifest.** Save a JSON record of exactly what happened — input hash,
   parameters, counts, and library versions:

   ```python
   report.save_manifest("run_manifest.json")
   ```

Anyone with `people.ttl`, the manifest, and `rdf-augmenter` can reproduce your
result exactly.

## 6. Exporting to other formats

RDF has several interchange syntaxes. Export to whichever your downstream tool
expects:

```python
aug.export("graph.ttl",    fmt="turtle")     # Turtle
aug.export("graph.nt",     fmt="ntriples")   # N-Triples
aug.export("graph.rdf",    fmt="xml")        # RDF/XML
aug.export("graph.jsonld", fmt="json-ld")    # JSON-LD
aug.export("graph.trig",   fmt="trig")       # TriG
```

## 7. Where to go next

- **`02-advanced-ontology-aware-augmentation.md`** — control augmentation with
  ontologies, validate the output with SHACL, build a SKOS thesaurus, and track
  per-triple provenance.
- **`notebooks/01_quickstart.ipynb`** — this tutorial as a runnable notebook.
- **`notebooks/02_visualizing_augmentation.ipynb`** — chart the augmentation
  effect.
- **`docs/concepts.md`** — a plain-language glossary of every term used here.

---

### Recap

You loaded an RDF graph, augmented selected literals with synonyms, hypernyms,
and related terms, exported the result, and recorded a reproducibility manifest
— all offline and deterministically. That is the core workflow; everything else
in this toolkit builds on it.
