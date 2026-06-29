# Concepts Glossary

A plain-language reference for the ideas behind `rdf-augmenter`. Use it as a
companion to the tutorials, or as lecture material for a Semantic Web, knowledge
engineering, or graph machine-learning course. Each entry is short, with a
pointer to where the toolkit uses the idea.

## RDF (Resource Description Framework)

A W3C standard for representing information as **triples**:
`(subject, predicate, object)`. Subjects and predicates are URIs; objects are
URIs or literal values. RDF is the lingua franca of the Semantic Web because it
lets independently published data link together. *In this toolkit:* the input
and output are RDF graphs, handled via `rdflib`.

## Knowledge graph

A graph-structured knowledge base: nodes are entities (people, places,
concepts) and edges are typed relationships. RDF is one common way to serialize
a knowledge graph. Famous public examples include **DBpedia**, **Wikidata**,
**YAGO**, and **Bio2RDF**. *In this toolkit:* the thing you augment.

## Triple

The atomic unit of RDF. `(ex:Person2, ex:occupation, "Surgeon")` states one
fact. *In this toolkit:* augmentation works by adding new, plausible triples.

## Literal

A plain data value (string, number, date) appearing as a triple's object, e.g.
`"Surgeon"`. Literals are exactly what `rdf-augmenter` reads when it looks for
terms to augment. *In this toolkit:* the `predicates=` argument selects which
literals are eligible.

## Ontology

A formal specification of the classes, properties, and constraints in a domain —
the "schema" of a knowledge graph. Ontologies let machines reason about data
(e.g. "every Surgeon is a Physician"). *In this toolkit:* supplying a domain
lexicon, or validating with SHACL/OWL, keeps augmentation **ontology-aware**.

## Hypernym / Hyponym

A **hypernym** is a broader term ("physician" is a hypernym of "surgeon"); a
**hyponym** is the narrower one. The hypernym relation forms the backbone of
taxonomies. *In this toolkit:* the `hypernym` relation type adds "is a kind of"
links.

## Synonym

A different label for the same concept ("attorney" ≈ "lawyer"). *In this
toolkit:* the `synonym` relation type adds alternative labels (`skos:altLabel`).

## SKOS (Simple Knowledge Organization System)

A W3C vocabulary for thesauri and taxonomies, with properties like
`skos:prefLabel`, `skos:altLabel`, `skos:broader`, and `skos:related`. *In this
toolkit:* `attach="concept"` mode builds a SKOS thesaurus from your literals.

## SPARQL

The W3C query language for RDF. It uses graph patterns to retrieve and
manipulate triples, much as SQL queries tables. *In this toolkit:* the advanced
tutorial uses SPARQL to list every synthetic triple via its provenance.

## OWL (Web Ontology Language)

A richer ontology language layered on RDF that supports classes, property
characteristics, and logical axioms enabling **reasoning** (inferring new facts).
*In this toolkit:* you can run OWL-RL closure to check augmentation preserves
consistency.

## RDFS (RDF Schema)

A lightweight ontology vocabulary (`rdfs:subClassOf`, `rdfs:domain`, etc.) for
basic class/property hierarchies and simple inference. *In this toolkit:* SHACL
validation can run with `inference="rdfs"` to respect subclass relationships.

## SHACL (Shapes Constraint Language)

A W3C standard for **validating** RDF graphs against declared "shapes"
(constraints). It answers "is this graph well-formed?" and reports violations.
*In this toolkit:* the recommended way to guarantee an augmented graph is
semantically valid before you publish or benchmark with it.

## Graph augmentation

Adding new, plausible nodes/edges to a graph to make it denser and more useful,
analogous to data augmentation in computer vision. It can improve downstream
link prediction, classification, search, and question answering. *In this
toolkit:* the core operation, via synonyms, hypernyms, and related terms.

## Graph completion (knowledge graph completion)

Predicting **missing** true triples in an existing graph (e.g. via embeddings or
GNNs). Augmentation is related but broader: it may add new terms and structure,
not only complete known patterns. *In this toolkit:* augmented graphs are useful
training/evaluation inputs for completion models.

## Negative triples

Plausible-but-false triples used as negative examples when training or
evaluating link-prediction models. Quality negatives are notoriously important.
*In this toolkit:* a planned capability on the augmentation roadmap.

## Provenance (PROV-O)

Metadata describing *how* data came to be — who/what produced it, when, and how.
The W3C **PROV-O** ontology standardizes this. *In this toolkit:* every run can
record a `prov:Activity`, and every synthetic triple can be reified and linked
to it, so original and generated data are always distinguishable.

## Reproducibility

The ability for others to obtain the same result from the same inputs and
procedure. Requires fixed seeds, recorded parameters, and pinned dependencies.
*In this toolkit:* the `seed` argument plus the JSON **manifest** make every run
reproducible.

## Backend

An interchangeable component that proposes related terms. *In this toolkit:*
`LexiconBackend` (offline, deterministic, default), `WordNetBackend` (optional),
and `BertBackend` (the published contextual method, optional).

---

*See also:* the [W3C Semantic Web standards](https://www.w3.org/standards/semanticweb/)
for the authoritative specifications of RDF, RDFS, OWL, SKOS, SPARQL, and SHACL.
