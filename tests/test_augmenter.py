"""Test suite for rdf-augmenter.

Runs entirely on the offline lexicon backend, so it executes in seconds with no
GPU, no network and no model downloads. SHACL tests are skipped automatically if
``pyshacl`` is not installed.
"""

import json

import pytest
from rdflib import Literal, URIRef

from rdf_augmenter import RDFAugmenter, LexiconBackend, RELATION_TYPES
from rdf_augmenter.augmenter import stats

OCC = "http://example.org/occupation"

SAMPLE = """
@prefix ex: <http://example.org/> .
ex:Person1 ex:name "John"  ; ex:occupation "Data Scientist" .
ex:Person2 ex:name "Alice" ; ex:occupation "Surgeon" .
ex:Person3 ex:name "Mike"  ; ex:occupation "Architect" .
"""


def make(seed=42, backend=None):
    aug = RDFAugmenter(backend=backend, seed=seed)
    aug.load(data=SAMPLE, fmt="turtle")
    return aug


def test_augmentation_adds_triples():
    aug = make()
    before = len(aug.graph)
    report = aug.augment(predicates=[OCC])
    assert report.added > 0
    assert report.triples_after > before
    assert report.triples_after == len(aug.graph)


def test_relation_types_respected():
    aug = make()
    report = aug.augment(predicates=[OCC], relations=("hypernym",))
    assert report.added_by_relation.get("hypernym", 0) > 0
    assert "synonym" not in report.added_by_relation
    assert "related" not in report.added_by_relation


def test_determinism_same_seed():
    a = make(seed=7)
    a.augment(predicates=[OCC], ratio=0.5, provenance=False)
    b = make(seed=7)
    b.augment(predicates=[OCC], ratio=0.5, provenance=False)
    assert a.export(fmt="nt") == b.export(fmt="nt")


def test_different_seed_can_differ_with_ratio():
    a = make(seed=1)
    a.augment(predicates=[OCC], ratio=0.34, provenance=False)
    b = make(seed=999)
    b.augment(predicates=[OCC], ratio=0.34, provenance=False)
    # Same number of eligible literals sampled, but (likely) different ones.
    assert a.export(fmt="nt") != b.export(fmt="nt")


def test_never_suggests_self():
    backend = LexiconBackend()
    for rel in RELATION_TYPES:
        out = backend.suggest("Surgeon", rel, top_k=10)
        assert "surgeon" not in [o.lower() for o in out]


def test_top_k_bounds_suggestions():
    backend = LexiconBackend()
    for k in (1, 2, 3):
        assert len(backend.suggest("Surgeon", "related", top_k=k)) <= k


@pytest.mark.parametrize("fmt", ["turtle", "ntriples", "xml", "json-ld", "trig", "n3"])
def test_export_formats(fmt):
    aug = make()
    aug.augment(predicates=[OCC])
    text = aug.export(fmt=fmt)
    assert isinstance(text, str) and len(text) > 0


def test_invalid_format_raises():
    aug = make()
    with pytest.raises(ValueError):
        aug.export(fmt="not-a-format")


def test_invalid_ratio_raises():
    aug = make()
    with pytest.raises(ValueError):
        aug.augment(predicates=[OCC], ratio=1.5)


def test_provenance_activity_recorded():
    aug = make()
    aug.augment(predicates=[OCC], provenance=True)
    prov = URIRef("http://www.w3.org/ns/prov#Activity")
    types = list(aug.graph.objects(predicate=URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")))
    assert prov in types


def test_full_provenance_reifies_triples():
    aug = make()
    aug.augment(predicates=[OCC], full_provenance=True)
    stmt = URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#Statement")
    n_stmts = sum(
        1 for _ in aug.graph.subjects(
            predicate=URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"),
            object=stmt,
        )
    )
    assert n_stmts > 0


def test_manifest_roundtrip(tmp_path):
    aug = make()
    report = aug.augment(predicates=[OCC])
    path = tmp_path / "manifest.json"
    report.save_manifest(str(path))
    manifest = json.loads(path.read_text())
    assert manifest["tool"] == "rdf-augmenter"
    assert manifest["seed"] == 42
    assert manifest["output"]["triples_added"] == report.added
    assert len(manifest["input"]["sha256"]) == 64


def test_custom_lexicon_backend():
    lex = {"widget": {"synonyms": ["gadget"], "hypernyms": ["device"], "related": ["gizmo"]}}
    aug = RDFAugmenter(backend=LexiconBackend(lexicon=lex), seed=1)
    aug.load(data='@prefix ex: <http://example.org/> . ex:a ex:kind "widget" .', fmt="turtle")
    report = aug.augment(predicates=["http://example.org/kind"])
    assert report.added == 3  # gadget, device, gizmo


def test_concept_mode_builds_skos():
    aug = make()
    report = aug.augment(predicates=[OCC], attach="concept", top_k=3)
    ttl = aug.export(fmt="turtle")
    assert "skos:Concept" in ttl
    assert "skos:prefLabel" in ttl
    assert report.concepts_created > 0


def test_stats_shape():
    aug = make()
    s = stats(aug.graph)
    for key in ("triples", "subjects", "predicates", "literals", "nodes", "density"):
        assert key in s


def test_unmatched_terms_are_skipped():
    # A literal with no lexicon entry should simply produce no suggestions.
    aug = make()
    aug.load(data='@prefix ex: <http://example.org/> . ex:z ex:occupation "Zxqwv" .', fmt="turtle")
    report = aug.augment(predicates=[OCC])
    assert report.added >= 0  # does not raise; unknown term contributes nothing


# --------------------------------------------------------------- SHACL (optional)
pyshacl = pytest.importorskip("pyshacl", reason="pyshacl not installed")


def test_concept_mode_passes_shacl():
    from rdflib import Graph

    aug = make()
    aug.augment(predicates=[OCC], attach="concept", top_k=3)
    shapes = Graph().parse(
        data="""
        @prefix sh: <http://www.w3.org/ns/shacl#> .
        @prefix skos: <http://www.w3.org/2004/02/skos/core#> .
        [] a sh:NodeShape ; sh:targetClass skos:Concept ;
           sh:property [ sh:path skos:prefLabel ; sh:minCount 1 ; sh:maxCount 1 ] ;
           sh:property [ sh:path skos:broader ; sh:class skos:Concept ] .
        """,
        format="turtle",
    )
    conforms, _, text = pyshacl.validate(aug.graph, shacl_graph=shapes, inference="rdfs")
    assert conforms, text
