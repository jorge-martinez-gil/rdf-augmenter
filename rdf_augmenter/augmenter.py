"""The :class:`RDFAugmenter` orchestrator and its reporting helpers."""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import platform
import random
import sys
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Union

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import PROV, RDF, RDFS, SKOS, XSD

from .backends import Backend, LexiconBackend, RELATION_TYPES

#: Namespace for terms minted by rdf-augmenter (predicates and the run agent).
AUG = Namespace("https://w3id.org/rdf-augmenter/ns#")

#: How each relation type is attached to the *subject* in ``attach="subject"``.
DEFAULT_PREDICATE_MAP = {
    "synonym": SKOS.altLabel,
    "hypernym": AUG.hypernym,
    "related": AUG.relatedTerm,
}

#: rdflib serialization formats exposed by :meth:`RDFAugmenter.export`.
EXPORT_FORMATS = {
    "turtle": "ttl",
    "ntriples": "nt",
    "nt": "nt",
    "xml": "pretty-xml",
    "rdfxml": "pretty-xml",
    "json-ld": "json-ld",
    "jsonld": "json-ld",
    "trig": "trig",
    "n3": "n3",
}


@dataclass
class AugmentationReport:
    """Outcome of one :meth:`RDFAugmenter.augment` call.

    Carries before/after statistics, per-relation counts and everything needed
    to write a reproducibility manifest.
    """

    backend: str
    seed: int
    params: Dict = field(default_factory=dict)
    triples_before: int = 0
    triples_after: int = 0
    added: int = 0
    added_by_relation: Dict[str, int] = field(default_factory=dict)
    concepts_created: int = 0
    input_sha256: str = ""
    stats_before: Dict = field(default_factory=dict)
    stats_after: Dict = field(default_factory=dict)
    timestamp: str = ""

    def to_dict(self) -> Dict:
        return {
            "tool": "rdf-augmenter",
            "tool_version": _version(),
            "timestamp_utc": self.timestamp,
            "backend": self.backend,
            "seed": self.seed,
            "parameters": self.params,
            "input": {
                "sha256": self.input_sha256,
                "triples": self.triples_before,
                "stats": self.stats_before,
            },
            "output": {
                "triples": self.triples_after,
                "triples_added": self.added,
                "added_by_relation": self.added_by_relation,
                "concepts_created": self.concepts_created,
                "stats": self.stats_after,
            },
            "environment": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "rdflib": _rdflib_version(),
            },
        }

    def save_manifest(self, path: str) -> None:
        """Write the reproducibility manifest to ``path`` as JSON."""
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False)

    def summary(self) -> str:
        lines = [
            "RDF augmentation report",
            "-----------------------",
            f"backend            : {self.backend}",
            f"seed               : {self.seed}",
            f"triples before     : {self.triples_before}",
            f"triples after      : {self.triples_after}",
            f"triples added      : {self.added}",
        ]
        for rel, n in sorted(self.added_by_relation.items()):
            lines.append(f"  - {rel:<14}: {n}")
        if self.concepts_created:
            lines.append(f"concepts created   : {self.concepts_created}")
        return "\n".join(lines)

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.summary()


class RDFAugmenter:
    """Augment an RDF graph with backend-proposed related terms.

    Parameters
    ----------
    backend:
        Any :class:`~rdf_augmenter.backends.Backend`. Defaults to the offline
        :class:`~rdf_augmenter.backends.LexiconBackend`.
    seed:
        Seed for the random generator that drives ``ratio`` sampling, so runs
        are reproducible.
    namespace:
        Namespace used for minted predicates/resources (defaults to :data:`AUG`).
    """

    def __init__(
        self,
        backend: Optional[Backend] = None,
        seed: int = 42,
        namespace: Namespace = AUG,
    ):
        self.backend = backend or LexiconBackend()
        self.seed = seed
        self.ns = namespace
        self.graph = Graph()
        self._bind_namespaces()

    # ------------------------------------------------------------------ I/O
    def _bind_namespaces(self) -> None:
        self.graph.bind("aug", self.ns)
        self.graph.bind("skos", SKOS)
        self.graph.bind("prov", PROV)

    def load(
        self,
        source: Optional[str] = None,
        data: Optional[str] = None,
        fmt: str = "turtle",
    ) -> Graph:
        """Load RDF into the augmenter from a file path or an inline string."""
        if (source is None) == (data is None):
            raise ValueError("Provide exactly one of `source` or `data`.")
        if source is not None:
            self.graph.parse(source, format=fmt)
        else:
            self.graph.parse(data=data, format=fmt)
        self._bind_namespaces()
        return self.graph

    def export(self, path: Optional[str] = None, fmt: str = "turtle") -> str:
        """Serialize the (augmented) graph. Returns the string; writes if ``path``."""
        rdflib_fmt = EXPORT_FORMATS.get(fmt.lower())
        if rdflib_fmt is None:
            raise ValueError(
                f"Unsupported format '{fmt}'. Choose one of: "
                f"{', '.join(sorted(set(EXPORT_FORMATS)))}."
            )
        text = self.graph.serialize(format=rdflib_fmt)
        if path is not None:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
        return text

    # ------------------------------------------------------------ augment
    def augment(
        self,
        predicates: Optional[Sequence[Union[str, URIRef]]] = None,
        relations: Sequence[str] = RELATION_TYPES,
        top_k: int = 4,
        ratio: float = 1.0,
        attach: str = "subject",
        predicate_map: Optional[Dict[str, URIRef]] = None,
        provenance: bool = True,
        full_provenance: bool = False,
    ) -> AugmentationReport:
        """Augment the loaded graph in place and return an :class:`AugmentationReport`.

        Parameters
        ----------
        predicates:
            Only literals reached through these predicates are augmented. If
            ``None`` *every* predicate with literal objects is eligible.
        relations:
            Which relation types to generate (subset of :data:`RELATION_TYPES`).
        top_k:
            Maximum suggestions requested per (term, relation).
        ratio:
            Fraction (0..1) of eligible literals to augment, sampled with the
            seeded RNG. ``1.0`` augments all of them.
        attach:
            ``"subject"`` adds triples to the original subject (faithful to the
            published method). ``"concept"`` instead builds a SKOS thesaurus:
            each distinct literal becomes a ``skos:Concept`` linked to the
            subject via ``aug:concept`` and connected to its synonyms
            (``skos:altLabel``), hypernyms (``skos:broader``) and related
            terms (``skos:related``).
        predicate_map:
            Override the predicate used per relation type in ``"subject"`` mode.
        provenance:
            Record a PROV-O activity describing this run in the graph.
        full_provenance:
            Additionally reify every generated triple and link it to the run
            activity (verbose, but fully traceable per triple).
        """
        relations = tuple(r for r in relations if r in RELATION_TYPES)
        if not relations:
            raise ValueError(f"`relations` must be a subset of {RELATION_TYPES}.")
        if not 0.0 <= ratio <= 1.0:
            raise ValueError("`ratio` must be between 0 and 1.")
        if attach not in ("subject", "concept"):
            raise ValueError("`attach` must be 'subject' or 'concept'.")

        pmap = dict(DEFAULT_PREDICATE_MAP)
        if predicate_map:
            pmap.update(predicate_map)

        target_preds = None
        if predicates is not None:
            target_preds = {URIRef(str(p)) for p in predicates}

        rng = random.Random(self.seed)
        before = stats(self.graph)
        input_sha = _sha256(self.graph.serialize(format="nt"))

        # Collect eligible (subject, predicate, literal) triples deterministically.
        eligible = [
            (s, p, o)
            for s, p, o in sorted(self.graph, key=lambda t: (str(t[0]), str(t[1]), str(t[2])))
            if isinstance(o, Literal)
            and (target_preds is None or p in target_preds)
        ]
        if ratio < 1.0 and eligible:
            k = max(1, int(round(len(eligible) * ratio)))
            eligible = sorted(rng.sample(eligible, k), key=lambda t: (str(t[0]), str(t[1]), str(t[2])))

        # Provenance activity node.
        activity = self.ns[f"run-{self.seed}"]
        if provenance:
            self._record_activity(activity, relations, top_k, ratio, attach)

        added = 0
        by_rel: Dict[str, int] = {r: 0 for r in relations}
        concept_index: Dict[str, URIRef] = {}
        minted_concepts: set = set()
        new_triples: List = []

        def declare_concept(label: str) -> URIRef:
            """Ensure ``label`` has a well-formed skos:Concept node and return it."""
            uri = self._concept_uri(label)
            if uri not in minted_concepts:
                minted_concepts.add(uri)
                for decl in (
                    (uri, RDF.type, SKOS.Concept),
                    (uri, SKOS.prefLabel, Literal(label)),
                ):
                    if decl not in self.graph and decl not in new_triples:
                        new_triples.append(decl)
            return uri

        for s, p, o in eligible:
            term = str(o)
            if attach == "concept":
                concept = declare_concept(term)
                concept_index[term.lower()] = concept
                link = (s, self.ns.concept, concept)
                if link not in self.graph and link not in new_triples:
                    new_triples.append(link)

            for rel in relations:
                for cand in self.backend.suggest(term, rel, top_k=top_k):
                    # In concept mode, broader/related point to other concepts,
                    # which must themselves be declared for a valid SKOS graph.
                    if attach == "concept" and rel in ("hypernym", "related"):
                        declare_concept(cand)
                    triple = self._build_triple(s, p, term, rel, cand, pmap, attach, concept_index)
                    if triple is None:
                        continue
                    if triple in self.graph or triple in new_triples:
                        continue
                    new_triples.append(triple)
                    by_rel[rel] += 1
                    added += 1
                    if full_provenance and provenance:
                        self._reify(triple, activity, rel, term)

        for t in new_triples:
            self.graph.add(t)

        if provenance:
            self.graph.add(
                (activity, self.ns.triplesAdded, Literal(added, datatype=XSD.integer))
            )

        after = stats(self.graph)
        return AugmentationReport(
            backend=self.backend.name,
            seed=self.seed,
            params={
                "predicates": [str(p) for p in (target_preds or [])] or "ALL",
                "relations": list(relations),
                "top_k": top_k,
                "ratio": ratio,
                "attach": attach,
                "provenance": provenance,
                "full_provenance": full_provenance,
                "backend_detail": self.backend.describe(),
            },
            triples_before=before["triples"],
            triples_after=after["triples"],
            added=added,
            added_by_relation=by_rel,
            concepts_created=len(minted_concepts),
            input_sha256=input_sha,
            stats_before=before,
            stats_after=after,
            timestamp=_dt.datetime.now(_dt.timezone.utc).isoformat(),
        )

    # --------------------------------------------------------- internals
    def _build_triple(self, s, p, term, rel, cand, pmap, attach, concept_index):
        if attach == "subject":
            if rel == "synonym":
                # Stay faithful to the source predicate for alternative values.
                pred = p if pmap.get("synonym") is None else pmap["synonym"]
            else:
                pred = pmap[rel]
            return (s, pred, Literal(cand))
        # attach == "concept": build the thesaurus around the literal's concept.
        concept = concept_index[term.lower()]
        cand_concept = self._concept_uri(cand)
        if rel == "synonym":
            return (concept, SKOS.altLabel, Literal(cand))
        if rel == "hypernym":
            return (concept, SKOS.broader, cand_concept)
        return (concept, SKOS.related, cand_concept)

    def _concept_uri(self, label: str) -> URIRef:
        slug = "".join(
            ch if ch.isalnum() else "-" for ch in label.strip().lower()
        ).strip("-")
        return self.ns[f"concept/{slug or 'unknown'}"]

    def _record_activity(self, activity, relations, top_k, ratio, attach) -> None:
        agent = self.ns["agent/rdf-augmenter"]
        self.graph.add((agent, RDF.type, PROV.SoftwareAgent))
        self.graph.add((agent, RDFS.label, Literal(f"rdf-augmenter {_version()}")))
        self.graph.add((activity, RDF.type, PROV.Activity))
        self.graph.add((activity, PROV.wasAssociatedWith, agent))
        self.graph.add(
            (activity, PROV.startedAtTime,
             Literal(_dt.datetime.now(_dt.timezone.utc).isoformat(), datatype=XSD.dateTime))
        )
        self.graph.add((activity, self.ns.backend, Literal(self.backend.name)))
        self.graph.add((activity, self.ns.seed, Literal(self.seed, datatype=XSD.integer)))
        self.graph.add((activity, self.ns.topK, Literal(top_k, datatype=XSD.integer)))
        self.graph.add((activity, self.ns.ratio, Literal(ratio, datatype=XSD.decimal)))
        self.graph.add((activity, self.ns.attach, Literal(attach)))

    def _reify(self, triple, activity, rel, source_term) -> None:
        s, p, o = triple
        stmt = self.ns[
            "stmt/" + _sha256(f"{s}|{p}|{o}".encode())[:16]
        ]
        self.graph.add((stmt, RDF.type, RDF.Statement))
        self.graph.add((stmt, RDF.subject, s))
        self.graph.add((stmt, RDF.predicate, p))
        self.graph.add((stmt, RDF.object, o))
        self.graph.add((stmt, PROV.wasGeneratedBy, activity))
        self.graph.add((stmt, self.ns.relationType, Literal(rel)))
        self.graph.add((stmt, self.ns.sourceTerm, Literal(source_term)))


# --------------------------------------------------------------- helpers
def stats(graph: Graph) -> Dict:
    """Compute summary statistics for an RDF graph."""
    subjects, predicates, objects, literals = set(), set(), set(), 0
    pred_counts: Dict[str, int] = {}
    for s, p, o in graph:
        subjects.add(s)
        predicates.add(p)
        objects.add(o)
        pred_counts[str(p)] = pred_counts.get(str(p), 0) + 1
        if isinstance(o, Literal):
            literals += 1
    n_triples = len(graph)
    n_nodes = len(subjects | objects)
    return {
        "triples": n_triples,
        "subjects": len(subjects),
        "predicates": len(predicates),
        "objects": len(objects),
        "literals": literals,
        "nodes": n_nodes,
        "density": round(n_triples / n_nodes, 4) if n_nodes else 0.0,
        "predicate_counts": dict(sorted(pred_counts.items())),
    }


def _sha256(data) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _version() -> str:
    try:
        from . import __version__

        return __version__
    except Exception:  # pragma: no cover
        return "unknown"


def _rdflib_version() -> str:
    try:
        import rdflib

        return rdflib.__version__
    except Exception:  # pragma: no cover
        return "unknown"
