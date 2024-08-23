import logging
from rdflib import Graph, Literal, URIRef
from transformers import pipeline, AutoModelForMaskedLM, AutoTokenizer, BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from nltk import pos_tag, word_tokenize

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure the necessary resources are downloaded
nltk.download('averaged_perceptron_tagger')

# Sample RDF data
rdf_data = """
@prefix ex: <http://example.org/> .

ex:Person1 ex:name "John" ;
          ex:occupation "Data Scientist" .

ex:Person2 ex:name "Alice" ;
          ex:occupation "Surgeon" .

ex:Person3 ex:name "Michael" ;
          ex:occupation "Architect" .

ex:Person4 ex:name "Sophie" ;
          ex:occupation "Journalist" .

ex:Person5 ex:name "Emma" ;
          ex:occupation "Lawyer" .
"""

# Load RDF graph
g = Graph()
g.parse(data=rdf_data, format="turtle")

# Load BERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)
bert_model = BertModel.from_pretrained(model_name)
bert_tokenizer = BertTokenizer.from_pretrained(model_name)
nlp = pipeline("fill-mask", model=model, tokenizer=tokenizer)

def get_contextual_related_terms(term, context, relation_type='type', top_k=5):
    if relation_type == 'type':
        masked_text = f"{term} is a type of [MASK] in a professional context."
    elif relation_type == 'related':
        masked_text = f"{term} is closely related to [MASK] in the field of {term.lower()}."
    
    result = nlp(masked_text, top_k=top_k)
    related_terms = [r['token_str'].strip().lower() for r in result if r['token_str'].strip().lower() != term.lower()]
    return related_terms

def get_pos_tag(word):
    pos = pos_tag(word_tokenize(word))
    return pos[0][1]

def valid_triple(s, p, o, original_word):
    original_pos = get_pos_tag(original_word)
    new_word_pos = get_pos_tag(str(o))
    if original_pos.startswith('N') and new_word_pos.startswith('N'):
        return True
    if original_pos.startswith('V') and new_word_pos.startswith('V'):
        return True
    if original_pos.startswith('J') and new_word_pos.startswith('J'):
        return True
    if original_pos.startswith('R') and new_word_pos.startswith('R'):
        return True
    return False

def filter_terms(term, candidates):
    term_embedding = get_embedding(term)
    filtered_terms = []
    for candidate in candidates:
        candidate_embedding = get_embedding(candidate)
        similarity = cosine_similarity(term_embedding, candidate_embedding)
        if similarity > 0.85 and not is_irrelevant(candidate):
            filtered_terms.append(candidate)
    return filtered_terms

def is_irrelevant(term):
    irrelevant_terms = {'animal', 'horse', 'locomotive', 'game', 'robot', 'noun', 'productivity', 'position', 'person'}
    return term in irrelevant_terms

def get_embedding(text):
    inputs = bert_tokenizer(text, return_tensors="pt")
    outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

def should_augment(predicate):
    return predicate == URIRef("http://example.org/occupation")

# New predicates for hypernyms (types) and speculative related terms
type_of_predicate = URIRef("http://example.org/isTypeOf")
speculative_predicate = URIRef("http://example.org/possiblyRelatedTo")

# Iterate through literals in the RDF graph selectively
for s, p, o in g:
    if isinstance(o, Literal) and should_augment(p):
        term = str(o)
        context = f"{s} {p} {o}"
        logger.info(f"Generating context-aware related terms for term '{term}' with context '{context}'")
        
        # Generate context-aware synonyms (using the original method)
        synonyms = get_contextual_related_terms(term, context, relation_type='type', top_k=4)
        
        # Get hypernyms (types) and speculative related terms
        hypernyms = get_contextual_related_terms(term, context, relation_type='type', top_k=4)
        speculative_terms = get_contextual_related_terms(term, context, relation_type='related', top_k=4)
        
        # Combine all candidates and filter them
        candidates = synonyms + hypernyms + speculative_terms
        filtered_terms = filter_terms(term, candidates)
        
        # Deduplicate and add terms
        added_terms = set()
        
        # Add synonyms using the same predicate
        for synonym in filtered_terms:
            if synonym not in added_terms and valid_triple(s, p, Literal(synonym), term):
                logger.info(f"Adding synonym '{synonym}' for term '{term}'")
                g.add((s, p, Literal(synonym)))
                added_terms.add(synonym)

        # Add hypernyms using the type_of predicate
        for hypernym in hypernyms:
            if hypernym not in added_terms and valid_triple(s, type_of_predicate, Literal(hypernym), term):
                logger.info(f"Adding hypernym '{hypernym}' for term '{term}'")
                g.add((s, type_of_predicate, Literal(hypernym)))
                added_terms.add(hypernym)

        # Add speculative related terms using the speculative predicate
        for speculative_term in speculative_terms:
            if speculative_term not in added_terms and valid_triple(s, speculative_predicate, Literal(speculative_term), term):
                logger.info(f"Adding speculative related term '{speculative_term}' for term '{term}'")
                g.add((s, speculative_predicate, Literal(speculative_term)))
                added_terms.add(speculative_term)

# Serialize and print the augmented graph
output_data = g.serialize(format="turtle")
print(output_data)
