# RDF Graph Augmentation with BERT-based Contextual Embeddings

This project enhances an RDF (Resource Description Framework) graph by generating and adding context-aware related terms, hypernyms, and speculative related terms using BERT-based models. The augmentation process is driven by natural language processing techniques to enrich the graph with semantically relevant information.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [How it Works](#how-it-works)
- [Logging](#logging)
- [Dependencies](#dependencies)
- [Related Works](#related-works)
- [License](#license)

## Introduction

The primary goal of this project is to augment RDF graphs by leveraging the capabilities of pre-trained BERT models. The program parses a given RDF graph, identifies specific literals (like occupations), and enriches them with contextually relevant synonyms, hypernyms (types), and speculative related terms. This can be particularly useful in knowledge graph construction, ontology building, and other applications that require enhanced semantic relationships.

## Features

- **Contextual Term Generation**: Uses BERT to generate related terms based on the context of the existing triples in the RDF graph.
- **Hypernym and Speculative Term Augmentation**: Adds hypernyms and speculative related terms to the graph, enriching its semantic depth.
- **POS Tagging for Validation**: Ensures that only contextually valid triples are added by comparing the part of speech (POS) tags of terms.
- **Customizable Augmentation**: Allows the customization of the predicates and relation types to be augmented.

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/rdf-augmentation.git
   cd rdf-augmentation
   ```

2. **Create and Activate a Virtual Environment (Optional but Recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Required Packages:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Prepare RDF Data**: Modify the `rdf_data` string in the code to include your RDF triples in Turtle format.

2. **Run the Script**: Execute the main script to process and augment your RDF graph:
   ```bash
   python rdf_augmenter.py
   ```

3. **View the Output**: The augmented RDF graph will be serialized and printed in Turtle format.

## How it Works

The script follows these steps:

1. **RDF Parsing**: The input RDF graph is parsed using `rdflib`.
2. **BERT Model Initialization**: The BERT model and tokenizer are loaded for generating context-aware embeddings.
3. **Contextual Term Generation**: For each relevant literal in the RDF graph, the script generates context-aware synonyms, hypernyms, and speculative related terms.
4. **POS Validation**: Ensures that newly generated terms are contextually appropriate based on their POS tags.
5. **Graph Augmentation**: Adds the valid terms to the RDF graph using predefined predicates.
6. **Serialization**: Outputs the augmented RDF graph in Turtle format.

## Logging

The script includes logging to track the augmentation process:
- Logs are outputted to the console, detailing the terms generated and added to the RDF graph.

## Dependencies

- Python 3.7+
- `rdflib`
- `transformers`
- `torch`
- `sklearn`
- `nltk`

To install all dependencies, use the following command:
```bash
pip install -r requirements.txt
```

## Related Works

```
@article{martinez2022kgaugmentation,
  author       = {Jorge Martinez-Gil and
                  Shaoyi Yin and
                  Josef K{\"{u}}ng and
                  Franck Morvan},
  title        = {Knowledge Graph Augmentation for Increased Question Answering Accuracy},
  journal      = {Trans. Large Scale Data Knowl. Centered Syst.},
  volume       = {52},
  pages        = {70--85},
  year         = {2022},
  url          = {https://doi.org/10.1007/978-3-662-66146-8\_3},
  doi          = {10.1007/978-3-662-66146-8\_3}
}
```

## License

This project is licensed under the MIT License.