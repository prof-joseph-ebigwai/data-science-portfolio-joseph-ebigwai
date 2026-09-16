# Medical Question Answering with Retrieval-Augmented Generation (RAG)

## Project Overview

This project explores the use of Retrieval-Augmented Generation (RAG) for medical question answering.

The objective was to improve the factual reliability of large language model responses by retrieving relevant information from an external medical knowledge source before generating an answer.

## Project Workflow

The project involved:

- Loading medical reference documents
- Splitting text into overlapping chunks
- Generating semantic embeddings
- Storing embeddings in a vector database
- Retrieving relevant passages for user questions
- Prompt engineering
- Generating responses with a large language model
- Evaluating response groundedness and relevance
- Investigating retrieval and context-window limitations

## Technologies Used

- Python
- LangChain
- Hugging Face
- Mistral-7B
- llama-cpp-python
- Sentence Transformers
- all-MiniLM-L6-v2 embeddings
- Chroma vector database
- PyMuPDF
- Retrieval-Augmented Generation

## Retrieval Pipeline

The medical reference material was loaded and divided into overlapping chunks.

A token-aware text splitter was used with:

- Chunk size: 400
- Chunk overlap: 50

Semantic embeddings were generated using the `all-MiniLM-L6-v2` sentence-transformer model and stored in a Chroma vector database.

Relevant document chunks were then retrieved through semantic similarity search before answer generation.

## Language Model

The project used a Mistral-7B instruction-tuned language model.

The model was first evaluated as a standalone language model and was then combined with external medical knowledge through the RAG pipeline.

This comparison demonstrated the value of grounding generated answers in retrieved evidence.

## Evaluation

The generated responses were evaluated using:

- Groundedness
- Relevance

Groundedness measured whether the generated response was supported by the retrieved context.

Relevance measured whether the generated response appropriately addressed the user's question.

## Key Learning

The project demonstrated that Retrieval-Augmented Generation can improve the reliability of AI-generated medical information by connecting language models to authoritative external knowledge.

It also showed that retrieval quality depends on factors such as:

- chunk size
- chunk overlap
- retrieval depth
- prompt design
- context-window limitations

## Responsible Use

The system is intended as an information-support tool and not as a replacement for qualified medical professionals.

AI-generated medical responses should remain subject to appropriate human oversight.

## Skills Demonstrated

- Python
- Natural Language Processing
- Large Language Models
- Retrieval-Augmented Generation
- Semantic Search
- Embeddings
- Vector Databases
- Prompt Engineering
- RAG Evaluation
- Groundedness and Relevance Assessment

## Author

**Joseph Kayefor Ebigwai**

Data Science Portfolio Project
