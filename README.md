# N-Gram-Language-Model-and-Text-Generator

This project implements a classic N-gram language model from scratch in Python. The model is trained on the NLTK Gutenberg corpus and can generate new text based on a given prefix. 

## Table of Contents
- Features
- Project Structure
- How It Works
- Setup and Installation
- Usage
- Model Evaluation
- Assignment Requirements

## Features

- **N-Gram Model:** Implements a standard N-gram model (configurable n, defaults to bigrams, n=2).
- **Data Preprocessing:** Collects, cleans (lowercase, removes special characters), and tokenizes text from the NLTK Gutenberg corpus.
- **Probabilistic Generation:** Generates text by predicting the next word based on the probability distribution of the preceding n-1 words.
- **Laplace Smoothing:** Implements Laplace (add-one) smoothing to handle unseen n-grams and avoid zero-probability issues.
- **Context Backoff:** If an (n-1)-gram context has not been seen, the model backs off to a shorter context (n-2, etc.) to make a prediction.
- **Perplexity Evaluation:** Includes a function to calculate the perplexity of the model on a test set, a standard metric for evaluating language models.
- **Interactive Web UI:** A user-friendly interface built with Streamlit that allows users to:
  - Enter a custom text prefix.
  - Adjust the desired output length.
  - Generate multiple text variations with a single click.

