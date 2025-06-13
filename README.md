# N-Gram-Language-Model-and-Text-Generator
This project implements a classic N-gram language model from scratch in Python. The model is trained on the NLTK Gutenberg corpus and can generate new text based on a given prefix. It also includes a web-based user interface built with Streamlit for interactive text generation.

This project was completed as part of the "DS 5983: Large Language Models" course (PA1).

Table of Contents
Features
Project Structure
How It Works
Setup and Installation
Usage
Model Evaluation
Assignment Requirements
Features
N-Gram Model: Implements a standard N-gram model (configurable n, defaults to bigrams, n=2).
Data Preprocessing: Collects, cleans (lowercase, removes special characters), and tokenizes text from the NLTK Gutenberg corpus.
Probabilistic Generation: Generates text by predicting the next word based on the probability distribution of the preceding n-1 words.
Laplace Smoothing: Implements Laplace (add-one) smoothing to handle unseen n-grams and avoid zero-probability issues.
Context Backoff: If an (n-1)-gram context has not been seen, the model backs off to a shorter context (n-2, etc.) to make a prediction.
Perplexity Evaluation: Includes a function to calculate the perplexity of the model on a test set, a standard metric for evaluating language models.
Interactive Web UI: A user-friendly interface built with Streamlit that allows users to:
Enter a custom text prefix.
Adjust the desired output length.
Generate multiple text variations with a single click.
Project Structure
.
├── ngram.py            # Core logic for the N-gram model, data processing, and perplexity evaluation.
├── app.py              # Streamlit web application for interactive text generation.
├── requirements.txt    # Python dependencies for the project.
└── README.md           # This file.
Use code with caution.
ngram.py: Contains all the core functions for building and evaluating the language model. This includes data loading, tokenization, n-gram frequency calculation, model building with Laplace smoothing, sentence generation, and perplexity computation.
app.py: Provides the front-end for the model. It imports functions from ngram.py, creates a user interface with Streamlit, and allows users to generate text interactively in their browser.
How It Works
1. Data Collection and Preprocessing
The model uses the gutenberg corpus from the nltk library. The text is preprocessed by:

Converting all text to lowercase.
Removing all non-alphanumeric characters (except spaces).
Tokenizing the cleaned text into a flat list of words.
Building a vocabulary limited to the 2,000 most common words to improve efficiency. Words not in the vocabulary are treated as <UNK> (unknown).
2. N-Gram Model Implementation
Frequencies: The model creates n-grams (sequences of n words) from the tokenized text and uses nltk.FreqDist to count the frequency of each unique n-gram.
Transition Probabilities: A transition model is built as a dictionary where keys are contexts (tuples of n-1 words) and values are dictionaries of possible next words and their probabilities.
Laplace Smoothing: To handle words that never appeared after a given context, Laplace (add-one) smoothing is applied. The probability of a word w following a context c is calculated as:
P(w|c) = (count(c, w) + 1) / (count(c) + V)
where V is the size of the vocabulary.
3. Text Generation
The generate_sentence function creates text by:

Taking an initial prefix (list of words).
Using the last n-1 words as the current context.
Looking up the context in the transition model to find the probability distribution of the next words.
Backing off to a shorter context if the current one is not found.
Choosing the next word probabilistically using random.choices based on the weights (probabilities).
Appending the new word to the sentence and repeating the process until the desired length is reached.
