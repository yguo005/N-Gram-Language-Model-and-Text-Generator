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
## Code Structure

### Files

- **ngram.py**  
  Contains all the core functions for building and evaluating the language model. This includes:
  - Data loading and preprocessing
  - Tokenization
  - N-gram frequency calculation
  - Model building with Laplace smoothing
  - Sentence/text generation
  - Perplexity computation

- **app.py**  
  Provides the front-end for the model. It imports functions from `ngram.py`, creates a user interface with Streamlit, and allows users to generate text interactively in their browser.

---

## How It Works

### 1. Data Collection and Preprocessing

- **Corpus:** Uses the Gutenberg corpus from the `nltk` library.
- **Preprocessing steps:**
  - Convert all text to lowercase.
  - Remove all non-alphanumeric characters (except spaces).
  - Tokenize the cleaned text into a flat list of words.
- **Vocabulary:**  
  Build a vocabulary limited to the 2,000 most common words for efficiency. Words not in the vocabulary are replaced with `<UNK>` (unknown).

---

### 2. N-Gram Model Implementation

- **Frequencies:**  
  The model creates n-grams (sequences of n words) from the tokenized text and uses `nltk.FreqDist` to count the frequency of each unique n-gram.

- **Transition Probabilities:**  
  A transition model is built as a dictionary:
  - Keys: contexts (tuples of n-1 words)
  - Values: dictionaries of possible next words and their probabilities

- **Laplace Smoothing:**  
  To handle words that never appeared after a given context, Laplace (add-one) smoothing is applied:
  ```
  P(w|c) = (count(c, w) + 1) / (count(c) + V)
  ```
  where:
  - `count(c, w)` = count of word `w` following context `c`
  - `count(c)` = count of context `c`
  - `V` = vocabulary size

---

### 3. Text Generation

The `generate_sentence` function creates text by:
1. Taking an initial prefix (list of words).
2. Using the last n-1 words as the current context.
3. Looking up the context in the transition model to get the probability distribution of next words.
4. Backing off to a shorter context if the current one is not found.
5. Choosing the next word probabilistically using `random.choices` based on the weights (probabilities).
6. Appending the new word to the sentence and repeating until the desired length is reached.
