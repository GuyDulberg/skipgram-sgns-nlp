# skipgram-sgns-nlp

Implementation of **Skip-Gram with Negative Sampling (SGNS)** for learning word embeddings in Python.  

## Overview
The goal of this project is to build a full algorithmic pipeline for learning distributional semantics:
- Build a hands-on implementation of the SGNS algorithm from scratch in Python.
- Train word embeddings using Skip-Gram with Negative Sampling.
- Understand the training process: logistic regression, loss functions, gradient descent, backpropagation.
- Explore semantic similarity and analogy tasks using the learned embeddings.

## Key Features
- **Text preprocessing**: normalization and tokenization of raw text.
- **Skip-Gram training pipeline**:
  - Generate (target, context) pairs.
  - Apply negative sampling.
  - Update embeddings using stochastic gradient descent.
  - Support for early stopping and saving intermediate models.
- **Evaluation utilities**:
  - Compute similarity between words.
  - Retrieve closest words in the embedding space.
  - Solve analogy tasks (e.g., *king - man + woman ≈ queen*).

## Implemented API
- `normalize_text(fn)`
- `load_model(fn)`
- `SkipGram` class with:
  - `compute_similarity(w1, w2)`
  - `get_closest_words(w, n)`
  - `learn_embeddings(step_size, epochs, early_stopping, model_path)`
  - `combine_vectors(T, C, combo, model_path)`
  - `find_analogy(w1, w2, w3)`
  - `test_analogy(w1, w2, w3, w4, n)`

## Example Usage
```python
from src.ex2 import normalize_text, SkipGram

# Load and normalize corpus
sentences = normalize_text("data/drSeuss.txt")

# Initialize model
sg = SkipGram(sentences, d=100, neg_samples=4, context=4, word_count_threshold=5)

# Train embeddings
sg.learn_embeddings(step_size=1e-4, epochs=5, early_stopping=2, model_path="models/sgns.pkl")

# Query examples
print("Similarity (cat, dog):", sg.compute_similarity("cat", "dog"))
print("Closest to 'cat':", sg.get_closest_words("cat", 5))
print("Analogy king - man + woman ≈", sg.find_analogy("king", "man", "woman"))

Similarity (cat, dog): 0.64
Closest to 'cat': ['dog', 'mouse', 'pet', 'kitten', 'animal']
Analogy king - man + woman ≈ queen
