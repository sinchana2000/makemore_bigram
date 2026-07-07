# Character-Level Bigram Language Model

A character-level bigram language model built from scratch, following Andrej Karpathy's *makemore* series. This project trains on a dataset of names and generates new, name-like strings one character at a time.

## What It Does

* **Data Processing:** Reads a dataset of names and builds character-level bigram statistics.
* **Dual Modeling Approach:** Models `P(next_char | current_char)` using two distinct methods:
  * **Counting:** Constructs a 27×27 frequency matrix, normalized into probabilities.
  * **Neural Network:** Uses a single linear layer trained with gradient descent, reproducing the same distribution.
* **Text Generation:** Samples from the model to generate new names.
* **Evaluation:** Evaluates quality using average negative log-likelihood (loss).

## Key Concepts Explored

* **Bigram / n-gram language modeling**
* **Turning counts into a probability distribution** (normalization, smoothing)
* **One-hot encoding and a single-layer neural net** as an equivalent formulation
* **Negative log-likelihood** as a loss function
* **Sampling** from a multinomial distribution
