# Character-Level Bigram Language Model

A character-level bigram language model built from scratch, following Andrej Karpathy's *makemore* series. This project trains on a dataset of names and generates new, name-like strings one character at a time.

## What It Does

* **Data Processing:** Reads a dataset of names and builds character-level bigram statistics.
* **Dual Modeling Approach:** Models `P(next_char | current_char)` using two distinct methods:
  * **Counting:** Constructs a 27×27 frequency matrix, normalized into probabilities.
  * **Neural Network:** Uses a single linear layer trained with gradient descent, converging toward the same distribution.
* **Text Generation:** Samples from the model to generate new names.
* **Evaluation:** Scores both models with average negative log-likelihood (NLL) over every bigram in the dataset.

## Key Concepts Explored

* **Bigram / n-gram language modeling**
* **Turning counts into a probability distribution** (normalization, smoothing)
* **One-hot encoding and a single-layer neural net** as an equivalent formulation
* **Negative log-likelihood** as a loss function
* **Sampling** from a multinomial distribution

## Results

Dataset: 32,033 names, 228,146 bigrams.

| Model | Average NLL |
|-------|-------------|
| Counting (add-one smoothing) | 2.4546 |
| Single linear layer, 300 steps of gradient descent (lr 50, small L2 penalty) | about 2.46 |

The neural net closes in on the counting model as training continues, which is expected because they are two parameterizations of the same bigram distribution. Both numbers are measured on the data the models were fit on, so they compare fit, not generalization.

## Run it

```bash
pip install torch matplotlib jupyter
jupyter notebook makemore.ipynb   # run from the repo root so names.txt is found
```

In Google Colab, upload `names.txt` next to the notebook first.
