## Introduction

This project serves as Artifact 1 in the "Concepts of Formal AI" (CFAI) series, developed specifically for use in educational materials and presentations. It explores the foundational premise that every artificial intelligence model is, ultimately, equivalent to a computer program. Grounded in the Church-Turing thesis, we empirically observe this truth whenever a neural network executes on deterministic hardware. This artifact demonstrates strict observational equivalence between a deterministic Python script and a Multilayer Perceptron (MLP), both replicating Run-Length Encoding (RLE) logic with 100% accuracy within a closed universe.

We can analyze this relationship through several theoretical lenses:

* **Observational Equivalence:** Both systems yield the exact same outputs for a given set of inputs, functioning as black-box equivalents. This project successfully achieves this.
* **Weak Equivalence:** The systems achieve the same result but arrive there using entirely different internal processes (e.g., sequential algorithmic loops versus matrix multiplication).
* **Strong Isomorphism:** A theoretical state where the internal architectures map perfectly onto one another. Because of the fundamental differences between state machines and feedforward networks, this project exhibits weak equivalence rather than strong isomorphism.
* **Information Loss:** Analyzing what operational details are abstracted away or lost when transitioning from a strict, step-by-step algorithm to a continuous weight matrix.

While these lenses provide analytical depth, the primary objective is to illustrate the core thesis of Formal AI. Rather than treating machine learning as an opaque statistical phenomenon, Formal AI acknowledges its programmatic reality and flips the paradigm: we should deliberately construct AI systems as human-readable computer programs and data from the start.

## 1. Development Iterations

The development process was not strictly planned in advance but emerged through trial-and-error, experimentation, and human collaboration with Google Gemini. The project progressed through three major iterations to adapt statistical function approximation to rigid logical rules:

* **Iteration 1: Continuous Value Regression.** Initial regression attempts to predict characters and counts failed to converge reliably due to character-weight bias.
* **Iteration 2: Classification Transition.** Redefining the problem as multi-class classification using one-hot tokenization and cross-entropy loss raised accuracy to 99.97%.
* **Iteration 3: Closed Universe Balancing.** Training on a fixed, equally distributed corpus of all 256 possible permutations of an 8-element binary sequence eliminated statistical drift, locking in 100% accuracy.

## 2. Model Architecture

The MLP bypasses sequential time steps, mapping the RLE task spatially via a fixed input matrix.

### Feature Encoding

* **Input:** An 8-character string maps to one-hot vectors (`A = [1.0, 0.0]`, `B = [0.0, 1.0]`), flattened into a 16-element vector.
* **Output:** The network outputs an 88-element vector, reshaped to 8 sequence slots. Each slot is an 11-way classification: Class 0 (Padding), Class 1 (`A`), Class 2 (`B`), and Classes 3-10 (Run counts 1 through 8).

### Network Structure

* Layer 1: `nn.Linear(16, 256)` -> `nn.ReLU()`
* Layer 2: `nn.Linear(256, 256)` -> `nn.ReLU()`
* Layer 3: `nn.Linear(256, 128)` -> `nn.ReLU()`
* Layer 4: `nn.Linear(128, 88)` -> Reshaped to `(-1, 8, 11)`

## 3. Workspace Structure

| File | Description |
| --- | --- |
| `corpus.py` | Generates the balanced 25,600-row text dataset. |
| `rle_ann.py` | Translation bridge for loading weights and running inference. |
| `rle_corpus.csv` | Compressed dataset containing all permutations. |
| `rle_model.pt` | Saved weights tensor for the classification network. |
| `rle_py.py` | Reference deterministic execution script. |
| `test_demo.py` | Evaluation script with CLI corpus-loading flags. |

## 4. Setup & Execution

Requires Python 3 and PyTorch. Hardware acceleration is automatically detected.

Install dependencies:

```bash
  pip install torch

```

Build the corpus and train the model:

```bash
  python corpus.py
  python train_model.py

```

## 5. Test Driver Command Line Options

Use `test_demo.py` to customize the testing scenario.

| Flag | Description | Default |
| --- | --- | --- |
| `-m`, `--module` | Framework to test: `rle_py`, `rle_ann`, or `both`. | `both` |
| `-r`, `--runs` | Integer count of test sequences to evaluate. | 1000 |
| `-c`, `--corpus` | File path to a saved CSV to test static sequences. | `None` |
| `-v`, `--verbose` | Enables step-by-step terminal reporting. | `False` |

### Verification Examples

Run the complete 25,600-line corpus through both engines:

```bash
  python test_demo.py -m both -c rle_corpus.csv -r 25600

```

Run 20 lines from the corpus with the step-by-step visualizer:

```bash
  python test_demo.py -m both -c rle_corpus.csv -r 20 -v

```

Test model stability against 5,000 randomly generated sequences:

```bash
  python test_demo.py -m both -r 5000

```

## 6. Evaluation and Results

The primary metric for success was achieving total coverage and absolute parity within the bounded environment. By evaluating the models against the complete closed universe of all 256 possible sequence permutations, the project secured a 100.00% accuracy rate. Both the deterministic Python script and the Multilayer Perceptron yield identical results for every possible input configuration. Testing confirmed that no statistical drift or information loss occurs across the defined parameters, fully satisfying the requirement for strict observational equivalence.

## 7. Limitations and Scope Acceptance

The objective of this artifact is not to demonstrate strong isomorphism (where internal architectures map perfectly onto one another) but to achieve strict observational equivalence. From the perspective of inputs and outputs, both systems must remain indistinguishable.

Achieving this required intentional design choices. True recurrent algorithms rely on continuous probabilistic calculations, which inevitably suffer from statistical drift over long sequences. This project bypasses those vulnerabilities by using a feedforward network on a bounded universe, effectively structuring the neural network as a truth-table lookup system.

In computer science, algorithms and lookup tables are dual representations of the same underlying logic, illustrating the fundamental space-time tradeoff. Any deterministic function operating on a finite, bounded domain can be expressed as a lookup table. The MLP utilizes spatial memory (network weights) to map inputs directly to outputs, trading computational time for parameter space to guarantee absolute mathematical stability. Conversely, the Python script computes the answer sequentially over time. While the Python program could easily have been written as a 256-entry lookup table to mirror the neural network's function, an algorithmic loop was chosen instead because it is concise, human-readable, and visually practical for educational presentations.

To secure this structural stability and ensure 100% reliability, the model accepts the following technical constraints:

* **No Algorithmic Generalization:** The model is structurally locked to an input length of 8 elements.
* **Closed Universe Dependency:** 100% reliability requires total population coverage during training to eliminate gradient descent biases.

Ultimately, these constraints serve the artifact's core purpose: illustrating a conceptual framework to help researchers prepare to think about artificial intelligence through the lens of Formal AI. In the live presentation, the compiled hexadecimal representations of both the Python script (the `.pyc` file) and the Multilayer Perceptron (the `.pt` model) will be displayed side-by-side. The point of this comparison is to show that in their compiled forms, both are inherently opaque and difficult for humans to understand. Traditional machine learning attempts to extract knowledge or explainability from that opaque model after the fact. Formal AI contends that we should instead begin from the outset with human-readable source code and understandable data. In this paradigm, the program itself is the model, erasing the distinction between a system's description, explanation, and implementation.

## 8. Disclaimer

I designed this project in collaboration with Google Gemini 3.1 Pro and Google Antigravity IDE using the Gemini Flash 3 and Pro 3.1 models. This specific README document is primarily written by AI with HITL, where I acted as an editor. The goal of using run-length encoding (RLE) as a simple demonstration goal, choice of programming languages, and the actual decisions on what to use were my choices. I justified those choices through evaluation tools and tests, rather than relying on the AI.
