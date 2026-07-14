# Black-Box Optimisation (BBO) Capstone

**Monalisa Saha** · Imperial College London Professional Certificate in ML & AI  
Reading, UK · [LinkedIn](https://www.linkedin.com/in/monnaliisa-saha/)

---

This repository documents a thirteen-round optimisation challenge across eight hidden black-box functions. The goal: find the input that maximises each function's output — without ever seeing the equation, the gradient, or the shape of the landscape. One query per function per week. Every choice counts.

**Project Documentation**
- [Datasheet for Dataset](DATASHEET.md) — composition, collection process, gaps and assumptions
- [Model Card](MODEL_CARD.md) — pipeline details, strategy evolution, performance and limitations
- [Strategy Log](docs/strategy_log.md) — round-by-round decisions and beta evolution
- [Results Summary](docs/results_summary.md) — all outputs across all 13 rounds
---

This project tackled a challenge called Black-Box Optimisation — finding the best possible inputs for eight hidden mathematical functions, where the only feedback available is the output value returned each week. Think of it as a treasure hunt where you get one dig per location per week and no map. Over 13 weeks, an intelligent search system was built that learned from each result, decided whether to keep exploring new territory or dig deeper in promising areas, and adapted its strategy separately for each of the eight functions. The system combined four core components — a Gaussian Process to model the landscape, Upper Confidence Bound scoring to balance exploration and exploitation, a Support Vector Machine to filter poor regions early, and a Neural Network to estimate the best direction to move. The strategy was further shaped by reinforcement learning principles, clustering analysis, and dimensionality reduction insights drawn from the broader Imperial College ML & AI programme. 

## The Eight Functions

| Function | Dimensions | Real-world scenario |
|----------|-----------|---------------------|
| F1 | 2D | Radiation source detection — only proximity to the source gives a non-zero signal |
| F2 | 2D | Noisy ML model scoring — outputs carry noise and the landscape has multiple local peaks |
| F3 | 3D | Drug discovery — find the compound combination that minimises adverse reactions |
| F4 | 4D | Warehouse logistics — tune the hyperparameters of a costly placement model |
| F5 | 4D | Chemical process yield — unimodal function, one clear peak to find |
| F6 | 5D | Recipe scoring — five ingredient amounts, total score is negative by design |
| F7 | 6D | ML hyperparameter tuning — six parameters, maximise model performance |
| F8 | 8D | High-dimensional hyperparameter search — find the strongest local maximum |

All inputs are in `[0, 1]^n`. All tasks are maximisation problems.

---

## Pipeline

Built incrementally across rounds — one component added at a time:

```
Round 1   Random sampling — establish baseline
Round 2   Gaussian Process (GP) surrogate added
Round 3   UCB acquisition function added
Round 4   SVM region filter added
Round 5   Neural Network gradient refinement added
Round 6+  Full pipeline with adaptive per-function configuration
```

## Pipeline Components — Full Detail

### 1. Gaussian Process (GP) — the map
* Bayesian Optimisation and Gaussian Processes *

A Gaussian Process is a probabilistic surrogate model. Given everything observed so far, it estimates two things for every candidate point in the search space: the predicted output value (mean) and how uncertain that prediction is (standard deviation). Points with dense nearby observations get tight, confident predictions. Points far from any observed data get wide, uncertain estimates.

The kernel used throughout is **Matern 2.5** — it assumes the function is smooth and once-differentiable without being overly optimistic. `normalize_y=True` was set to handle the scale differences between functions (F5 outputs are in the thousands; F1 outputs are near zero).

```python
kernel = Matern(nu=2.5)
gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)
gp.fit(X_observed, y_observed)
mean, std = gp.predict(candidates, return_std=True)
```

### 2. Upper Confidence Bound (UCB) — the scoring rule
* Exploration-Exploitation Trade-offs *

UCB scores every candidate point using the formula:

```
score = mean + beta × std
```

The **mean term** exploits — high predicted value means probably good.  
The **std term** explores — high uncertainty means worth investigating.  
**Beta** controls the balance. The key insight from Module 13: a single beta value is too blunt for functions that behave differently. From Round 5 onwards, beta was tuned independently per function:

| Function | Final beta | Reasoning |
|----------|-----------|-----------|
| F1 | 2.5 | No signal found — keep exploring |
| F2 | 1.5 | Noisy — moderate, exploit confirmed region |
| F3 | 1.5 | Fragile peak — careful search |
| F4 | 1.5 | Narrow but real — tight exploit |
| F5 | 0.8 | Reliable gradient — pure exploitation |
| F6 | 2.0 | Multimodal — broad search maintained |
| F7 | 1.0 | Direction established — exploit |
| F8 | 0.5 | Near ceiling — micro-exploitation only |

### 3. Support Vector Machine (SVM) — the region filter
* Support Vector Machines *

Before UCB scoring, the 100,000 candidate points are filtered using an SVM binary classifier. The SVM is trained on all accumulated observations, labelling the top 30% of outputs as "promising" (positive class) and the remaining 70% as "unpromising" (negative class). Only candidates that the SVM classifies as promising are passed through to UCB scoring.

This was the single most impactful addition in the project — Round 3 produced three all-time bests in one week after the SVM filter was introduced. The filter narrows the search space from noise to signal before the expensive GP scoring even begins.

```python
threshold = np.percentile(y, 70)          # top 30% = positive class
labels = (y >= threshold).astype(int)
svm = SVC(kernel="rbf", C=1.0, gamma="scale")
svm.fit(X_scaled, labels)
promising_mask = svm.predict(candidates_scaled) == 1
```

### 4. Neural Network (NN) — the gradient pointer
* Neural Networks, TensorFlow and PyTorch *

A two-layer MLP (32 hidden units, ReLU activations) is trained on all accumulated observations. Rather than using its predictions directly, its **gradient** is used — backpropagating from the output to the input at the current best point to find which direction in the input space would increase the predicted output.

```python
x_t = torch.FloatTensor(x_best).requires_grad_(True)
output = nn_model(x_t)
output.backward()
gradient = x_t.grad.detach().numpy()
gradient /= np.linalg.norm(gradient) + 1e-8   # normalise to unit length
next_point = x_best + step_size * gradient     # nudge in improving direction
```

This was most reliable in the later rounds once the GP had 20+ observations — in early rounds with sparse data the gradient estimate was noisy and not trusted.

### 5. Reinforcement Learning (RL) framework — the decision policy
* Reinforcement Learning and Exploration-Exploitation *

The BBO challenge is structurally identical to a Reinforcement Learning problem:

| RL concept | BBO equivalent |
|-----------|---------------|
| **State** | All observations accumulated so far |
| **Action** | The next query point submitted |
| **Reward** | The output value returned by the portal |
| **Policy** | The per-function beta + step configuration |
| **Exploration** | High beta — query uncertain regions |
| **Exploitation** | Low beta — refine known good regions |
| **Q-value update** | GP refitted on new observation each round |
| **Episode** | One round of 8 function queries |

The key insight from studying RL: **a fixed policy is almost always wrong in a sequential setting**. The initial uniform beta of 2.576 was equivalent to a fixed policy that treated all eight functions as the same environment. The per-function adaptive beta introduced in Round 5 is analogous to a learned policy — different actions for different states, updated as evidence accumulates.

The **epsilon-greedy** principle (occasionally explore even when exploiting) also shaped the later-round strategy: functions that were being heavily exploited (F5, F8) still received occasional wider probes when the gradient signal seemed to be flattening — analogous to the epsilon term preventing premature convergence to a local maximum.

### 6. Clustering — landscape characterisation
* Unsupervised Learning and Clustering *

K-means clustering was applied analytically (not as a query-selection component) to the accumulated observations for each function. By clustering input coordinates and examining which clusters produced the highest output values, it was possible to identify the spatial structure of the landscape — whether the high-value region was a single compact basin (F5, F4) or scattered across multiple areas (F2, F6).

For F4 in particular, clustering confirmed that the true peak coordinates [0.360, 0.410, 0.430, 0.395] were isolated — no other cluster came close. This informed the decision to return to exact coordinates rather than probing adjacent points in later rounds.

### 7. Principal Component Analysis (PCA) — dimension importance
* Dimensionality Reduction and Unsupervised Learning *

PCA was applied analytically to the input coordinates of high-performing observations to understand which dimensions carried the most signal. For F7 (6D), PCA of the top quartile of observations showed that variation was concentrated in the x1 and x6 dimensions — consistent with the coordinates [0.050, ..., 0.770] that were confirmed as the best region from Round 7 onwards.

For F5 (4D), PCA confirmed that x1 was the dominant dimension driving output — reinforcing the gradient strategy of incrementally increasing x1 each round while holding x2, x3, x4 near their optimal values.

This is conceptually related to the **Attention mechanism** from Module 19 (Transformers and LLMs): just as attention weights indicate which input tokens matter most for a prediction, PCA loadings indicate which input dimensions carry the most signal for a function. Both are about identifying what to pay attention to in high-dimensional data.

---

## The Bigger Picture — What the Modules Taught Through Doing

The BBO capstone was more than a coding exercise. It put every concept from the programme into a setting where the stakes were real (one query per week, no second chances) and the feedback was immediate. 

**Bayesian thinking** : A surrogate model is a belief about the world. When new evidence arrives, the belief updates. The GP does this formally — every observation changes every prediction. The discipline of asking "what does the data actually support?" before acting on a model recommendation became the most important habit of the project.

**Classification under uncertainty** : The SVM filter showed that even a binary good/bad classifier with 15–20 training examples could meaningfully narrow a 100,000-point search space. Classification doesn't need to be perfect to be useful — it needs to be better than random, and the SVM consistently was.

**Gradient-based optimisation** : The NN gradient demonstrated that the direction of improvement matters more than the magnitude. A small step in the right direction (F5's x1 gradient) outperformed large steps in uncertain directions across every round from Round 4 to Round 13.

**Sequential decision-making** : The project confirmed that black-box optimisation under a strict budget is fundamentally an RL problem. The best decisions were not the ones that looked best in isolation — they were the ones that considered what information the query would provide for future rounds.

**Unsupervised learning as analysis, not search** : Clustering and PCA did not drive queries directly, but they informed the reasoning. Knowing which dimensions mattered (PCA) and where the high-value regions clustered (K-means) made every subsequent decision better-grounded than surrogate confidence alone.

---

## Results — All 13 Rounds

| Round | F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 |
|-------|-----|-----|------|-----|------|------|-----|------|
| R1 | ~0 | 0.1051 | -0.0319 | -0.2441 | ~800 | -1.007 | 0.747 | 9.820 |
| R2 | ~0 | 0.230 | -0.028 | -0.180 | ~900 | -0.900 | 0.950 | 9.870 |
| R3 | 2.8e-9 | 0.410 | **-0.0045** | 0.120 | 1200 | **-0.2957** | 1.200 | **9.9250** |
| R4 | 1.2e-9 | 0.380 | -0.018 | 0.310 | 1450 | -0.650 | 1.400 | 9.880 |
| R5 | **2.68e-9** | 0.490 | -0.022 | 0.280 | 1800 | -0.710 | 1.550 | 9.900 |
| R6 | 3.6e-10 | **0.6202** | -0.016 | 0.499 | 2366.81 | -0.793 | 1.7982 | 9.9238 |
| R7 | 2.68e-9 | 0.537 | -0.022 | **0.5534** | 2512.64 | -0.792 | **1.8116** | 9.9236 |
| R8 | 1.65e-9 | 0.523 | -0.020 | 0.361 | 2583.58 | -0.717 | 1.791 | 9.9222 |
| R9 | 2.68e-9 | 0.441 | -0.028 | 0.553 | 2699.31 | -0.707 | 1.807 | 9.9228 |
| R10 | 2.62e-9 | 0.564 | -0.036 | 0.486 | 2798.79 | -0.698 | 1.804 | 9.9230 |
| R11 | 2.68e-9 | 0.538 | -0.016 | **0.5534** | 2908.73 | -0.794 | 1.811 | 9.9228 |
| R12 | 2.68e-9 | 0.467 | -0.017 | **0.5534** | 3030.59 | -0.779 | **1.8116** | 9.9220 |
| R13 | 2.68e-9 | 0.512 | -0.023 | **0.5534** | **3166.01** | -0.683 | **1.8116** | 9.9220 |

**Bold** = all-time best for that function.

---

## Repository Structure

```
├── README.md                   — this file
├── DATASHEET.md                — dataset motivation, composition and limitations
├── MODEL_CARD.md               — pipeline details, strategy evolution and performance
├── requirements.txt
│
├── docs/
│   ├── strategy_log.md         — round-by-round decisions and beta evolution
│   └── results_summary.md      — all outputs and confirmed best coordinates
│
├── Round_01_README.md          — Round 1: random baseline — no surrogate
├── Round_01.py
├── Round_02_README.md          — Round 2: Gaussian Process introduced
├── Round_02.py
├── Round_03_README.md          — Round 3: SVM filter added — best single round
├── Round_03.py
├── Round_04_README.md          — Round 4: NN gradient refinement added
├── Round_04.py
├── Round_05_README.md          — Round 5: per-function beta introduced
├── Round_05.py
├── Round_06_README.md          — Round 6: return-to-best logic — 6/8 improved
├── Round_06.py
├── Round_07_README.md          — Round 7: hyperparameter tuning — 3 all-time bests
├── Round_07.py
├── Round_08_README.md          — Round 8: recovery round
├── Round_08.py
├── Round_09_README.md          — Round 9: F4 returned to exact confirmed coordinates
├── Round_09.py
├── Round_10_README.md          — Round 10: explicit decision documentation begins
├── Round_10.py
├── Round_11_README.md          — Round 11: F5 x1 push (0.550), F4 locked
├── Round_11.py
├── Round_12_README.md          — Round 12: penultimate push, F5 x1=0.580
├── Round_12.py
├── Round_13_README.md          — Round 13: final round, F5 x1=0.610, all-time best
└── Round_13.py
```

---

## Running the Scripts

```bash
pip install -r requirements.txt
python Round_05.py
```

Scripts use placeholder observations — replace `X_obs` and `y_obs` with accumulated portal data before running. Round 12 and Round 13 scripts include the actual submitted coordinates and received outputs in the main block for direct reference.

**Stack:** Python 3.10 · NumPy · scikit-learn · PyTorch · Matplotlib

---

## Key Learnings

The BBO challenge compressed the entire Imperial College ML & AI programme into a setting where every decision had a real cost and every result was immediate feedback. The most important lesson was not about any specific algorithm — it was about **when to trust a model and when to question it**. The GP, SVM, NN and RL framework each contributed something different to that judgement:

- The **GP** told you where the model thought the treasure was — but also how confident it was. Low confidence means explore. High confidence near a confirmed peak means exploit.
- The **SVM** showed that even a simple binary classifier trained on 20 observations could dramatically narrow a 100,000-point search space. Cheap filters before expensive models is a pattern that generalises everywhere.
- The **NN gradient** showed that direction matters more than magnitude. Following a confirmed gradient (F5's x1) for 10 consecutive rounds produced more total gain than any single large exploratory leap.
- The **RL framing** made the biggest conceptual contribution: viewing each query as an action in a sequential decision process, where the reward is the output and the policy is the per-function configuration, made it natural to ask "what will this query teach me?" rather than just "what does the model say?"
- **Clustering and PCA** proved most valuable not as search tools but as analytical tools — confirming which regions and dimensions actually mattered before committing further queries there.

The functions that improved the most (F5, F4, F7) were the ones where the surrogate assumptions held and the strategy adapted to match. The functions that resisted (F1, F3, F6) were the ones where the GP's smoothness assumption was wrong and the strategy did not adapt fast enough. That asymmetry — between functions the model understood and functions it did not — is the most honest summary of what black-box optimisation under a strict budget actually looks like in practice.
