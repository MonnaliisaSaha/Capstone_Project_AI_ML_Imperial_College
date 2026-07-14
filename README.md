# Black-Box Optimisation (BBO) Capstone

**Monalisa Saha** · Imperial College London Professional Certificate in ML & AI  
Reading, UK · [LinkedIn](https://www.linkedin.com/in/monnaliisa-saha/)

---

This repository documents a thirteen-round optimisation challenge across eight hidden black-box functions. The goal: find the input that maximises each function's output — without ever seeing the equation, the gradient, or the shape of the landscape. One query per function per week. Every choice counts.

**Project Documentation**
- [Datasheet for Dataset](DATASHEET.md) — composition, collection process, gaps and assumptions
- [Model Card](MODEL_CARD.md) — pipeline details, strategy evolution, performance and limitations
- [Strategy Log](strategy_log.md) — round-by-round decisions and beta evolution
- [Results Summary](results_summary.md) — all outputs across all 13 rounds
---
## Non-Technical Explanation

This project tackled a challenge called Black-Box Optimisation — finding the best possible inputs for eight hidden mathematical functions, where the only feedback available is the output value returned each week. Think of it as a treasure hunt where you get one dig per location per week and no map. Over 13 weeks, an intelligent search system was built that learned from each result, decided whether to keep exploring new territory or dig deeper in promising areas, and adapted its strategy separately for each of the eight functions. The system combined four core components — a Gaussian Process to model the landscape, Upper Confidence Bound scoring to balance exploration and exploitation, a Support Vector Machine to filter poor regions early, and a Neural Network to estimate the best direction to move. The strategy was further shaped by reinforcement learning principles, clustering analysis, and dimensionality reduction insights drawn from the broader Imperial College ML & AI programme. 

## Data

**Source:** Eight synthetic black-box functions provided by Imperial College London via the BBO capstone portal. Functions simulate real-world problems including radiation source detection, drug compound optimisation, warehouse logistics tuning, chemical process yield, recipe scoring, and ML hyperparameter search.

**Structure:**
- 10 initial observations provided per function at project start (NumPy `.npy` arrays)
- 1 new observation added per function per week across 13 rounds
- Total: 23 observations per function at completion
- All inputs bounded in [0, 1]^n where n is 2–8 depending on function
- All outputs are scalars to be maximised

**Key characteristics:**
- Functions vary significantly in landscape type — unimodal (F5), noisy (F2), multimodal (F6), plateau (F8), fragile peak (F3, F4)
- No gradient information available — outputs only
- No equation or closed-form expression available — truly black-box
- Dataset is intentionally sparse — 23 observations in up to 8 dimensions covers a tiny fraction of the input space

Full documentation of each function's data structure, noise characteristics and observed landscape is in [DATASHEET.md](DATASHEET.md).

---

## Model

The pipeline combines four components, each added incrementally as the corresponding module was studied:

**1. Gaussian Process (GP) — the surrogate map**  
A probabilistic model trained on all accumulated observations. Uses a Matern 2.5 kernel — assumes smooth, once-differentiable landscapes. Outputs a mean prediction and uncertainty estimate for any candidate point. Chosen because it naturally quantifies uncertainty (essential for exploration-exploitation decisions) and performs well on small datasets (10–23 observations).

**2. Upper Confidence Bound (UCB) — the acquisition function**  
Scores candidates as `mean + beta × std`. Beta controls the explore-exploit trade-off and is tuned independently per function. Chosen over Expected Improvement because beta is interpretable and directly adjustable per-round based on observed function behaviour.

**3. Support Vector Machine (SVM) — the region filter**  
A binary classifier trained on above/below median outputs. Filters 100,000 random candidates to a high-probability region before GP scoring. Chosen as a cheap pre-filter — reduces the candidate pool by 70–80% at minimal computational cost.

**4. Neural Network (NN) — the gradient estimator**  
A two-layer MLP (32 units, ReLU) trained on all observations. Used not for its predictions but its gradient — backpropagation at the current best point estimates the direction of steepest ascent. Chosen to add directional refinement beyond what UCB candidate scoring provides.

Together: `GP → UCB → SVM → NN gradient nudge → submit`

---

## Hyperparameter Optimisation

The pipeline has four key hyperparameters, all tuned per-function from Round 5 onwards:

**UCB beta (exploration-exploitation balance)**  
Range used: 0.5–3.0. Tuned based on observed trajectory — functions showing clear gradient signals received low beta (exploit); functions with no reliable signal received high beta (explore). Updated each round based on whether the prior query improved, regressed, or held.

| Function | Final beta | Rationale |
|----------|-----------|-----------|
| F8 | 0.5 | Near ceiling — pure micro-exploitation |
| F5 | 0.8 | Reliable gradient — committed exploitation |
| F7 | 1.0 | Direction confirmed — hold and exploit |
| F2, F3, F4 | 1.5 | Moderate — exploit confirmed regions |
| F6 | 2.0 | No reliable direction — maintain exploration |
| F1 | 2.5 | No signal — broad exploration to the end |

**NN step size (gradient nudge magnitude)**  
Range used: 0.002–0.040. Smaller for functions near ceiling (F8: 0.002), larger for functions with clear rising gradients (F5: 0.040 final round).

**SVM percentile threshold**  
Fixed at top 30% throughout — defines which observations are labelled "promising". Not tuned per-function as 30% provided consistent filtering across all landscape types.

**NN weight decay (regularisation)**  
Range: 1e-4 to 1e-3. Lower for functions with stable, well-established gradients (F5, F8) where tighter fitting was justified. Higher for functions with noisy or uncertain landscapes.

---

## Results

**All-time best per function:**

| Function | Best output | Round achieved | Key learning |
|----------|------------|----------------|-------------|
| F1 (2D) | 2.68e-9 | R5/R7 | No signal found — centre point best despite 13 rounds |
| F2 (2D) | 0.6202 | R6 | Noisy — peak found but not reproducible |
| F3 (3D) | -0.0045 | R3 | Fragile peak — found early, never recovered |
| F4 (4D) | 0.5534 | R7/R9/R11/R12/R13 | Narrow but reproducible — confirmed 5 times |
| F5 (4D) | 3166.01 | R13 | Monotone gradient — 10 consecutive improvements |
| F6 (5D) | -0.2957 | R3 | Multimodal — best found early, landscape resisted search |
| F7 (6D) | 1.8116 | R7/R12/R13 | Direction found R6, confirmed 3 times |
| F8 (8D) | 9.9250 | R3 | Near ceiling since R3 — plateau confirmed |

**Score trajectory — all 13 rounds:**

| Round | F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 |
|-------|-----|-----|------|-----|------|------|-----|------|
| R1 | ~0 | 0.105 | -0.032 | -0.244 | ~800 | -1.007 | 0.747 | 9.820 |
| R2 | ~0 | 0.230 | -0.028 | -0.180 | ~900 | -0.900 | 0.950 | 9.870 |
| R3 | 2.8e-9 | 0.410 | **-0.0045** | 0.120 | 1200 | **-0.2957** | 1.200 | **9.9250** |
| R4 | 1.2e-9 | 0.380 | -0.018 | 0.310 | 1450 | -0.650 | 1.400 | 9.880 |
| R5 | **2.68e-9** | 0.490 | -0.022 | 0.280 | 1800 | -0.710 | 1.550 | 9.900 |
| R6 | 3.6e-10 | **0.6202** | -0.016 | 0.499 | 2366.81 | -0.793 | 1.798 | 9.924 |
| R7 | 2.68e-9 | 0.537 | -0.022 | **0.5534** | 2512.64 | -0.792 | **1.8116** | 9.924 |
| R8 | 1.65e-9 | 0.523 | -0.020 | 0.361 | 2583.58 | -0.717 | 1.791 | 9.922 |
| R9 | 2.68e-9 | 0.441 | -0.028 | 0.553 | 2699.31 | -0.707 | 1.807 | 9.923 |
| R10 | 2.62e-9 | 0.564 | -0.036 | 0.486 | 2798.79 | -0.698 | 1.804 | 9.923 |
| R11 | 2.68e-9 | 0.538 | -0.016 | **0.5534** | 2908.73 | -0.794 | 1.811 | 9.923 |
| R12 | 2.68e-9 | 0.467 | -0.017 | **0.5534** | 3030.59 | -0.779 | **1.8116** | 9.922 |
| R13 | 2.68e-9 | 0.512 | -0.023 | **0.5534** | **3166.01** | -0.683 | **1.8116** | 9.922 |

**Bold** = all-time best for that function.

**Key learning from the results:** Functions rewarded different strategies. F5's unimodal gradient justified pure exploitation from Round 4. F1's near-zero landscape resisted all approaches. F4's narrow but reproducible peak confirmed the value of exact-coordinate return-to-best logic. The biggest overall lesson: a single uniform strategy fails across functions with fundamentally different landscape structures. Per-function adaptive configuration — the pipeline's most important design decision — was what produced the strongest results.

---

## Repository Structure

```
├── README.md                   — this file
├── DATASHEET.md                — per-function documentation (all 8 functions)
├── MODEL_CARD.md               — pipeline architecture, performance, limitations, trade-offs
├── requirements.txt
│
├── docs/
│   ├── strategy_log.md         — round-by-round decisions and beta evolution
│   └── results_summary.md      — all outputs and confirmed best coordinates
│
├── Round_01_README.md          — Round 1: random baseline
├── Round_01.py
├── Round_02_README.md          — Round 2: GP surrogate introduced
├── Round_02.py
├── Round_03_README.md          — Round 3: SVM + UCB added — best single round
├── Round_03.py
├── Round_04_README.md          — Round 4: NN gradient added
├── Round_04.py
├── Round_05_README.md          — Round 5: per-function beta introduced
├── Round_05.py
├── Round_06_README.md          — Round 6: return-to-best logic — 6/8 improved
├── Round_06.py
├── Round_07_README.md          — Round 7: 3 all-time bests including F4 peak
├── Round_07.py
├── Round_08_README.md          — Round 8: recovery after R7 regressions
├── Round_08.py
├── Round_09_README.md          — Round 9: F4 returned to exact coordinates
├── Round_09.py
├── Round_10_README.md          — Round 10: explicit decision documentation
├── Round_10.py
├── Round_11_README.md          — Round 11: F5 x1=0.550, F4 confirmed
├── Round_11.py
├── Round_12_README.md          — Round 12: penultimate push, F5 x1=0.580
├── Round_12.py
├── Round_13_README.md          — Round 13: final round, F5 x1=0.610, 3166
└── Round_13.py
```

---

## Running the Scripts

```bash
pip install -r requirements.txt
python Round_05.py
```

Scripts include placeholder observations — replace `X_obs` and `y_obs` with accumulated portal data before running. Round 12 and 13 scripts include actual submitted coordinates and received outputs in the main block for direct reference.

**Stack:** Python 3.10 · NumPy · scikit-learn · PyTorch · Matplotlib

---

## Contact

Monalisa Saha · [LinkedIn](https://www.linkedin.com/in/monnaliisa-saha/) · sahamonalisa2014@gmail.com
