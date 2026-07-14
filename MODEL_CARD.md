## Model Description

**Input:** Multi-dimensional input vector X of shape (n_dims,) where n_dims is 2–8 depending on the function. All values bounded in [0, 1]. At query time, the pipeline also receives all accumulated historical observations (X_history, y_history) from prior rounds — these train the surrogate models.

**Output:** A single query point — a vector of shape (n_dims,) representing the next input to submit to the black-box function for evaluation. One query per function per round.

**Model Architecture:** A four-component sequential pipeline:

```
Step 1  GP (Gaussian Process, Matern 2.5 kernel, normalize_y=True)
        → trained on all accumulated observations
        → outputs mean prediction + uncertainty (std) per candidate

Step 2  UCB (Upper Confidence Bound) acquisition
        → score = mean + beta × std
        → beta tuned per function (0.5 exploit ↔ 3.0 explore)
        → selects best candidate from 10,000 random points in [0,1]^n

Step 3  SVM (Support Vector Machine, RBF kernel) region filter
        → trained on top 30% vs bottom 70% of observed outputs
        → removes candidates in unpromising regions before UCB scoring

Step 4  NN (Neural Network, 2-layer MLP, 32 units, ReLU, PyTorch)
        → trained on all observations
        → gradient computed at current best via backpropagation
        → nudges UCB-selected candidate in direction of steepest ascent

Step 5  Clip to [0,1]^n → format → submit
```

Built incrementally: GP added Round 2, SVM+UCB Round 3, NN Round 4, per-function configuration Round 5.

---

## Performance

Performance measured as the all-time maximum output achieved per function (Ymax) across 13 rounds of one query per function per week.

| Function | Scenario | Starting value (R1) | Best achieved | Round | Improvement |
|----------|---------|---------------------|--------------|-------|-------------|
| F1 (2D) | Radiation detection | ~0 | 2.68e-9 | R5/R7 | Minimal — no signal found |
| F2 (2D) | Noisy ML scoring | 0.105 | 0.6202 | R6 | +490% — noisy, not reproducible |
| F3 (3D) | Drug optimisation | -0.032 | -0.0045 | R3 | Improvement — never recovered |
| F4 (4D) | Warehouse logistics | -0.244 | +0.5534 | R7/R9/R11/R12/R13 | Sign flip — 5 reproductions |
| F5 (4D) | Chemical yield | ~800 | 3166.01 | R13 | ~4x — 10 consecutive improvements |
| F6 (5D) | Recipe scoring | -1.007 | -0.2957 | R3 | Improvement — landscape resisted |
| F7 (6D) | ML hyperparameters | 0.747 | 1.8116 | R7/R12/R13 | +142% — 3 reproductions |
| F8 (8D) | High-dim search | 9.820 | 9.9250 | R3 | Near ceiling from R3 onwards |

**Strongest performer:** F5 — monotone gradient found in Round 4, exploited for 10 consecutive rounds, improving from ~800 to 3166.

**Most robust result:** F4 — peak of 0.5534 reproduced identically at [0.360, 0.410, 0.430, 0.395] in Rounds 7, 9, 11, 12 and 13. Five independent reproductions confirm a real, stable feature.

**Most challenging:** F1 — 13 rounds produced near-zero outputs throughout. True peak likely in a region never sampled.

---

## Limitations

**Small dataset regime:** 23 observations per function at completion. The GP surrogate extrapolates for most of the input space — confidence should be treated qualitatively, not as calibrated probability.

**Smoothness assumption:** The Matern 2.5 kernel assumes smooth, once-differentiable landscapes. This assumption holds for F5 and F7 but is likely violated by F2 (genuinely stochastic), F1 (possible spike), and F3 (fragile narrow peak).

**One query per round:** Standard Bayesian optimisation benefits from parallel evaluations to validate surrogate recommendations. A single bad query can misdirect the strategy for multiple rounds before enough evidence accumulates to correct it.

**No formal uncertainty calibration:** GP uncertainty estimates are not calibrated against held-out data. The model's expressed confidence should not be treated as a reliable probability.

**Exploitation bias in later rounds:** Adaptive search concentrates queries in high-performing regions. Large areas of the input space — especially in high-dimensional functions — are never sampled. F8's 8D space with 23 observations covers an infinitesimally small fraction of the possible inputs.

**NN gradient reliability:** The neural network gradient estimate is only informative when the surrogate has sufficient data to model the local surface. In early rounds with sparse data, gradient directions can be misleading. Most reliable from Round 10 onwards.

---

## Trade-offs

**Exploration vs exploitation — the central trade-off**

Higher UCB beta increases exploration but reduces exploitation efficiency. In early rounds, high beta was necessary because the landscape was unknown. In later rounds, high beta on functions with confirmed peaks (F5, F4) would have wasted queries on uncertain regions with no payoff. The per-function beta configuration resolved this — but required ongoing human judgement each round rather than an automatic mechanism.

Functions where exploitation was prioritised too early suffered regressions (F4, Round 8 — a nudge from the confirmed peak returned 0.361). Functions where exploration was maintained too long gave up potential exploitation gains.

**Model complexity vs data availability**

The four-component pipeline is more sophisticated than the data volume justifies in early rounds. With 10–15 observations, the NN gradient is unreliable and the SVM classifier has very few examples of each class. Adding the NN in Round 4 was the right timeline — earlier would have added noise rather than signal.

Conversely, the GP surrogate performs well even with 10 observations because it is designed for small-data settings. The Matern kernel with `normalize_y=True` is a good default choice for this regime.

**Per-function configuration vs unified pipeline**

The adaptive per-function beta configuration produced better results than a unified approach but required weekly human judgement. A fully automatic adaptive mechanism (e.g., Thompson sampling, which self-tunes without a manual beta parameter) would be more robust at the cost of interpretability.

**Sample efficiency vs coverage**

Committing to exploitation early (F5 from Round 4) produced strong results but meant large areas of the 4D space were never explored. If F5 had a second, higher peak in an unexplored region, the strategy would have missed it entirely. The same trade-off applies to F8 — micro-exploitation of the confirmed plateau region may have locked in a local rather than global maximum.

**Short-term gains vs long-term information value**

Several rounds produced no score improvement but gave useful information (confirming F4's peak is reproducible, confirming F8's plateau structure). These rounds were not failures — their value was in reducing uncertainty about the landscape. A strategy that prioritises only immediate score improvement would have made different, worse decisions.

---

## Ethical Considerations

This pipeline operates on entirely synthetic numerical data with no connection to real people, personal information or consequential decisions. No ethical review was required.

Transparency and reproducibility are built into the repository structure. Every query decision is documented with its rationale in each round's README. Scripts are fully commented. The strategy log records how hyperparameters evolved across all 13 rounds.

For any real-world application of this pipeline, additional validation would be required: uncertainty calibration against held-out data, robustness testing across different function types, and domain-appropriate governance before deployment.
