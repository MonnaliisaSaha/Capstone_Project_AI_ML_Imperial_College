# Model Card — BBO Optimisation Pipeline

---

## Overview

**Name:** GP + UCB + SVM + NN Bayesian Optimisation Pipeline  
**Type:** Surrogate-guided sequential black-box optimisation  
**Version:** 2.0 (complete — Rounds 1 through 13)

This pipeline selects one query point per function per round using a four-component surrogate stack — Gaussian Process, Upper Confidence Bound acquisition, SVM region filter, and Neural Network gradient refinement. Each component was added incrementally across rounds as the pipeline matured.

---

## Intended Use

**Suitable for**

- Educational demonstration of Bayesian Optimisation under strict evaluation budgets
- Sequential query selection for multi-dimensional black-box functions
- Comparing surrogate-guided versus random or grid search strategies
- Reproducible optimisation workflow documentation

**Use cases to avoid**

- Safety-critical or regulatory decision-making
- Contexts requiring formally calibrated uncertainty estimates
- Functions known to be discontinuous or to have isolated spike-like peaks — the smoothness assumption in the GP kernel will cause the surrogate to model these poorly
- Parallel or batch evaluation settings — the pipeline is strictly sequential

---

## Strategy and Evolution

The pipeline was built one component at a time. Each round's approach was driven by what the data from previous rounds had shown.

**Rounds 1–4: Building the pipeline**

Round 1 used pure random sampling to establish a baseline. Round 2 introduced the Gaussian Process surrogate with UCB acquisition using a uniform beta of 2.576 across all functions. Round 3 added an SVM region classifier trained on the top 30% of observed outputs — this was the most productive round of the project, producing three all-time bests in a single week. Round 4 added a two-layer neural network used for gradient analysis: backpropagation at the current best point estimates which direction in the input space would improve the output.

**Round 5: Per-function configuration**

The most important structural change in the project. Functions showed clearly different characters by Round 5 — F5 had a reliable exploitable gradient, F1 had almost no signal, F4 kept oscillating. A single beta of 2.576 was too blunt. Round 5 introduced function-specific beta, step size, SVM percentile and NN weight decay, tuned based on observed behaviour.

**Rounds 6–10: Exploitation and recovery**

The pipeline ran in near-full exploitation mode for functions with clear gradient signals, while maintaining exploration for functions with no reliable pattern. A return-to-best logic was introduced: when a function regressed, it was returned to its last confirmed best coordinates rather than pushed further.

**Rounds 11–13: Final lock-in**

The strategy became almost purely exploitative. F5 was pushed along its x1 gradient for three final rounds (x1: 0.550 → 0.580 → 0.610), producing new bests each time. F4 and F7 were returned to their exact confirmed best coordinates and held there. F8 received only micro-refinements. F2, F3 and F6 received final perturbation attempts before locking in their best historical results.

**Full pipeline (Round 6 onwards)**

```
Step 1  Fit GP on all accumulated observations
        Kernel: Matern 2.5 (assumes smoothness + once-differentiability)
        normalize_y=True

Step 2  Generate 10,000 random candidate points in [0,1]^n

Step 3  SVM filter: train on top 30% of outputs as positive class
        RBF kernel, removes candidates in unpromising regions

Step 4  UCB scoring: score = mean + beta × std
        beta tuned per function based on trajectory
        Low beta = exploit, High beta = explore

Step 5  Train 2-layer NN (32 units, ReLU, PyTorch) on all observations
        Backpropagate at current best to get gradient direction
        Nudge best UCB candidate by step_size in gradient direction

Step 6  Clip to [0,1]^n. Format. Submit.
```

---

## Performance

Metric used: all-time maximum output achieved per function (Ymax).

| Function | Scenario | Peak output | Round achieved |
|----------|---------|------------|----------------|
| F1 (2D) | Radiation source detection | 2.68e-9 | R5/R7 |
| F2 (2D) | Noisy ML model score | 0.6202 | R6 |
| F3 (3D) | Drug compound optimisation | -0.0045 | R3 |
| F4 (4D) | Warehouse logistics | +0.5534 | R7 / confirmed R9 |
| F5 (4D) | Chemical process yield | 2798.79 | R10 |
| F6 (5D) | Recipe scoring | -0.2957 | R3 |
| F7 (6D) | ML hyperparameter tuning | 1.8116 | R7 |
| F8 (8D) | High-dimensional search | 9.9250 | R3 |

**Function-specific notes**

F5 is the standout result — the x1 gradient direction produced improvement in ten consecutive rounds (R4–R13), from ~800 in Round 1 to 3166 in Round 13. This is exactly the kind of reliable signal Bayesian Optimisation is designed to find and exploit.

F4 has a narrow but reproducible peak. The same output (0.5534) was produced at identical coordinates in Rounds 7, 9, 12 and 13. Four independent reproductions confirm the peak is a real feature of the function.

F7 similarly reproduced its best (1.8116) at identical coordinates in Rounds 7, 12 and 13. The gradient direction established in Round 6 held throughout the project.

F2 is genuinely stochastic — identical coordinate regions returned 0.6202 in Round 6 and 0.467 in Round 12. The GP smoothness assumption is violated here, and any exploitation of this function carries higher uncertainty than the model estimates.

F3, F6 and F8 all achieved their best results in Round 3 and were not recovered. The SVM filter located these peaks early but subsequent rounds were unable to return to them precisely — likely because the peaks are narrow and the step sizes used for refinement were slightly too large.

F1 returned near-zero across all thirteen rounds, suggesting either an extremely narrow spike or a non-smooth surface that the GP kernel cannot represent accurately. A boundary diagnostic in Rounds 1–2 would have been the correct approach.

---

## Assumptions and Limitations

**Core assumptions**

Local smoothness — the GP, SVM and NN all assume that nearby inputs produce similar outputs. This assumption holds for F5 but is likely violated by F2 (genuinely noisy) and potentially F1 (possible spike or discontinuity).

Stationarity — the pipeline assumes each function's landscape is fixed across all rounds. F4's scenario (warehouse logistics) may imply dynamic behaviour, which could explain why its peak is narrow and small deviations cause significant regression.

Gradient reliability — the NN gradient is only informative when the surrogate has enough data to model the local surface accurately. With 10–23 observations in eight dimensions, this assumption is weakest for F8.

**Limitations**

Small dataset — 23 observations per function after Round 13. The GP extrapolates for most of the search space rather than interpolating from dense coverage.

Exploitation bias — adaptive search concentrates queries in high-performing regions. Large areas of the search space, especially in high-dimensional functions, are never sampled.

One query per round — standard Bayesian Optimisation benefits from parallel candidate evaluations to validate surrogate recommendations. A single bad query can misdirect the strategy for multiple rounds before enough new evidence accumulates to correct it.

No formal uncertainty calibration — the GP provides uncertainty estimates but these are not calibrated against held-out data. Surrogate confidence should be treated qualitatively rather than as a reliable probability.

---

## Ethical Considerations

This pipeline operates on entirely synthetic numerical data with no connection to real people, personal information or consequential decisions. No ethical review was required.

Transparency and reproducibility are built into the structure of this repository. Every query decision is documented with its rationale in each round's README. The scripts are fully commented so another researcher can follow and reproduce the exact workflow. The strategy log records how beta, step size and other hyperparameters evolved across rounds, making it possible to audit why each function was handled differently.

---

## Reflection

The most valuable thing this project demonstrated is that a principled surrogate-guided approach outperforms uniform strategies — but only when the surrogate's assumptions match the function's actual behaviour. The functions where the pipeline performed best (F5, F4, F7) are the ones where the smoothness assumption held and the gradient direction was reliable. The functions where the pipeline struggled most (F1, F2, F3) are precisely the ones where that assumption was weakest.

The single most impactful improvement across the project was the introduction of per-function beta in Round 5. Before that, a uniform configuration was applied to all eight functions regardless of how differently they were behaving. After that, the strategy adapted to the evidence. That principle — letting data drive the decision rather than applying one approach to everything — is the most transferable lesson from this project.
