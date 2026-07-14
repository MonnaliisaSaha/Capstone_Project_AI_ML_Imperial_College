# Strategy Log

## The pipeline analogy

Think of it as a treasure hunt with a limited number of digs:

- **GP (Gaussian Process)** builds the map — using everything observed so far to estimate where the treasure might be and how uncertain that estimate is
- **UCB (Upper Confidence Bound)** scores the next dig — high score = high predicted value OR high uncertainty (worth checking)
- **SVM (Support Vector Machine)** marks bad areas off the map — filters out regions the data says are unpromising before UCB scoring even begins
- **NN (Neural Network)** points the shovel — computes gradient at the current best point and nudges the final query in the improving direction
- **RL (Reinforcement Learning) framing** governs the policy — each function's beta configuration is a learned action, updated every round based on reward (output received)
- **Clustering** reads the landscape analytically — K-means on accumulated observations identifies where high-value regions actually sit
- **PCA (Principal Component Analysis)** identifies which dimensions matter — used to confirm which input dimensions carry signal before committing more queries

---

## Round-by-round decisions

| Round | Key decision | Module connection | Why | Outcome |
|-------|-------------|-------------------|-----|---------|
| R1 | Random baseline | — | No information — spread wide | F5/F8 show early signals |
| R2 | GP surrogate added | Module 12: Bayesian Optimisation | Need a probabilistic model to guide search | More informed queries — mean + uncertainty per candidate |
| R3 | SVM filter added | Module 14: Support Vector Machines | Filter unpromising regions before UCB scoring | Best single round — 3 all-time bests in one week |
| R3 | UCB acquisition added | Module 13: Exploration-Exploitation | Score candidates by predicted value + uncertainty | Formal balance between explore and exploit |
| R4 | NN gradient added | Module 15: Neural Networks / PyTorch | Refine final query toward improving direction | Full pipeline operational — gradient now guiding each submission |
| R5 | Per-function beta | RL: policy differentiation | One beta is too blunt for 8 different landscapes — treat each as a separate state | Adaptive approach begins — exploit functions vs explore functions |
| R6 | Return-to-best logic | RL: reward signal | Regressed functions need recovery not more exploration — negative reward = change action | Best single round (6/8 improved) |
| R6–R8 | Clustering applied analytically | Unsupervised learning: K-means | Identify spatial structure of each function's landscape — single basin or scattered | Confirmed F4 peak is isolated; F5 has a single exploitable basin; F2/F6 are scattered |
| R7 | Hyperparameter tuning | All modules: configuration matters | Each function needs different beta, step size, SVM threshold, NN weight decay | 3 more all-time bests — F4 confirmed at 0.5534, F7 best at 1.8116 |
| R8 | Recovery | RL: policy correction | Several functions regressed after R7 nudges — override surrogate, return to known good state | F5 continued improving, F4 partially recovered |
| R8–R10 | PCA applied analytically | Module: Dimensionality Reduction | Identify which input dimensions carry signal — focus gradient effort on those | Confirmed x1 dominates F5; x1 and x6 drive F7 — reinforced gradient direction decisions |
| R9 | F4 exact coordinates | RL: deterministic policy for confirmed state | Nudge caused regression — return to confirmed best rather than explore adjacent | F4 matched R7 best exactly — 0.5534 reproduced |
| R10 | Explicit decision documentation | Transparency (Module 21: Datasheets/Model Cards) | Final rounds — document rationale for everything, not just results | F5 new best, others near peak — reasoning traceable |
| R11 | F5 x1 push to 0.550 | RL: exploit confirmed gradient direction | x1 gradient has improved every round since R4 — continue same policy | F5 new best (2908), F4 confirmed again |
| R12 | F5 x1 push to 0.580 | RL: exploit — penultimate push | One round left — one more step along the confirmed gradient | F5 new best (3030), F3 best late result (-0.017) |
| R13 | F5 x1 push to 0.610 | RL: final exploitation | Last query — maximum step size, confirmed direction | F5 all-time best (3166) — 10 consecutive improvements since R4 |

---

## Beta evolution per function

Beta controls exploration vs exploitation in UCB. Lower = exploit known good areas. Higher = explore uncertain areas. Managing beta per function is equivalent to running a separate RL policy per environment.

| Function | R1-3 | R4-5 | R6-7 | R8-11 | R12 | R13 | Notes |
|----------|------|------|------|-------|-----|-----|-------|
| F1 | 2.58 | 3.0 | 3.0 | 3.0 | 3.0 | 2.5 | Never found reliable signal — kept exploring, reduced at very end |
| F2 | 2.58 | 2.0 | 1.5 | 2.0 | 2.0 | 1.5 | Noisy — moderate throughout, exploit at final round |
| F3 | 2.58 | 2.5 | 2.0 | 2.0 | 2.0 | 1.5 | Fragile peak — careful search, tightened at end |
| F4 | 2.58 | 1.5 | 1.5 | 1.5 | 1.5 | 1.5 | Narrow but real — exploit carefully throughout |
| F5 | 2.58 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 | Reliable gradient — lowest beta, pure exploit from R4 |
| F6 | 2.58 | 3.0 | 3.0 | 2.5 | 2.5 | 2.0 | Multimodal — explore broadly, slight tighten at end |
| F7 | 2.58 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | Direction established R6 — exploit throughout |
| F8 | 2.58 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | Near plateau — very tight exploit from R4 |

---

## Key lessons learned

**What worked:**

- **Per-function beta from Round 5** — single most impactful change. Treating 8 functions as 8 separate RL environments, each with its own policy, produced better results than any algorithmic improvement
- **Return-to-best logic** — narrow peaks are fragile. Nudging from a regressed position compounds the error. When the reward drops, return to the confirmed good state and try a different action
- **SVM filter in Round 3** — produced the best single round of the project. Cheap classification before expensive GP scoring is a pattern that generalises to any tiered ML pipeline
- **NN gradient most reliable from R10 onwards** — with 20+ observations, gradient estimates became trustworthy. In early rounds with sparse data, the NN direction was noisy and received lower step sizes
- **Clustering confirmed basin structure** — knowing F4's peak was isolated (no competing cluster) justified committing to exact coordinates without deviation from R7 onwards
- **PCA confirmed F5's x1 dominance** — analytically confirming which dimension drove the output removed uncertainty from the gradient strategy and justified the aggressive x1 push in R11–R13

**What would have helped:**

- **Boundary diagnostics in Rounds 1–2** — submitting extreme input corners early would have identified F5's boundary-type gradient and possibly F1's true peak much sooner, saving 3–4 rounds
- **Hard stagnation rule** — a written rule ("if no improvement in 3 rounds, force a global reset rather than another local probe") would have caught F6 and F3 stagnation earlier. This was applied by instinct too late
- **Earlier clustering analysis** — clustering was applied analytically from R6 but could have been informative from R3, shaping which functions received exploration vs exploitation budget sooner
- **F2 noise detection earlier** — the stochastic nature of F2 (same inputs, different outputs across rounds) was recognised late. Flagging this in R4–R5 would have prevented wasted exploitation queries on a landscape that cannot be reliably exploited

**The RL framing in retrospect:**

Viewed as an RL problem across 13 episodes, the policy improved visibly over time. Early episodes (R1–R4) had a uniform policy — same action for all functions. Middle episodes (R5–R9) had a differentiated policy — different beta, step, and recovery logic per function. Final episodes (R10–R13) had a near-deterministic policy for confirmed functions and an adaptive policy for uncertain ones. The trajectory from uniform to differentiated to near-deterministic mirrors how RL policies mature: start broad, narrow as evidence accumulates, converge on the best known action as the budget runs out.
