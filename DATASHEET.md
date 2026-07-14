# Datasheet — BBO Capstone Project
## All Eight Functions

---

# Function 1 — Radiation Source Detection (2D)

## Function overview

1. **Function:** F1 — Radiation Source Detection
2. **Real-world scenario:** Locating a 2D radiation source. Only proximity to the true source location returns a non-zero signal. The output represents radiation intensity at a given sensor position.
3. **Dimensionality:** 2D input space — [x1, x2], both bounded in [0, 1]
4. **Initial data points:** 10 observations provided at project start. Shape: inputs (10, 2), outputs (10,)
5. **Output represents:** Radiation intensity detected at the queried sensor position

## Nature of the data

1. **Initial dataset structure:** 10 input vectors of shape (10, 2), outputs of shape (10,). All input values in [0, 1]. Outputs near zero across all 10 baseline observations.
2. **Dataset evolution:** 13 new observations added across 13 rounds. Later queries clustered around [0.500, 0.500] — the centre point that returned the highest observed value. Large regions of the 2D space remain entirely unsampled.
3. **Noise/randomness:** No clear stochasticity detected. The same centre point coordinates consistently returned 2.68e-9 across multiple rounds. However, values varied between 1.2e-9 and 3.6e-10 at nearby coordinates, suggesting either a very narrow peak or numerical sensitivity.
4. **Landscape type:** Likely spike-like or highly localised — near-zero values across the entire sampled domain suggest the true peak is an extremely narrow feature. The GP surrogate could not model this structure meaningfully as the smooth kernel assumption requires gradual variation, which is absent here.

## Optimisation strategy

1. **Methods used:** GP + UCB (beta=3.0) for exploration, SVM filter, NN gradient. UCB beta held at maximum throughout — no exploitable signal was ever found to justify reducing it.
2. **Why this method:** GP-UCB is the standard approach for uncertain landscapes. Beta=3.0 maximises exploration, which is appropriate when no reliable gradient direction has been found after 13 rounds.
3. **Exploration-exploitation balance:** Fully exploration throughout. Beta never reduced below 3.0 for F1. Return-to-best logic held the centre point [0.500, 0.500] as the anchor.
4. **Strategy changes:** The only change was a minor beta reduction to 2.5 in Round 13 — no material difference. In retrospect, a boundary diagnostic (submitting corner points [0, 0], [1, 0], [0, 1], [1, 1]) in Rounds 1–2 may have located the true source.

## Data handling and preprocessing

1. **Rescaling:** No rescaling of inputs — naturally bounded in [0, 1]. GP uses `normalize_y=True` to handle the extreme scale (near 1e-9) without distorting kernel fitting.
2. **Surrogate model:** GP with Matern 2.5 kernel. With all outputs near zero, the GP had almost no signal to learn from — its predictions were essentially uninformative.
3. **Preprocessing:** `normalize_y=True` was essential — without it, the GP would treat all outputs as identical (near zero). A log transform was considered but not applied, as negative or zero values would cause problems.
4. **Outliers:** No meaningful outliers — outputs were uniformly near-zero. The highest value (2.68e-9) was not an outlier but the confirmed best.

## Weekly iteration and learning

1. **How new data changed understanding:** Each round confirmed the near-zero floor across the sampled region. By Round 5, it was clear the true peak was either at the centre or in an entirely unsampled region. New data reinforced the hypothesis of a very narrow spike.
2. **Local optima:** The entire sampled domain may be a local flat region far from the true peak. There is no way to confirm whether [0.500, 0.500] is near the global maximum or simply the best of a uniformly poor set of samples.
3. **Most informative inputs:** None were especially informative — all returned near-zero. The most useful observation in retrospect was that the centre point consistently returned the highest value (2.68e-9), establishing it as the best-known anchor.
4. **What to do differently:** Submit boundary points [0.1, 0.1], [0.9, 0.9], [0.1, 0.9], [0.9, 0.1] in Rounds 1–2 to map the corners before committing to centre-region exploration.

## Performance and results

1. **Best output:** 2.6752879910742468e-9
2. **Best input vector:** [0.500, 0.500]
3. **Confidence in global maximum:** Very low. With 23 total observations in a 2D space, vast regions are unsampled. The true peak is likely in a location never queried.
4. **Aligned with expectations:** Partially — the radiation source scenario implies a highly localised signal, consistent with the near-zero observations everywhere else. The challenge was that the GP surrogate is poorly suited to spike-like landscapes.

## Ethical, practical and general considerations

1. **Real-world applications:** Nuclear facility monitoring, environmental contamination mapping, search-and-rescue location problems — any scenario where a physical sensor must be positioned to detect a spatially localised signal.
2. **Limitations from synthetic nature:** The synthetic function may have a sharper peak than real radiation fields, which typically spread over a broader area. Real deployments would also have sensor noise and safety constraints that limit where measurements can be taken.
3. **Scalability:** The GP-UCB approach would scale poorly to 3D or higher-dimensional radiation source problems without much larger observation budgets. The boundary diagnostic strategy would remain valuable regardless of dimensionality.
4. **Risks/pitfalls:** The main risk is concluding that the centre region is the global maximum when it may be the best of a poorly explored space. Any user of this function should treat the result as a lower bound on the true optimum.

---

# Function 2 — Noisy ML Model Scoring (2D)

## Function overview

1. **Function:** F2 — Noisy ML Model Scoring
2. **Real-world scenario:** Evaluating a machine learning model's performance score across a 2D hyperparameter space. The output carries genuine stochasticity — the same input can return different outputs on different evaluations.
3. **Dimensionality:** 2D — [x1, x2], bounded in [0, 1]
4. **Initial data points:** 10 observations. Shape: inputs (10, 2), outputs (10,)
5. **Output represents:** Model performance score (higher = better)

## Nature of the data

1. **Initial dataset structure:** 10 input vectors of shape (10, 2). Outputs showed variation across the range 0.1 to 0.4 in baseline observations.
2. **Dataset evolution:** 13 new observations added. Queries clustered around [0.675, 0.936] from Round 6 onwards — the confirmed best region. The dataset reflects an exploitation bias in later rounds.
3. **Noise/randomness:** Confirmed stochastic. The same coordinate region returned 0.6202 in Round 6 and 0.467 in Round 12. This is genuine noise — the function itself carries randomness, not just measurement error.
4. **Landscape type:** Noisy with a likely narrow peak around [0.675, 0.936]. The GP smoothness assumption is violated here — the function cannot be reliably exploited because repeated queries return different values.

## Optimisation strategy

1. **Methods used:** GP + UCB (beta=2.0), SVM filter, NN gradient. Beta kept moderate — neither fully exploring nor fully exploiting given the stochastic nature.
2. **Why this method:** GP-UCB handles uncertainty through its std term, which naturally accounts for noisy landscapes. The NN gradient was less reliable here due to stochasticity.
3. **Exploration-exploitation balance:** Moderate beta (1.5–2.0) throughout. The stochastic nature meant tight exploitation was risky — a good result could be a lucky draw rather than a true peak.
4. **Strategy changes:** Beta adjusted from 2.58 (R1–3) down to 1.5 (R6–7) when the peak appeared, then back up to 2.0 when it became clear the peak was noisy and could not be reliably reproduced.

## Data handling and preprocessing

1. **Rescaling:** No input rescaling needed. `normalize_y=True` on GP.
2. **Surrogate model:** GP with Matern 2.5 kernel. Limited reliability due to stochasticity.
3. **Preprocessing:** Standard. Alpha parameter (nugget) in GP helps absorb noise in the likelihood.
4. **Outliers:** The Round 6 result of 0.6202 could be treated as a lucky draw rather than a true peak value. It was preserved as the best-known result but treated with appropriate scepticism in subsequent rounds.

## Weekly iteration and learning

1. **How new data changed understanding:** Each round reinforced the conclusion that F2 is genuinely noisy. The inability to reproduce the Round 6 peak despite returning to the same coordinate region confirmed stochasticity.
2. **Local optima:** Uncertain — the apparent peak at [0.675, 0.936] may be a noise event. No way to confirm from 23 total observations.
3. **Most informative inputs:** Round 6 query was most informative — it established the best-known region and revealed the noise structure when later rounds failed to reproduce it.
4. **What to do differently:** Submit the same coordinates twice in early rounds to detect stochasticity explicitly. This would have changed the strategy from Round 3 rather than Round 8.

## Performance and results

1. **Best output:** 0.6202 (Round 6)
2. **Best input vector:** [0.675, 0.936]
3. **Confidence in global maximum:** Low. The best result is likely partly a noise draw. Confidence that the true expected value near [0.675, 0.936] is the global optimum is moderate — it is the most consistently high-performing region observed.
4. **Aligned with expectations:** Partially — a noisy ML scoring landscape is realistic, and the difficulty of reproducing peaks under stochasticity is a known challenge in hyperparameter optimisation.

## Ethical, practical and general considerations

1. **Real-world applications:** Hyperparameter tuning for production ML models where each evaluation involves training and validation — a genuinely expensive and noisy process.
2. **Limitations:** The synthetic noise may not match real ML evaluation noise distributions. Real model training has structured randomness (seed-dependence) that can be partially controlled.
3. **Scalability:** The strategy would not scale well to higher-dimensional noisy functions without a much larger observation budget and explicit noise modelling (e.g., GP with a noise kernel).
4. **Risks/pitfalls:** Overcommitting to a single observed peak in a noisy function. Any future user should explicitly test whether results are reproducible before deploying a configuration.

---

# Function 3 — Drug Compound Optimisation (3D)

## Function overview

1. **Function:** F3 — Drug Compound Optimisation
2. **Real-world scenario:** Finding the combination of three compound concentrations that minimises adverse reactions in drug discovery. The output is negative by design — the goal is to maximise (find the least negative value).
3. **Dimensionality:** 3D — [x1, x2, x3], bounded in [0, 1]
4. **Initial data points:** 10 observations. Shape: inputs (10, 3), outputs (10,)
5. **Output represents:** Negative adverse reaction severity — higher (less negative) is better

## Nature of the data

1. **Initial dataset structure:** 10 input vectors of shape (10, 3). All baseline outputs negative, ranging approximately -0.12 to -0.03.
2. **Dataset evolution:** 13 new observations. Early rounds concentrated around [0.500, 0.300, 0.450] after Round 3 found the best result. Later rounds were unable to return to this region reliably.
3. **Noise/randomness:** Low apparent noise but a fragile peak. Deviations of 0.01–0.02 in any dimension caused large output drops, suggesting a very sharp peak rather than genuine stochasticity.
4. **Landscape type:** Appears to have a spike-like or very narrow peak. Smooth in most of the space (consistently negative), with one sharp local maximum in the [0.500, 0.300, 0.450] region that was found in Round 3 and never recovered.

## Optimisation strategy

1. **Methods used:** GP + UCB (beta=2.0–2.5), SVM filter, NN gradient.
2. **Why this method:** GP-UCB handles smooth landscapes well. However, the narrow peak violated the smoothness assumption — the GP modelled the region as smoother than it actually is.
3. **Exploration-exploitation balance:** Moderate exploration (beta=2.0) throughout. Tightened to beta=1.5 in Round 13 when the region near Round 12's result was the final target.
4. **Strategy changes:** Tried perturbation at different step sizes when the Round 3 peak could not be recovered. Final rounds returned to the vicinity of the confirmed best coordinates. Round 12 perturbation produced -0.017 — the best late result.

## Data handling and preprocessing

1. **Rescaling:** No input rescaling. `normalize_y=True` on GP handles the negative output range.
2. **Surrogate model:** GP with Matern 2.5 kernel.
3. **Preprocessing:** Standard. No special handling needed for negative outputs — the GP normalises internally.
4. **Outliers:** Round 3 result of -0.0045 is significantly better than all subsequent observations. It was preserved as the all-time best but treated as potentially reflecting a narrow, hard-to-reproduce peak.

## Weekly iteration and learning

1. **How new data changed understanding:** Early data suggested a smooth landscape with one good region. Later data revealed the peak is extremely sensitive to small input changes — more spike-like than smooth.
2. **Local optima:** Round 3 peak at -0.0045 may itself be a local maximum in a multimodal landscape. With 23 observations in 3D, the global structure remains largely unknown.
3. **Most informative inputs:** Round 3 query was the most informative — it found the best region and revealed the landscape structure. Round 12 perturbation at [0.504, 0.296, 0.454] was the most informative late query, returning -0.017.
4. **What to do differently:** Use smaller step sizes from Round 4 onwards. The gradient nudge step of 0.010 was too large for a function with a fragile, narrow peak.

## Performance and results

1. **Best output:** -0.0045 (Round 3)
2. **Best input vector:** [0.500, 0.300, 0.450]
3. **Confidence in global maximum:** Low. Round 3 found one good point but 13 subsequent rounds could not improve on it. The true global maximum may sit in a region never sampled.
4. **Aligned with expectations:** Partially — drug compound landscapes are known to have narrow peaks (structure-activity relationships). The difficulty of recovering the Round 3 peak is consistent with the steep drop-offs typical in compound optimisation problems.

## Ethical, practical and general considerations

1. **Real-world applications:** Pharmaceutical lead optimisation, where each experiment is expensive and the space of possible compounds is vast. The limited observation budget mirrors real drug discovery constraints.
2. **Limitations:** Real compound landscapes have many more dimensions (molecular descriptors, dosage, formulation) and cannot be bounded to [0,1]^3.
3. **Scalability:** The strategy would not scale to realistic drug discovery dimensionality without domain-specific surrogate models and active learning frameworks with much larger budgets.
4. **Risks/pitfalls:** Concluding that -0.0045 is near the global optimum. In real drug discovery, this could mean terminating search too early and missing significantly better compounds.

---

# Function 4 — Warehouse Logistics Optimisation (4D)

## Function overview

1. **Function:** F4 — Warehouse Logistics Optimisation
2. **Real-world scenario:** Tuning four hyperparameters of a warehouse placement model. The output represents a logistics performance score — higher is better.
3. **Dimensionality:** 4D — [x1, x2, x3, x4], bounded in [0, 1]
4. **Initial data points:** 10 observations. Shape: inputs (10, 4), outputs (10,)
5. **Output represents:** Logistics performance score of the warehouse placement model

## Nature of the data

1. **Initial dataset structure:** 10 input vectors of shape (10, 4). Baseline outputs ranged from approximately -0.24 to +0.12. Started in negative territory.
2. **Dataset evolution:** 13 new observations. Later queries concentrated around [0.360, 0.410, 0.430, 0.395] — the confirmed best coordinates — reproduced identically in Rounds 7, 9, 11, 12 and 13.
3. **Noise/randomness:** Low noise confirmed — the same coordinates returned 0.5534 in Rounds 7, 9, 11, 12 and 13. This reproducibility distinguishes F4 from F2 and suggests the peak is a real, stable feature.
4. **Landscape type:** Narrow but real unimodal-like peak. Deviations of 0.01–0.02 in any dimension caused notable output drops (e.g., Round 8 returned 0.361 after a small nudge from the Round 7 best). The function may be dynamic — description mentions a placement model whose landscape could shift.

## Optimisation strategy

1. **Methods used:** GP + UCB (beta=1.5), SVM filter, NN gradient. Once the peak was found in Round 7, the strategy shifted to exact coordinate reproduction.
2. **Why this method:** GP-UCB found the general region. Return-to-best logic (a key RL-inspired heuristic) prevented further deviation once the peak was confirmed.
3. **Exploration-exploitation balance:** Beta=1.5 throughout — moderate exploitation. After Round 7, the confirmed best was returned to exactly when any nudge caused regression.
4. **Strategy changes:** Major pivot after Round 8 — a nudge from the Round 7 best caused regression to 0.361. Decision: stop nudging and return to exact coordinates. This produced 0.553 again in Round 9. The lesson was applied consistently for the remaining 5 rounds.

## Data handling and preprocessing

1. **Rescaling:** No input rescaling. `normalize_y=True` on GP.
2. **Surrogate model:** GP with Matern 2.5 kernel. With 23 observations in 4D, the surrogate provided useful directional guidance in early rounds but was not trusted for refinement once the peak was confirmed.
3. **Preprocessing:** Standard. No outlier handling required — outputs were consistent and reproducible.
4. **Outliers:** No meaningful outliers. Round 1 returned -0.2441 which was the lowest value — driven by the initial random sample, not an error.

## Weekly iteration and learning

1. **How new data changed understanding:** Rounds 4–6 revealed a positive region was reachable. Round 7 found the peak. Round 8's regression after a small nudge was the most important learning — it confirmed the peak is narrow and requires exact coordinates, not local refinement.
2. **Local optima:** Possible. The confirmed peak at 0.5534 may be a local maximum. With 23 observations in 4D, the global structure is largely unknown. However, the SVM filter and GP-UCB consistently directed queries to the same region, which is consistent evidence for a single dominant peak.
3. **Most informative inputs:** Round 7 query — found the peak. Round 8 query — confirmed via regression that the peak is narrow and fragile. Round 9 — confirmed reproducibility by returning exact coordinates and matching Round 7 output exactly.
4. **What to do differently:** Start with a broader sweep across all four dimensions in early rounds before committing to any region. The initial random sample happened to contain good coordinates — a different random seed might have taken several more rounds to find the same region.

## Performance and results

1. **Best output:** 0.5533948144101939
2. **Best input vector:** [0.360, 0.410, 0.430, 0.395]
3. **Confidence in global maximum:** Moderate. Four independent reproductions at the same coordinates give high confidence this is a local maximum. Whether it is the global maximum is unknown — the 4D space is only partially explored.
4. **Aligned with expectations:** Yes — warehouse logistics models typically have one or a small number of good configuration regions, consistent with the narrow but reproducible peak observed.

## Ethical, practical and general considerations

1. **Real-world applications:** Automated warehouse management, supply chain optimisation, robot routing configurations — all involve expensive evaluation of placement policies where each trial has real operational cost.
2. **Limitations:** Real warehouse optimisation involves many more parameters, stochastic demand, and time-varying conditions. The synthetic function captures the core optimisation challenge but not the full complexity.
3. **Scalability:** The return-to-best logic and exact-coordinate reproducibility check would be directly applicable to real warehouse systems. The GP surrogate would need extension for higher-dimensional configuration spaces.
4. **Risks/pitfalls:** Treating the confirmed peak as globally optimal and deploying without further validation. In a real system, the operational context may shift (demand patterns change, new product lines), making periodic re-optimisation necessary.

---

# Function 5 — Chemical Process Yield (4D)

## Function overview

1. **Function:** F5 — Chemical Process Yield Optimisation
2. **Real-world scenario:** Optimising four process parameters (temperature, pressure, catalyst concentration, reaction time) to maximise yield. A unimodal function with one clear peak.
3. **Dimensionality:** 4D — [x1, x2, x3, x4], bounded in [0, 1]
4. **Initial data points:** 10 observations. Shape: inputs (10, 4), outputs (10,)
5. **Output represents:** Chemical process yield (higher = more product per unit input)

## Nature of the data

1. **Initial dataset structure:** 10 input vectors of shape (10, 4). Baseline outputs in the range ~289–800, indicating a clear and large signal from the start.
2. **Dataset evolution:** 13 new observations. All later queries concentrated along the x1 gradient direction, with x1 increasing from ~0.23 (early rounds) to 0.610 (Round 13). Strong exploitation bias in later rounds — the function consistently rewarded it.
3. **Noise/randomness:** No meaningful noise. The function returned consistent values for similar inputs. The monotone x1 gradient was reproducible and exploitable across all 13 rounds.
4. **Landscape type:** Unimodal with a boundary-type maximum — the peak appears to be at or near the x1=1.0 boundary. The GP surrogate modelled this well, and the NN gradient consistently pointed in the x1-increasing direction.

## Optimisation strategy

1. **Methods used:** GP + UCB (beta=0.8), SVM filter, NN gradient (step=0.030–0.040). Lowest beta of any function — strong exploitation justified by consistent gradient.
2. **Why this method:** Clear gradient signal from Round 4 onwards meant the GP surrogate was reliable and the NN gradient pointed consistently in the same direction. Pure exploitation was the right approach.
3. **Exploration-exploitation balance:** Predominantly exploitation from Round 4. Beta held at 0.8 — lowest in the project. Step size increased each round (0.030 → 0.035 → 0.040) as confidence in the gradient grew.
4. **Strategy changes:** None after Round 4. The x1 gradient was established in Round 4 and followed uninterrupted through Round 13 — ten consecutive improvements. PCA analysis of top-quartile observations confirmed x1 as the dominant dimension, reinforcing the strategy.

## Data handling and preprocessing

1. **Rescaling:** No input rescaling. `normalize_y=True` on GP essential — F5 outputs are in thousands while other functions are near zero or negative.
2. **Surrogate model:** GP with Matern 2.5 kernel. Very reliable for this function — the smooth, unimodal landscape matched the kernel's assumptions perfectly.
3. **Preprocessing:** Lower NN weight decay (1e-4 vs 1e-3 for other functions) — the large output range and clear structure meant the network could fit more tightly without overfitting.
4. **Outliers:** No outliers. The monotone trajectory from ~800 (R1) to 3166 (R13) is consistent and smooth.

## Weekly iteration and learning

1. **How new data changed understanding:** Each round confirmed the x1 gradient. By Round 6, the direction was beyond doubt. Rounds 7–13 were refinements of the same confirmed strategy.
2. **Local optima:** None detected. The function appears to have a single basin of attraction leading toward the boundary. No round produced a regression when following the gradient.
3. **Most informative inputs:** Round 4 — first query that produced a significant improvement by following the GP-UCB recommendation toward higher x1. This established the gradient direction that drove all subsequent rounds.
4. **What to do differently:** Submit a boundary point [0.9, 0.9, 0.9, 0.9] or [1.0, 1.0, 1.0, 1.0] in Round 1 or 2. This would have revealed the boundary-type landscape immediately and saved several rounds of gradient-following.

## Performance and results

1. **Best output:** 3166.0084438960635 (Round 13)
2. **Best input vector:** [0.610, 0.938, 0.960, 0.955]
3. **Confidence in global maximum:** Moderate-high for a local maximum; uncertain for global. The function was still rising at Round 13 — the true peak may sit beyond x1=0.610, possibly at the x1=1.0 boundary. The result is the best achievable within the 13-round budget, not necessarily the true global maximum.
4. **Aligned with expectations:** Yes — chemical process yields often have clear operating windows with a dominant parameter (here, x1 analogous to temperature or catalyst loading) driving most of the variation.

## Ethical, practical and general considerations

1. **Real-world applications:** Chemical manufacturing optimisation, pharmaceutical synthesis, materials processing — all involve tuning process parameters within safety constraints to maximise output.
2. **Limitations:** Real chemical processes have non-stationarity (catalyst deactivation, equipment degradation), safety constraints, and multi-objective trade-offs (yield vs purity vs cost) not captured here.
3. **Scalability:** The gradient-following strategy scales well to higher-dimensional smooth functions. The GP surrogate would remain useful with larger observation budgets.
4. **Risks/pitfalls:** The function is still rising at Round 13. Any deployment of the best-found configuration should include safety margin testing near the boundary, as extreme parameter values can cause unexpected behaviour in real processes.

---

# Function 6 — Recipe Scoring (5D)

## Function overview

1. **Function:** F6 — Recipe Scoring Optimisation
2. **Real-world scenario:** Optimising five ingredient amounts in a recipe to maximise a quality score. The score is negative by design across the observed domain — the goal is to maximise (find the least negative value).
3. **Dimensionality:** 5D — [x1, x2, x3, x4, x5], bounded in [0, 1]
4. **Initial data points:** 10 observations. Shape: inputs (10, 5), outputs (10,)
5. **Output represents:** Recipe quality score (negative — higher/less negative is better)

## Nature of the data

1. **Initial dataset structure:** 10 input vectors of shape (10, 5). Baseline outputs all negative, ranging approximately -1.1 to -0.3.
2. **Dataset evolution:** 13 new observations. Queries were spread more widely than other functions due to high exploration beta. No single region proved consistently better — the best result (-0.2957, Round 3) was not recovered in any subsequent round.
3. **Noise/randomness:** Possible. Round 13 returned -0.683 at a region that returned -0.779 in Round 12 — different outputs from nearby coordinates. Could be noise or steep landscape variation.
4. **Landscape type:** Multimodal or highly deceptive. The Round 3 best at [0.350, 0.200, 0.500, 0.500, 0.200] was never recovered. The function appears to have multiple local peaks scattered across the 5D space, making systematic search difficult with 23 total observations.

## Optimisation strategy

1. **Methods used:** GP + UCB (beta=2.5–3.0), SVM filter, NN gradient. Highest beta of any function — broad exploration maintained throughout.
2. **Why this method:** No reliable gradient was ever found. High beta forces the GP-UCB to explore uncertain regions rather than exploit a region that may not be the true peak.
3. **Exploration-exploitation balance:** Predominantly exploration. Beta held at 2.5–3.0. Wide candidate pools. The SVM filter was less useful here — with outputs spread widely across the 5D space, the top 30% threshold captured many different regions.
4. **Strategy changes:** Multiple adjustments as different perturbations were tried. No strategy consistently improved the function. Final rounds settled on corrections toward the Round 3 peak region, producing marginal improvement (-0.683 in Round 13 vs -0.779 in Round 12).

## Data handling and preprocessing

1. **Rescaling:** No input rescaling. `normalize_y=True` on GP handles the negative output range.
2. **Surrogate model:** GP with Matern 2.5 kernel. Limited reliability — the multimodal landscape was poorly represented by a unimodal-biased surrogate with only 23 observations.
3. **Preprocessing:** Standard. No special handling.
4. **Outliers:** Round 3 result of -0.2957 was the clear best — approximately 3x better than the next-best observed value. Treated as a real peak rather than an outlier, but impossible to confirm with subsequent observations failing to reproduce it.

## Weekly iteration and learning

1. **How new data changed understanding:** Each round reinforced the difficulty of the function. No consistent direction emerged. The Round 3 peak remained isolated — no nearby query ever returned a comparable value.
2. **Local optima:** Multiple suspected. The function appears to have at least two distinct local peaks: the Round 3 region and a different region around [0.315, 0.132, 0.398, 0.635, 0.132] that returned moderately better values in later rounds.
3. **Most informative inputs:** Round 3 query — found the best region. All subsequent queries were informative mainly in the negative sense, confirming the landscape was not exploitable with the available budget.
4. **What to do differently:** Apply a systematic Latin Hypercube Sampling or Sobol sequence in the first 4–5 rounds to get better coverage of the 5D space before committing any surrogate-guided search.

## Performance and results

1. **Best output:** -0.2957 (Round 3)
2. **Best input vector:** [0.350, 0.200, 0.500, 0.500, 0.200]
3. **Confidence in global maximum:** Very low. The best result was found in Round 3 and never reproduced. The 5D space is severely undersampled with 23 observations. The true global maximum is almost certainly in a region never queried.
4. **Aligned with expectations:** Recipe scoring is known to involve complex ingredient interactions that create multimodal landscapes. The difficulty of this function is consistent with real culinary or formulation optimisation problems.

## Ethical, practical and general considerations

1. **Real-world applications:** Food formulation, cosmetics ingredient optimisation, pharmaceutical excipient selection — all involve complex ingredient interactions in high-dimensional spaces with expensive sensory or clinical evaluation.
2. **Limitations:** Real recipe scoring involves human sensory evaluation, which introduces significant noise and subjectivity not captured by this synthetic function.
3. **Scalability:** The strategy does not scale. With 5 dimensions and 23 observations, coverage is already inadequate. Real formulation problems with 10+ ingredients would require hundreds of observations and more sophisticated surrogate models.
4. **Risks/pitfalls:** Any conclusion about the optimal recipe from 23 observations in 5D is highly uncertain. Real deployment should include extensive validation around the best-found configuration before any production use.

---

# Function 7 — ML Hyperparameter Tuning (6D)

## Function overview

1. **Function:** F7 — ML Hyperparameter Tuning
2. **Real-world scenario:** Tuning six hyperparameters of a machine learning model to maximise performance. Parameters might include learning rate, regularisation strength, dropout, batch size, layer width, and momentum.
3. **Dimensionality:** 6D — [x1, x2, x3, x4, x5, x6], bounded in [0, 1]
4. **Initial data points:** 10 observations. Shape: inputs (10, 6), outputs (10,)
5. **Output represents:** ML model performance score (higher = better)

## Nature of the data

1. **Initial dataset structure:** 10 input vectors of shape (10, 6). Baseline outputs ranged approximately 0.15–0.75.
2. **Dataset evolution:** 13 new observations. From Round 7 onwards, queries concentrated around [0.050, 0.250, 0.240, 0.230, 0.430, 0.770] — the confirmed best region. Reproduced identically in Rounds 7, 12 and 13.
3. **Noise/randomness:** Low noise. The Round 7 best of 1.8116 was reproduced at the same coordinates in Rounds 12 and 13. Some variation in nearby coordinates (Round 11 returned 1.811 at slightly different inputs) but generally stable.
4. **Landscape type:** Smooth with a clear but localised peak. PCA analysis of top-quartile observations identified x1 (low) and x6 (high) as the dominant dimensions — consistent with the confirmed best coordinates [0.050, ..., 0.770].

## Optimisation strategy

1. **Methods used:** GP + UCB (beta=1.0), SVM filter, NN gradient. Medium-low beta — exploit once the direction was confirmed in Round 6.
2. **Why this method:** The GP identified the general region early. Round 6 established a clear gradient direction (x1 decreasing, x6 increasing). Beta reduced to 1.0 from Round 5 onwards to exploit this signal.
3. **Exploration-exploitation balance:** Exploration in Rounds 1–5, exploitation from Round 6. Return-to-best logic applied when Round 11 produced a slight dip — Rounds 12 and 13 returned to exact Round 7 coordinates.
4. **Strategy changes:** Beta reduced from 2.58 to 1.0 after Round 5. Step size reduced from 0.010 to 0.003–0.005 in final rounds to avoid deviating from confirmed coordinates.

## Data handling and preprocessing

1. **Rescaling:** No input rescaling. `normalize_y=True` on GP.
2. **Surrogate model:** GP with Matern 2.5 kernel. Performed well — the smooth landscape was a good match for the kernel's assumptions.
3. **Preprocessing:** Standard. PCA applied analytically (not in the pipeline) to identify dominant dimensions. This informed the gradient direction without changing the core pipeline.
4. **Outliers:** No meaningful outliers. Round 1 returned 0.747 — higher than expected for a random baseline, suggesting the initial sample happened to land in a reasonable region.

## Weekly iteration and learning

1. **How new data changed understanding:** Round 6 was the turning point — a query that returned 1.7982 established a clear improvement direction. PCA confirmed x1 and x6 were dominant. Subsequent rounds refined around this finding.
2. **Local optima:** Possible — the confirmed peak at [0.050, 0.250, 0.240, 0.230, 0.430, 0.770] may be a local maximum. With 23 observations in 6D, global coverage is poor. However, three independent reproductions give reasonable confidence in the local structure.
3. **Most informative inputs:** Round 6 — established the gradient direction. Round 7 — confirmed the peak. Rounds 12 and 13 — confirmed reproducibility across three independent rounds.
4. **What to do differently:** Use ARD (Automatic Relevance Determination) kernel in the GP from Round 5 onwards — this would have automatically identified x1 and x6 as dominant dimensions without needing separate PCA analysis.

## Performance and results

1. **Best output:** 1.8116258350443388 (Rounds 7, 12, 13)
2. **Best input vector:** [0.050, 0.250, 0.240, 0.230, 0.430, 0.770]
3. **Confidence in global maximum:** Moderate. Three reproductions at the same coordinates give high confidence this is a stable local maximum. The 6D space is undersampled but the surrogate consistently pointed to this region from Round 6 onwards.
4. **Aligned with expectations:** Yes — ML hyperparameter landscapes are known to have localised peaks where specific combinations of regularisation, architecture, and learning rate work well together. The low x1, high x6 pattern is consistent with low regularisation and high momentum — a plausible configuration.

## Ethical, practical and general considerations

1. **Real-world applications:** Neural architecture search, AutoML, Bayesian hyperparameter optimisation in production ML pipelines — any setting where each model training run is expensive and the hyperparameter space is high-dimensional.
2. **Limitations:** Real hyperparameter optimisation involves training time constraints, computational cost, and dataset-specific performance that are not captured in a synthetic 6D function.
3. **Scalability:** The GP-UCB approach is commonly used for real hyperparameter tuning (GPyOpt, BoTorch, Ax). The per-dimension analysis (ARD) and trust-region strategies developed here are directly applicable.
4. **Risks/pitfalls:** The confirmed best configuration may not generalise across datasets or model architectures. Real hyperparameter tuning should validate the best-found configuration on held-out test sets before deployment.

---

# Function 8 — High-Dimensional Hyperparameter Search (8D)

## Function overview

1. **Function:** F8 — High-Dimensional Hyperparameter Search
2. **Real-world scenario:** Optimising eight hyperparameters simultaneously to find the strongest local maximum. Represents the most complex configuration space in the project.
3. **Dimensionality:** 8D — [x1, x2, x3, x4, x5, x6, x7, x8], bounded in [0, 1]
4. **Initial data points:** 10 observations. Shape: inputs (10, 8), outputs (10,)
5. **Output represents:** System performance score — higher is better, with an apparent ceiling near 9.925

## Nature of the data

1. **Initial dataset structure:** 10 input vectors of shape (10, 8). Baseline outputs already high — ranging approximately 9.4–9.82, suggesting a broadly high-performing landscape where most configurations produce good results.
2. **Dataset evolution:** 13 new observations. Queries clustered tightly around the Round 3 best region from Round 4 onwards. The dataset is heavily concentrated in one small area of the 8D space — the rest is completely unexplored.
3. **Noise/randomness:** Low. Values between 9.920 and 9.925 across multiple rounds at similar coordinates suggest a genuine plateau rather than noise. The all-time best of 9.9250 (Round 3) was not exceeded in 10 subsequent rounds.
4. **Landscape type:** Plateau-like near the ceiling. The function appears to have a broad flat region where most inputs return values near 9.92, with a very narrow peak at 9.9250 that was found in Round 3 and never recovered. With 23 observations in 8D, characterising the landscape fully is impossible.

## Optimisation strategy

1. **Methods used:** GP + UCB (beta=0.5), SVM filter, NN gradient (step=0.002–0.005). Lowest beta and step size of any function — micro-exploitation only.
2. **Why this method:** The function appeared near its ceiling from Round 3. Micro-refinements were the only viable strategy — any large exploratory step risked moving away from the plateau into a genuinely lower region.
3. **Exploration-exploitation balance:** Near-pure exploitation from Round 3. Beta=0.5 — the most aggressive exploitation setting in the project. Step size reduced to 0.002 in the final rounds.
4. **Strategy changes:** No fundamental changes — the micro-exploitation strategy was set in Round 3 and maintained throughout. The NN weight decay was set lower (1e-4) for F8 to allow tighter gradient fitting.

## Data handling and preprocessing

1. **Rescaling:** No input rescaling. `normalize_y=True` on GP — essential for fitting the narrow range of variation (9.920–9.925) relative to other functions.
2. **Surrogate model:** GP with Matern 2.5 kernel. Limited resolution — the variation between 9.920 and 9.925 is subtle, and with only 23 observations in 8D, the surrogate cannot reliably distinguish which micro-adjustments are genuinely better.
3. **Preprocessing:** Standard. Lower NN weight decay to allow the network to model the subtle variation in the plateau region.
4. **Outliers:** The Round 3 all-time best of 9.9250 was achieved early and never exceeded — this appears to be a real peak rather than a noise event, but its precise location was not reliably recoverable in subsequent rounds.

## Weekly iteration and learning

1. **How new data changed understanding:** Early rounds confirmed the function is broadly high-performing. Each round's micro-adjustment returned values within 0.003 of 9.922, confirming a plateau. The failure to exceed Round 3's 9.9250 across 10 subsequent rounds suggests the all-time best may be a narrow feature within the plateau.
2. **Local optima:** Highly likely. The plateau at ~9.922 is almost certainly a local (possibly global) maximum within the sampled region. Whether better regions exist elsewhere in the 8D space is unknown — 23 observations cannot cover it.
3. **Most informative inputs:** Round 3 — found the plateau and established the best-known value. Each subsequent round was informative mainly in confirming the plateau structure and the difficulty of exceeding it.
4. **What to do differently:** Dedicate 3–4 early rounds to broad exploration of the 8D space using Sobol sequences rather than random sampling. This might have revealed whether the plateau extends broadly or whether there are better regions accessible with different initial coordinates.

## Performance and results

1. **Best output:** 9.9250 (Round 3)
2. **Best input vector:** [0.050, 0.195, 0.055, 0.115, 0.875, 0.415, 0.060, 0.465]
3. **Confidence in global maximum:** Low. The plateau at ~9.922 covers a small region of the 8D space. The all-time best (9.9250) was found early and not recovered — it may be a narrow peak within the plateau. With 23 observations in 8D, the global structure is fundamentally unknowable.
4. **Aligned with expectations:** High-dimensional hyperparameter landscapes often have broad plateau regions where many configurations perform similarly — consistent with the observed near-ceiling performance across almost all rounds.

## Ethical, practical and general considerations

1. **Real-world applications:** Large-scale neural network training, distributed system configuration, multi-parameter industrial process control — any setting with 8+ interdependent parameters where evaluation is expensive.
2. **Limitations:** Real 8D optimisation requires much larger observation budgets. The curse of dimensionality means 23 observations in 8D covers an infinitesimally small fraction of the space. Practical hyperparameter optimisation frameworks (Ax, BOHB) would use 100+ observations as a minimum.
3. **Scalability:** The micro-exploitation strategy is transferable but the GP surrogate degrades in 8D without dense observation coverage. Scalable alternatives (sparse GPs, random forests, neural surrogates) would be needed for real 8D problems.
4. **Risks/pitfalls:** The plateau structure means small configuration changes may not matter — but larger structural changes (different architecture, different algorithm family) might produce dramatically different results that are invisible within the plateau. A broader initial sweep is essential before committing to micro-refinement.

---

*Last updated: Round 13 (final round). All observation counts reflect 10 baseline + 13 weekly queries = 23 total per function.*
