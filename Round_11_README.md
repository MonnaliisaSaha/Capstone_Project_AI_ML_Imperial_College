# Round 11 — Final Push Before Extended Rounds

## What this round was about

F5 gets one more x1 push — the gradient has worked every single round since Round 4 and there is no reason to stop. F4 goes back to its confirmed best coordinates exactly. F3 gets a careful perturbation — it produced its best late result in this round. Everything else stays within tight range of the best known value.

---

## Strategy

F5 has improved every round since Round 4. The x1 gradient is the most reliable signal in the project. x1 moved from ~0.520 (R10) to **0.550** this round — one more step along the confirmed direction.

F4 returns to exact confirmed best [0.360, 0.410, 0.430, 0.395]. These coordinates reproduced 0.5534 in Round 7 and again here — confirming the peak is a real, reproducible feature of the landscape.

F7 returns to coordinates near its Round 7 best. The x2-decreasing, x1-low direction established in Round 6 has held throughout.

F3 receives a careful perturbation at [0.502, 0.298, 0.452] — slightly adjusted from prior rounds. This produced its best late result of the project.

F8 gets a micro-adjustment — it has been hovering near 9.922 since Round 3 and only needs tiny refinements at this stage.

F1, F2, F6 are returned to their best historical regions or held steady.

---

## Pipeline at this stage

Full pipeline — GP + UCB + SVM + NN — with per-function configuration. Clustering analysis conducted this round confirmed F4's peak is spatially isolated — no competing high-value cluster exists nearby, which justified returning to exact coordinates rather than probing adjacent points. PCA analysis of F5's top-quartile observations continued to confirm x1 as the dominant dimension.

| Function | Beta | Step | Mode |
|----------|------|------|------|
| F1 | 3.0 | 0.010 | Hold — centre point, no signal found |
| F2 | 2.0 | 0.003 | Moderate — return to best region |
| F3 | 2.0 | 0.010 | Careful perturbation — probe nearby |
| F4 | 1.5 | 0.010 | Exploit — exact confirmed coordinates |
| F5 | 0.8 | 0.030 | Exploit — x1 gradient push |
| F6 | 2.5 | 0.030 | Explore — landscape remains difficult |
| F7 | 1.0 | 0.010 | Exploit — hold near confirmed direction |
| F8 | 0.5 | 0.005 | Exploit — micro-refinement only |

---

## Queries submitted

```
F1 (2D): 0.500000-0.500000
F2 (2D): 0.672000-0.933000
F3 (3D): 0.502000-0.298000-0.452000
F4 (4D): 0.360000-0.410000-0.430000-0.395000
F5 (4D): 0.550000-0.934000-0.956000-0.951000
F6 (5D): 0.310000-0.130000-0.395000-0.640000-0.130000
F7 (6D): 0.050000-0.248000-0.239000-0.229000-0.430000-0.770000
F8 (8D): 0.050000-0.187000-0.055000-0.108000-0.879000-0.414000-0.059000-0.462000
```

---

## Results

| Function | Output | vs Prior Best | Assessment |
|----------|--------|--------------|------------|
| F1 | 2.6752879910742468e-9 | = tied best | Floor — no signal found across 11 rounds |
| F2 | 0.5378238067880806 | below R6 best | Moderate — landscape narrow and sensitive |
| F3 | -0.016493752340423537 | ↑ best late result | Perturbation worked — best since Round 3 |
| F4 | 0.5533948144101939 | = confirmed best | Peak reproduced exactly — third time at same coordinates |
| F5 | 2908.7301569659203 | ↑ new best | Gradient continues — x1 push effective |
| F6 | -0.7940878560149738 | ↓ worsened | Deceptive landscape — difficult throughout |
| F7 | 1.8107461235042484 | near best | Slight dip from R7 peak — near confirmed region |
| F8 | 9.9228091 | ≈ same | Near ceiling — stable |

---

## What the results showed

F5 at 2908 is a new all-time best — the x1 gradient has now produced improvement in eight consecutive rounds (R4–R11). The trajectory is unbroken and the function is still rising. x1 will be pushed further in Round 12.

F4 at 0.5534 was reproduced for the third time at identical coordinates (Rounds 7, 9, 11). Three independent reproductions at the same point confirm this is a real, stable peak and not a noise event. No deviation needed — return to exact coordinates again in Round 12.

F3 at -0.016 was a positive result — its best late-project output. The perturbation at [0.502, 0.298, 0.452] hit a slightly better local region than prior rounds. Round 12 will adjust slightly from this position.

F7 at 1.811 is just below its Round 7 best of 1.8116. The coordinates [0.050, 0.248, 0.239, 0.229, 0.430, 0.770] are very close to the confirmed best. Round 12 will return to the exact R7 coordinates.

F6 worsened again at [0.310, 0.130, 0.395, 0.640, 0.130] — this function has been the hardest throughout. The landscape is multimodal and the surrogate has not found a reliable direction. Round 12 will try a small adjustment.

F8 at 9.9228 — hovering near 9.922 as it has since Round 3. The local maximum appears confirmed. Micro-refinements only for the remaining rounds.

F2 at 0.538 — below the Round 6 best of 0.6202. The landscape near [0.672, 0.933] is sensitive and the output varies with small input changes. Round 12 will adjust slightly toward the confirmed R6 region.

F1 at 2.68e-9 — unchanged from the best value seen in Rounds 5 and 7. The centre point remains the best observed despite 11 rounds of queries. No further signal has been found.

---

## Next step

Round 12 — penultimate push. F5 x1 to 0.580. F4 and F7 held at confirmed coordinates. F3 adjusted from this round's successful region. F6 and F2 get final perturbation attempts.

---

*Full script with inline explanations: `Round_11.py`*
