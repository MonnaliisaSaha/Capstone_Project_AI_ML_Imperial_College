# Round 12 — Penultimate Push

## What this round was about

Two rounds left. The strategy here was to push F5 one step further along its established gradient, hold the confirmed peaks for F4, F7 and F8, and make one last adjustment attempt on F2, F3 and F6 before the final round locks everything in.

---

## Strategy

F5 has improved every round since Round 4. The x1 gradient is clear — each push toward the boundary produces a higher output. x1 moved from 0.520 (R10) to 0.550 (R11) to **0.580** this round.

F4 and F7 have both been confirmed at their peak coordinates across multiple rounds. No deviation — return exactly to the best known point.

F8 is hovering near its ceiling at 9.922. Micro-refinement only — step of 0.001 maximum.

F2 is noisy but the R6 best at [0.675, 0.936] is the target region. Adjusted slightly to [0.678, 0.939] to probe adjacent territory.

F3 and F6 remain negative. Small perturbations away from prior positions — if they don't improve this round, R13 will return to their best historical coordinates.

F1 has never produced meaningful signal. Held at [0.500, 0.500] — the centre point that returned the highest value (2.68e-9) in Rounds 5 and 7.

---

## Pipeline at this stage

Full pipeline — GP + UCB + SVM + NN — with per-function configuration:

| Function | Beta | Step | Mode |
|----------|------|------|------|
| F1 | 3.0 | 0.010 | Explore — no signal found |
| F2 | 2.0 | 0.003 | Moderate — probe adjacent region |
| F3 | 2.0 | 0.010 | Explore — perturbation attempt |
| F4 | 1.5 | 0.005 | Exploit — return to confirmed peak |
| F5 | 0.8 | 0.035 | Exploit — x1 gradient push |
| F6 | 2.5 | 0.030 | Explore — perturbation attempt |
| F7 | 1.0 | 0.005 | Exploit — hold confirmed coordinates |
| F8 | 0.5 | 0.003 | Exploit — micro-refinement only |

---

## Queries submitted

```
F1 (2D): 0.500000-0.500000
F2 (2D): 0.678000-0.939000
F3 (3D): 0.504000-0.296000-0.454000
F4 (4D): 0.360000-0.410000-0.430000-0.395000
F5 (4D): 0.580000-0.936000-0.958000-0.953000
F6 (5D): 0.320000-0.135000-0.400000-0.630000-0.135000
F7 (6D): 0.050000-0.250000-0.240000-0.230000-0.430000-0.770000
F8 (8D): 0.050000-0.188000-0.055000-0.107000-0.878000-0.413000-0.058000-0.461000
```

---

## Results

| Function | Output | vs Prior Best | Assessment |
|----------|--------|--------------|------------|
| F1 | 2.6752879910742468e-9 | = same | Floor — no signal found in 12 rounds |
| F2 | 0.4667123767190129 | ↓ below R6 best | Dipped — landscape is narrow and sensitive |
| F3 | -0.017028992981422686 | ↑ improvement | Best result to this point — perturbation worked |
| F4 | 0.5533948144101939 | = confirmed | Peak holds — reproducible at exact coordinates |
| F5 | 3030.585505008791 | ↑ new best | Gradient continues — x1 push working |
| F6 | -0.7786435888166022 | ↓ worsened | Deceptive landscape — adjustment backfired |
| F7 | 1.8116258350443388 | = confirmed | Peak holds at same coordinates |
| F8 | 9.9219609 | ≈ same | Near ceiling — stable |

---

## What the results showed

F5 confirmed the gradient continues — 3030 is a new all-time best and the trajectory since Round 4 is unbroken. x1 will be pushed to 0.610 in the final round.

F4 and F7 reproduced their peaks exactly, confirming these are real features of the landscape and not noise events.

F3 was the positive surprise — -0.017 is its best result since Round 3. The perturbation at [0.504, 0.296, 0.454] hit a slightly better local region. The final round will return close to this point.

F2 dipped to 0.467. The landscape around [0.675–0.678, 0.933–0.939] is narrow and any deviation can lower the output. The final round returns to the confirmed best [0.675, 0.936].

F6 worsened at [0.320, 0.135, 0.400, 0.630, 0.135]. This function has been the hardest throughout — multimodal, slow to respond, and consistently negative. The final round will try a small correction back toward the Round 3 peak region.

F8 held at 9.922 — it has effectively plateaued. Minor coordinate adjustment in the final round but no dramatic change expected.

---

## Next step

Final round. F5 gets one last aggressive push. F4, F7 are locked. F2 returns to confirmed best. F3 stays close to this round's successful point. F6 gets a correction. F8 micro-refinement only.

---

*Full script with inline explanations: `Round_12.py`*
