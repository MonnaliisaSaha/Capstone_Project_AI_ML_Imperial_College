# Round 13 — Final Round

## What this round was about

Last queries. Lock in the best achievable results. F5 gets one final x1 push. F4 and F7 return to exact confirmed best coordinates. F2 returns to its all-time best region. F3 stays close to Round 12's successful perturbation. F6 corrects slightly toward its Round 3 peak region. F8 micro-refinement only.

---

## Strategy

F5 has improved every single round since Round 4. The x1 gradient has been the most reliable signal in the entire project. x1 pushed from 0.580 (R12) to **0.610** for the final query — one more step along the confirmed direction.

F4 goes back to exact confirmed best [0.360, 0.410, 0.430, 0.395]. These coordinates have reproduced 0.5534 in Rounds 7, 9 and 12. No deviation.

F7 goes back to [0.050, 0.250, 0.240, 0.230, 0.430, 0.770]. Same logic — coordinates reproduced across multiple rounds.

F2 returns to [0.675, 0.936] — the Round 6 all-time best. Round 12's probe at [0.678, 0.939] dipped. Best evidence points back to the confirmed region.

F3: Round 12 gave -0.017 at [0.504, 0.296, 0.454]. This round steps slightly back to [0.502, 0.294, 0.452] — minor correction toward the Round 3 best region.

F6 corrects to [0.315, 0.132, 0.398, 0.635, 0.132] — a small step back toward the Round 3 best region.

F1 remains at [0.500, 0.500]. No signal found in 12 rounds — centre point remains the best observed despite near-zero output throughout.

F8 micro-adjusted to [0.050, 0.189, 0.055, 0.107, 0.877, 0.413, 0.058, 0.461] — a tiny move from Round 12 to probe the local neighbourhood at near-ceiling performance.

---

## Pipeline at this stage

Full pipeline — GP + UCB + SVM + NN — in near-pure exploitation mode for all functions. No exploratory moves in the final round.

| Function | Beta | Step | Mode |
|----------|------|------|------|
| F1 | 2.5 | 0.010 | Hold — centre point, no signal |
| F2 | 1.5 | 0.002 | Exploit — return to confirmed best |
| F3 | 1.5 | 0.005 | Exploit — tight correction |
| F4 | 1.5 | 0.005 | Exploit — exact confirmed coordinates |
| F5 | 0.8 | 0.040 | Exploit — final x1 push |
| F6 | 2.0 | 0.020 | Moderate — correction toward R3 peak |
| F7 | 1.0 | 0.003 | Exploit — exact confirmed coordinates |
| F8 | 0.5 | 0.002 | Exploit — micro-refinement only |

---

## Queries submitted

```
F1 (2D): 0.500000-0.500000
F2 (2D): 0.675000-0.936000
F3 (3D): 0.502000-0.294000-0.452000
F4 (4D): 0.360000-0.410000-0.430000-0.395000
F5 (4D): 0.610000-0.938000-0.960000-0.955000
F6 (5D): 0.315000-0.132000-0.398000-0.635000-0.132000
F7 (6D): 0.050000-0.250000-0.240000-0.230000-0.430000-0.770000
F8 (8D): 0.050000-0.189000-0.055000-0.107000-0.877000-0.413000-0.058000-0.461000
```

---

## Results

| Function | Output | vs All-time Best | Assessment |
|----------|--------|-----------------|------------|
| F1 | 2.6752879910742468e-9 | = tied best | Floor — confirmed across 13 rounds |
| F2 | 0.5123831787687212 | below R6 best | Landscape narrow — R6 peak not fully recovered |
| F3 | -0.0228833252266363 | below R12 | Slight dip — fragile landscape |
| F4 | 0.5533948144101939 | = confirmed best | Peak reproduced — real feature confirmed |
| F5 | **3166.0084438960635** | new all-time best | Gradient continues — strongest result of project |
| F6 | -0.6825685672384274 | improvement vs R12 | Small recovery — correction moved in right direction |
| F7 | 1.8116258350443388 | = confirmed best | Peak reproduced — confirmed stable |
| F8 | 9.9219614 | near ceiling | Marginal gain — near maximum reachable value |

---

## What the results showed

F5 at 3166 is the standout final result. The x1 gradient produced improvement in every round from Round 4 to Round 13 — ten consecutive improvements. Starting from ~800 in Round 1 to 3166 in Round 13 is a ~4x increase, and the function was still rising at the boundary.

F4 at 0.5534 was reproduced exactly for the fourth time at identical coordinates (Rounds 7, 9, 12, 13). This is the most reproducible result in the project — a narrow but real peak confirmed beyond doubt.

F7 at 1.8116 similarly confirmed at the same coordinates across multiple rounds. The Round 6 direction (x2 decreasing, x1 low) was correct and held to the end.

F8 at 9.922 reached what is effectively its ceiling. Every round since Round 3 has returned values between 9.920 and 9.925. The local maximum is confirmed.

F2 at 0.512 did not recover the Round 6 best of 0.6202. The landscape here is genuinely noisy — the same coordinate region returns different outputs across rounds.

F3 dipped slightly from Round 12 to -0.023. The Round 3 best of -0.0045 was never recovered. The peak appears extremely fragile.

F6 improved to -0.683 from -0.779 in Round 12 — the correction moved in the right direction but the Round 3 best remains the highest point ever found.

F1 at 2.68e-9 unchanged. After 13 rounds, no signal was ever found beyond what the centre region provided.

---

## Project complete

13 rounds. 8 functions. Full pipeline: GP → UCB → SVM → NN.

See MODEL_CARD.md for final performance summary and DATASHEET.md for full dataset documentation.

---

*Full script with inline explanations: `Round_13.py`*
