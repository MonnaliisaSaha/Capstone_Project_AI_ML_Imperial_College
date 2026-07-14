# Results Summary

## All rounds — one row per round

Add new results here as a new row at the bottom. Bold = all-time best for that function.

| Round | F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 |
|-------|-----|-----|-----|-----|------|-----|-----|------|
| R1 | ~0 | 0.1051 | -0.0319 | -0.2441 | ~800 | -1.007 | 0.747 | 9.820 |
| R2 | ~0 | 0.230 | -0.028 | -0.180 | ~900 | -0.900 | 0.950 | 9.870 |
| R3 | 2.8e-9 | 0.410 | **-0.0045** | 0.120 | 1200 | **-0.2957** | 1.200 | **9.9250** |
| R4 | 1.2e-9 | 0.380 | -0.018 | 0.310 | 1450 | -0.650 | 1.400 | 9.880 |
| R5 | **2.68e-9** | 0.490 | -0.022 | 0.280 | 1800 | -0.710 | 1.550 | 9.900 |
| R6 | 3.6e-10 | **0.6202** | -0.016 | 0.499 | 2366.81 | -0.793 | 1.7982 | 9.9238 |
| R7 | 2.675e-9 | 0.537 | -0.022 | **0.5534** | 2512.64 | -0.792 | **1.8116** | 9.9236 |
| R8 | 1.648e-9 | 0.523 | -0.020 | 0.361 | 2583.58 | -0.717 | 1.791 | 9.9222 |
| R9 | 2.675e-9 | 0.441 | -0.028 | 0.553 | 2699.31 | -0.707 | 1.807 | 9.9228 |
| R10 | 2.624e-9 | 0.564 | -0.036 | 0.486 | **2798.79** | -0.698 | 1.804 | 9.923 |
| R11 | 2.68e-9 | 0.538 | -0.016 | 0.5534 | 2908.73 | -0.794 | 1.811 | 9.9228 |
| R12 | 2.675e-9 | 0.467 | -0.017 | 0.5534 | 3030.59 | -0.779 | 1.8116 | 9.9220 |
| R13 | 2.675e-9 | 0.512 | -0.023 | 0.5534 | **3166.01** | -0.683 | 1.8116 | 9.9220 |
---

## Confirmed best coordinates

| Function | Best output | Coordinates |
|----------|------------|-------------|
| F1 | 2.68e-9 | [0.500, 0.500] |
| F2 | 0.6202 | [0.675, 0.936] |
| F3 | -0.0045 | [0.500, 0.300, 0.450] |
| F4 | +0.5534 | [0.360, 0.410, 0.430, 0.395] |
| F5 | 2798.79 | [0.520, 0.932, 0.954, 0.949] |
| F6 | -0.2957 | [0.350, 0.200, 0.500, 0.500, 0.200] |
| F7 | 1.8116 | [0.050, 0.250, 0.240, 0.230, 0.430, 0.770] |
| F8 | 9.9250 | [0.050, 0.195, 0.055, 0.115, 0.875, 0.415, 0.060, 0.465] |

---

## Function character notes

**F1** — Near-zero throughout all 13 rounds. The centre point [0.5, 0.5] consistently returned the highest value but improvement was near-zero since Round 5. Likely a very narrow spike in a region never sampled — a boundary diagnostic in Rounds 1–2 may have found it.

**F2** — Genuinely stochastic. Best achieved in Round 6 at 0.6202. Later rounds probing the same region returned lower values, confirming some random noise component. The landscape is narrow and sensitive to small input changes.

**F3** — Fragile peak. Best in Round 3 at -0.0045, never recovered. Best late result was -0.017 in Round 12. Small deviations cause disproportionate drops.

**F4** — Narrow but real. Peak at 0.5534 reproduced identically in Rounds 7, 9, 12 and 13 at the same coordinates. Most reproducible result in the project — confirmed real feature, not noise.

**F5** — Most reliable function. Clear monotone gradient on x1. Improved in every single round from Round 4 to Round 13. Final best: 3166 — approximately 4x the Round 1 baseline.

**F6** — Multimodal and slow. Best in Round 3 at -0.2957. Later rounds never recovered that region. Consistently negative throughout — the landscape resisted all surrogate-guided approaches.

**F7** — Gradient direction established in Round 6. Best of 1.8116 achieved in Round 7 and reproduced in Rounds 12 and 13 at identical coordinates. Confirmed stable peak.

**F8** — Plateau. Hovering near 9.920–9.925 since Round 3. Local maximum confirmed — the function has a ceiling in this region and no further improvement was achievable with remaining budget.
