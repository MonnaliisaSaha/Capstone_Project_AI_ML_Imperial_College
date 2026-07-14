"""
Round 12 — Penultimate Push
============================
Two rounds left. Push F5 one more step along the confirmed gradient.
Hold confirmed peaks for F4, F7 and F8. Final adjustment attempts
on F2, F3 and F6 before the last round locks everything in.

The pipeline has four components working in sequence:

  1. GP (Gaussian Process)
     Fits a probabilistic surface to all observed data.
     Uses a Matern 2.5 kernel — assumes the function is smooth enough
     that nearby inputs give similar outputs. Outputs a mean prediction
     AND an uncertainty estimate for any candidate point.

  2. UCB (Upper Confidence Bound)
     Scores each candidate: score = mean + beta × std
     High beta → prioritise uncertain regions (exploration)
     Low beta  → prioritise high predicted mean (exploitation)
     Beta is tuned per function based on observed behaviour.

  3. SVM (Support Vector Machine)
     Trains a classifier on all observations, labelling the top 30%
     as "promising". Removes candidates in regions the data says are poor.
     This narrows the search space before UCB scoring.

  4. NN (Neural Network)
     Trains a small MLP on all observations. Runs backpropagation
     at the current best point to estimate the gradient direction.
     Nudges the final query toward improvement.

Round 12 changes vs Round 11:
  - F5: step increased to 0.035 — more aggressive x1 push (x1: 0.550 → 0.580)
  - F2: slight coordinate adjustment to probe adjacent territory
  - F3, F6: perturbation attempt before final lock-in
  - All exploit functions (F4, F7, F8): same config, tight exploitation
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")


# ── Per-function configuration ─────────────────────────────────────────────
# Beta and step size reflect each function's behaviour after 12 rounds.
# F5, F4, F7, F8 are in exploit mode — low beta, tight search.
# F1, F2, F3, F6 remain in moderate/explore mode — higher beta.

FUNCTION_CONFIG = {
    # fn_id: beta (explore/exploit balance), step (gradient nudge size)
    1: {"beta": 3.0,  "step": 0.010},   # No signal — keep exploring
    2: {"beta": 2.0,  "step": 0.003},   # Noisy — probe adjacent region
    3: {"beta": 2.0,  "step": 0.010},   # Fragile — careful perturbation
    4: {"beta": 1.5,  "step": 0.005},   # Narrow peak — tight exploit
    5: {"beta": 0.8,  "step": 0.035},   # Clear gradient — push x1 further
    6: {"beta": 2.5,  "step": 0.030},   # Multimodal — one more attempt
    7: {"beta": 1.0,  "step": 0.005},   # Direction confirmed — hold tight
    8: {"beta": 0.5,  "step": 0.003},   # Near ceiling — micro-refinement
}

FUNCTION_DIMS = {1: 2, 2: 2, 3: 3, 4: 4, 5: 4, 6: 5, 7: 6, 8: 8}


# ── Component 1: Gaussian Process ──────────────────────────────────────────

def fit_gp(X: np.ndarray, y: np.ndarray) -> GaussianProcessRegressor:
    """
    Fit a Gaussian Process surrogate to the observed data.

    The Matern 2.5 kernel is a good default for most smooth functions —
    it assumes continuity and once-differentiability without being overly
    optimistic about the smoothness. normalize_y=True helps when outputs
    span very different scales across functions (F5 in thousands, F1 near zero).
    """
    kernel = Matern(nu=2.5)
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-6,            # small nugget for numerical stability
        normalize_y=True,      # standardise outputs before fitting
        n_restarts_optimizer=5 # avoid local optima in kernel fitting
    )
    gp.fit(X, y)
    return gp


# ── Component 2: UCB Acquisition ───────────────────────────────────────────

def ucb_score(gp: GaussianProcessRegressor,
              candidates: np.ndarray,
              beta: float) -> np.ndarray:
    """
    Score candidates using Upper Confidence Bound.

    UCB = mean + beta × std

    The mean term exploits what we know — high mean = probably good.
    The std term explores — high uncertainty = worth checking.
    Beta controls the balance. It varies by function based on whether
    we have found a reliable direction (low beta) or are still searching
    (high beta).

    By Round 12, F5/F8 have very low beta (near-pure exploitation),
    while F1/F6 remain high (still exploring — no reliable peak found).
    """
    mean, std = gp.predict(candidates, return_std=True)
    return mean + beta * std


# ── Component 3: SVM Region Filter ─────────────────────────────────────────

def svm_filter(X: np.ndarray,
               y: np.ndarray,
               candidates: np.ndarray,
               percentile: int = 30) -> np.ndarray:
    """
    Filter candidates using an SVM trained on promising vs unpromising regions.

    Labels the top [percentile]% of observed outputs as positive class.
    Trains a binary SVM on these labels. Returns a boolean mask marking
    which candidates the SVM thinks are in promising territory.

    If there aren't enough examples of each class (common in early rounds
    or for pathological functions like F1), skips filtering and returns
    everything (True mask).
    """
    threshold = np.percentile(y, 100 - percentile)
    labels = (y >= threshold).astype(int)

    if labels.sum() < 2 or (1 - labels).sum() < 2:
        return np.ones(len(candidates), dtype=bool)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    cands_scaled = scaler.transform(candidates)

    svm = SVC(kernel="rbf", C=1.0, gamma="scale")
    svm.fit(X_scaled, labels)
    return svm.predict(cands_scaled) == 1


# ── Component 4: Neural Network Gradient ───────────────────────────────────

class NNSurrogate(nn.Module):
    """
    A small two-layer MLP used as a differentiable surrogate.

    We don't use this for its predictions directly — we use it for its
    gradients. After training on the observed data, we backpropagate
    from the output to the input at the current best point to find
    which direction would increase the predicted output.

    With ~22 observations per function by Round 12, the network has
    more data to work with than in early rounds, making its gradient
    estimates more reliable — especially for F5 and F8.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_nn(X: np.ndarray, y: np.ndarray,
             epochs: int = 500,
             weight_decay: float = 1e-3) -> NNSurrogate:
    """
    Train the NN surrogate on all observed data.

    weight_decay controls regularisation — prevents the network from
    overfitting to a small dataset. F5 and F8 use lower weight_decay
    because their gradient directions have been stable across many rounds
    and the network can afford to fit them more tightly.
    """
    model = NNSurrogate(X.shape[1])
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.01, weight_decay=weight_decay
    )
    criterion = nn.MSELoss()

    X_t = torch.FloatTensor(X)
    y_t = torch.FloatTensor(y.reshape(-1, 1))

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(X_t), y_t)
        loss.backward()
        optimizer.step()
    return model


def nn_gradient_nudge(model: NNSurrogate,
                      x_best: np.ndarray,
                      step_size: float) -> np.ndarray:
    """
    Compute the gradient at x_best and nudge in that direction.

    Backpropagation tells us: "at this point in the input space,
    which direction should we move each input to increase the output?"
    We normalise the gradient to unit length and take a step of
    [step_size] in that direction.

    For F5, step_size is 0.035 this round — a deliberate push along
    the confirmed x1 gradient direction. For F8, step_size is 0.003 —
    micro-refinement only at near-ceiling performance.

    The result is clipped to [0, 1] to stay within valid bounds.
    """
    x_t = torch.FloatTensor(x_best).requires_grad_(True)
    model.eval()
    output = model(x_t)
    output.backward()

    gradient = x_t.grad.detach().numpy()
    gradient = gradient / (np.linalg.norm(gradient) + 1e-8)  # normalise

    next_point = x_best + step_size * gradient
    return np.clip(next_point, 0.0, 1.0)


# ── Full Pipeline ───────────────────────────────────────────────────────────

def generate_query(fn_id: int,
                   X: np.ndarray,
                   y: np.ndarray,
                   n_candidates: int = 10000) -> np.ndarray:
    """
    Run the full GP + UCB + SVM + NN pipeline for one function.

    Steps:
      1. Fit GP on all observed (X, y) pairs
      2. Generate [n_candidates] random candidate points
      3. Filter candidates with SVM — remove unpromising regions
      4. Score remaining candidates with UCB
      5. Select the highest-scoring candidate
      6. Refine with NN gradient nudge
      7. Return the refined point as the next query

    Args:
        fn_id: Function identifier (1-8)
        X:     All observed inputs so far, shape (n_obs, dims)
        y:     All observed outputs so far, shape (n_obs,)

    Returns:
        next_query: np.ndarray of shape (dims,), clipped to [0, 1]
    """
    config = FUNCTION_CONFIG[fn_id]
    dims = FUNCTION_DIMS[fn_id]
    beta = config["beta"]
    step_size = config["step"]

    # Step 1: Fit GP
    gp = fit_gp(X, y)

    # Step 2: Generate candidates
    candidates = np.random.uniform(0.0, 1.0, (n_candidates, dims))

    # Step 3: SVM filter — remove regions the data says are poor
    mask = svm_filter(X, y, candidates)
    filtered = candidates[mask] if mask.sum() > 0 else candidates

    # Step 4: UCB scoring — balance predicted value and uncertainty
    scores = ucb_score(gp, filtered, beta)
    best_candidate = filtered[np.argmax(scores)]

    # Step 5: NN gradient refinement — nudge toward improving direction
    # Lower weight_decay for functions with stable, reliable gradients
    weight_decay = 1e-4 if fn_id in [5, 8] else 1e-3
    nn_model = train_nn(X, y, weight_decay=weight_decay)
    refined = nn_gradient_nudge(nn_model, best_candidate, step_size)

    return refined


def format_for_portal(x: np.ndarray) -> str:
    """Format query vector for portal submission."""
    return "-".join(f"{v:.6f}" for v in x)


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("BBO Round 12 — Penultimate Push")
    print("GP + UCB (per-function beta) + SVM + NN gradient")
    print("=" * 60)

    # Round 12 actual queries submitted to portal
    # Replace X_obs / y_obs with your accumulated observation arrays
    # before re-running. Queries below are confirmed Round 12 submissions.

    round_12_queries = {
        1: [0.500000, 0.500000],
        2: [0.678000, 0.939000],
        3: [0.504000, 0.296000, 0.454000],
        4: [0.360000, 0.410000, 0.430000, 0.395000],
        5: [0.580000, 0.936000, 0.958000, 0.953000],
        6: [0.320000, 0.135000, 0.400000, 0.630000, 0.135000],
        7: [0.050000, 0.250000, 0.240000, 0.230000, 0.430000, 0.770000],
        8: [0.050000, 0.188000, 0.055000, 0.107000, 0.878000, 0.413000,
            0.058000, 0.461000],
    }

    round_12_outputs = {
        1: 2.6752879910742468e-9,
        2: 0.4667123767190129,
        3: -0.017028992981422686,
        4: 0.5533948144101939,
        5: 3030.585505008791,
        6: -0.7786435888166022,
        7: 1.8116258350443388,
        8: 9.9219609,
    }

    print("\nRound 12 results:")
    for fn_id in range(1, 9):
        dims = FUNCTION_DIMS[fn_id]
        cfg = FUNCTION_CONFIG[fn_id]
        q = round_12_queries[fn_id]
        out = round_12_outputs[fn_id]
        print(f"  F{fn_id} ({dims}D) beta={cfg['beta']} step={cfg['step']}: "
              f"{format_for_portal(np.array(q))}  →  {out}")

    print("\nTo generate fresh queries, replace X_obs/y_obs with")
    print("accumulated portal data and call generate_query(fn_id, X_obs, y_obs).")
