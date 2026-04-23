"""Reward calculation for retinal vessel tracing.

Seven components, matching the bachelor-thesis proposal's tolerance-aware reward
design.  Training signal is aligned with the eval metric (clDice) — see
``VesselTracingEnv._compute_fbeta`` which operates on ``covered_centerline``
(the same mask the eval loop uses), not on ``trajectory_mask``.

Per-step components (from the proposal):
  r_coverage    — β × (new GT centerline px covered)               [coverage]
  r_near        — α × max(0, 1 − D(p)/τ)                           [proximity]
  r_off_vessel  — γ_off  if  D(p) > τ                              [off-track]
  r_revisit     — −λ     if  pixel was previously visited           [no-loops]
  r_step_cost   — c_step                                            [efficiency]
  r_shaping     — potential-based shaping on Φ(p) = −min(D,τ)/τ    [dense guidance]

Terminal:
  r_terminal    — w × F_β(covered_centerline, GT)                  [clDice proxy]
                  plus early_stop_penalty on premature STOP,
                  oob_penalty for out-of-bounds termination.

Scales (default config):
    β (beta_coverage)      = 0.3    → ~270 units per 500-step episode (dominant)
    α (alpha_near)         = 0.01   → up to +5 per episode (continuous guidance)
    γ_off (gamma_off)      = -0.2   → ~-30 per episode at 30% off-track
    λ (lambda_revisit)     = 0.02   → ~-2 per episode for a normal policy
    c_step (step_cost)     = -0.01  → -5 per episode
    shaping_weight         = 1.0    → ~+1 per episode (policy-invariant delta)
    w (terminal_f1_weight) = 8.0    → ~+5 per terminal at f_β=0.65
    Total typical:                      ~+244 per episode (coverage-dominated)

Design notes:
  • Terminal F_β uses ``covered_centerline`` (NOT ``trajectory_mask``) so the
    training signal aligns with clDice: Tsens ≈ recall(covered, GT),
    Tprec ≈ precision(covered, GT).
  • Shaping must use shaping_gamma == training.ppo.gamma (enforced in
    PPOTrainer.__init__) for policy invariance (Ng, Daswani & Russell 1999).
  • Revisit penalty is computed before applying the step-cost on revisits,
    so a revisit step nets to (−λ − c_step) = -0.03 (small, directional).

All weights are read from ``config["reward"]``; see MODEL_CONFIG in config.py.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


# ── State dataclass ───────────────────────────────────────────────────────────


@dataclass
class RewardState:
    """Snapshot of environment state for a single reward computation.

    Build this in ``VesselTracingEnv.step()`` and pass it to
    ``RewardCalculator.compute()``.
    """

    # ── Episode routing ───────────────────────────────────────────────────
    is_terminal: bool
    terminal_reason: str       # "stop" | "off_track" | "max_steps" | "oob" | ""

    # ── Step-component inputs ─────────────────────────────────────────────
    new_coverage: float        # GT centreline pixels newly covered this step
    is_on_track: bool          # dt[position] ≤ tolerance
    distance: float            # distance-transform value at current position
    prev_distance: float       # distance-transform value at previous position

    # ── Terminal-component inputs ─────────────────────────────────────────
    coverage: float            # current episode coverage ratio [0, 1]
    f_beta_score: float        # pre-computed F-β; 0.0 for non-terminal steps

    # ── Context (not used in reward computation) ──────────────────────────
    position: Optional[np.ndarray] = None  # (y, x) — for logging / debugging
    step_number: int = 0
    junction_map_value: float = 0.0        # read by env for junction off-track tolerance
    is_revisit: bool = False               # True if current pixel was previously visited


# ── Reward calculator ─────────────────────────────────────────────────────────


class RewardCalculator:
    """Seven-component vessel-tracing reward (paper-aligned).

    Usage::

        calc = RewardCalculator(config)
        reward, breakdown = calc.compute(state)

    ``breakdown`` always contains all :attr:`BREAKDOWN_KEYS` so callers can
    accumulate per-component means without key-checking.
    """

    #: Ordered tuple of component names returned in every breakdown dict.
    BREAKDOWN_KEYS: Tuple[str, ...] = (
        "r_coverage",
        "r_near",
        "r_off_vessel",
        "r_revisit",
        "r_step_cost",
        "r_shaping",
        "r_terminal",
    )

    def __init__(self, config: Dict[str, Any]) -> None:
        rc = config.get("reward", {})
        ec = config.get("environment", {})

        # r_coverage — raw new-pixel count, moderately scaled
        self.beta: float = rc.get("beta_coverage", 0.3)

        # r_near — continuous proximity reward within tolerance.
        # α · max(0, 1 − D(p)/τ): peaks at α on the centerline, zero at/beyond τ.
        self.alpha_near: float = rc.get("alpha_near", 0.01)

        # r_off_vessel — flat, gentle penalty
        self.gamma_off: float = rc.get("gamma_off", -0.2)

        # r_revisit — explicit penalty for stepping onto an already-visited pixel.
        self.lambda_revisit: float = rc.get("lambda_revisit", 0.02)

        # r_step_cost
        self.step_cost: float = rc.get("step_cost", -0.01)

        # r_shaping
        self.shaping_weight: float = rc.get("shaping_weight", 1.0)
        self.shaping_gamma: float = rc.get("shaping_gamma", 0.99)
        self.tolerance: float = ec.get("tolerance", 2.0)

        # r_terminal — clDice-aligned (computed on covered_centerline)
        self.terminal_f1_weight: float = rc.get("terminal_f1_weight", 8.0)
        self.terminal_recall_beta_sq: float = float(
            rc.get("terminal_recall_beta_sq", 4.0)
        )
        self.min_stop_coverage: float = rc.get("min_stop_coverage", 0.05)
        self.early_stop_penalty: float = rc.get("early_stop_penalty", -1.0)
        self.oob_penalty: float = rc.get("oob_penalty", -5.0)

    # ── Public interface ──────────────────────────────────────────────────────

    def compute(self, state: RewardState) -> Tuple[float, Dict[str, float]]:
        """Compute total reward and per-component breakdown for one step.

        Returns ``(total_reward, breakdown)`` where ``breakdown`` is a dict
        with exactly the keys in :attr:`BREAKDOWN_KEYS`, all zero-filled for
        inactive components.
        """
        bd: Dict[str, float] = {k: 0.0 for k in self.BREAKDOWN_KEYS}

        # Out-of-bounds: strong penalty, no step components
        if state.terminal_reason == "oob":
            bd["r_terminal"] = self.oob_penalty
            return bd["r_terminal"], bd

        # ── Step components (skipped for the STOP action — no movement) ───
        if state.terminal_reason != "stop":

            # Raw new-pixel coverage — the dominant dense signal.
            bd["r_coverage"] = self.beta * state.new_coverage

            # Continuous proximity reward within tolerance (paper eq. 1):
            #   r_near = α · max(0, 1 − D(p)/τ)
            # Peaks at α when the agent is on the skeleton, zero at / beyond τ.
            if self.alpha_near != 0.0 and state.is_on_track:
                tol = max(self.tolerance, 1e-6)
                bd["r_near"] = self.alpha_near * max(
                    0.0, 1.0 - state.distance / tol
                )

            # Flat penalty for every step off the vessel.
            if not state.is_on_track:
                bd["r_off_vessel"] = self.gamma_off

            # Explicit revisit penalty (paper eq. 4).
            if self.lambda_revisit != 0.0 and state.is_revisit:
                bd["r_revisit"] = -self.lambda_revisit

            # Constant per-step cost.
            bd["r_step_cost"] = self.step_cost

            # Potential-based shaping: Φ(s) = −min(dt(s), ε) / ε
            if self.shaping_weight != 0.0:
                tol = max(self.tolerance, 1e-6)
                phi_prev = -min(state.prev_distance, self.tolerance) / tol
                phi_curr = -min(state.distance, self.tolerance) / tol
                bd["r_shaping"] = self.shaping_weight * (
                    self.shaping_gamma * phi_curr - phi_prev
                )

        # ── Terminal component (every episode end) ────────────────────────
        if state.is_terminal:
            r_t = self.terminal_f1_weight * state.f_beta_score
            # Penalise stopping before covering a meaningful fraction
            if (
                state.terminal_reason == "stop"
                and state.coverage < self.min_stop_coverage
            ):
                r_t += self.early_stop_penalty
            bd["r_terminal"] = r_t

        return sum(bd.values()), bd
