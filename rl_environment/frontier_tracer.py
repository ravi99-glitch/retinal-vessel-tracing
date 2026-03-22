# environment/frontier_tracer.py
"""Branch Coverage Manager for Retinal Vessel Tracing.
Implements the Frontier-Based Coverage (Algorithm 2) to trace the full
connected vascular tree.
"""

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm


class FrontierTracer:
    """Single source of truth for Frontier-Based Coverage (Algorithm 2).
    """

    def __init__(self, env, policy_model, device, obs_size: int = 65):
        self.env = env
        self.model = policy_model
        self.device = device
        self.obs_size = obs_size
        self.half = obs_size // 2

    def _execute_single_trace(
        self, start_pos: Tuple[int, int], combined_mask: np.ndarray
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Executes a single continuous trace until the agent stops or terminates.
        """
        obs, _ = self.env.reset(start_position=start_pos)
        path = [start_pos]
        done = False
        alternate_branches = []

        self.model.eval()
        with torch.no_grad():
            while not done:
                obs_t = (
                    torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
                )
                logits, _, _ = self.model(obs_t)
                action = logits.argmax(dim=-1).item()

                obs, _, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                y, x = self.env.position
                path.append((y, x))
                combined_mask[y, x] = 1.0

        return path, alternate_branches

    def trace_from_seeds(
        self, sample: Dict[str, Any], initial_seeds: List[Tuple[int, int]]
    ) -> Tuple[np.ndarray, List[List[Tuple[int, int]]]]:
        """True End-to-End Inference: Algorithm 2 using a stack-based frontier.
        """
        self._setup_env(sample)
        h, w = sample["image"].shape[:2]
        combined_mask = np.zeros((h, w), dtype=np.float32)
        all_paths = []

        frontier = list(initial_seeds)

        # Added progress bar for seed exploration
        pbar = tqdm(total=len(frontier), desc="Tracing Seeds", unit="seed", leave=False)

        while frontier:
            start_pos = frontier.pop()
            pbar.update(1)  # Manually update since the stack size changes dynamically

            if combined_mask[start_pos[0], start_pos[1]] > 0:
                continue

            path, alternate_branches = self._execute_single_trace(
                start_pos, combined_mask
            )
            all_paths.append(path)

            for branch_pos in alternate_branches:
                if combined_mask[branch_pos[0], branch_pos[1]] == 0:
                    frontier.append(branch_pos)
                    pbar.total += 1  # Increase total if new branches are found

        pbar.close()
        return combined_mask, all_paths

    def trace_with_gt_gaps(
        self,
        sample: Dict[str, Any],
        max_traces: int = 50,
        min_coverage_gain: float = 0.005,
    ) -> Tuple[np.ndarray, List[List[Tuple[int, int]]]]:
        """Evaluation method: Iteratively forces the agent into ground-truth gaps.
        """
        self._setup_env(sample)
        h, w = sample["image"].shape[:2]
        combined_mask = np.zeros((h, w), dtype=np.float32)
        all_paths = []
        gt_total = float(max(sample["centerline"].sum(), 1))

        # Added tqdm to the GT evaluation loop
        for trace_idx in tqdm(range(max_traces), desc="GT Gap Tracing", unit="trace"):
            start_pos = self._pick_frontier_seed_from_gt(
                sample["centerline"], combined_mask
            )

            if start_pos is None:
                tqdm.write(f"    Full coverage after {trace_idx} traces.")
                break

            covered_before = combined_mask.sum()
            path, _ = self._execute_single_trace(start_pos, combined_mask)
            all_paths.append(path)

            gain = (combined_mask.sum() - covered_before) / gt_total
            coverage_pct = combined_mask.sum() / gt_total

            # Using tqdm.write instead of print
            tqdm.write(
                f"    Trace {trace_idx+1:3d} from {start_pos} -> "
                f"{len(path)} steps  gain={gain:.3f}  coverage={coverage_pct:.3f}"
            )

            if trace_idx >= 3 and gain < min_coverage_gain:
                tqdm.write(f"    Early stop: gain {gain:.4f} < {min_coverage_gain}")
                break

        return combined_mask, all_paths

    def _setup_env(self, sample: Dict[str, Any]):
        self.env.set_data(
            image=sample["image"],
            centerline=sample["centerline"],
            distance_transform=sample["distance_transform"],
            fov_mask=sample["fov_mask"],
        )

    def _pick_frontier_seed_from_gt(
        self, gt_centerline: np.ndarray, covered: np.ndarray
    ) -> Optional[Tuple[int, int]]:
        uncovered = (gt_centerline > 0) & (covered == 0)
        if not uncovered.any():
            return None

        uncovered_pts = np.argwhere(uncovered)
        h, w = gt_centerline.shape

        covered_bin = (covered > 0).astype(np.uint8)
        if covered_bin.any():
            dist = cv2.distanceTransform(1 - covered_bin, cv2.DIST_L2, 5)
            scores = dist[uncovered_pts[:, 0], uncovered_pts[:, 1]]
            best = uncovered_pts[np.argmax(scores)]
        else:
            centre = np.array([h // 2, w // 2])
            dists = np.linalg.norm(uncovered_pts - centre, axis=1)
            best = uncovered_pts[np.argmin(dists)]

        y = int(np.clip(best[0], self.half, h - self.half - 1))
        x = int(np.clip(best[1], self.half, w - self.half - 1))
        return (y, x)
