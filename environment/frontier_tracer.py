# environment/frontier_tracer.py
"""Branch Coverage Manager for Retinal Vessel Tracing.
Implements the Frontier-Based Coverage (Algorithm 2) to trace the full
connected vascular tree.
"""

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from scipy.ndimage import convolve
from tqdm import tqdm


class FrontierTracer:
    """Single source of truth for Frontier-Based Coverage (Algorithm 2)."""

    def __init__(self, env, policy_model, device, obs_size: int = 65):
        self.env = env
        self.model = policy_model
        self.device = device
        self.obs_size = obs_size
        self.half = obs_size // 2

        # Preallocated inference buffer — filled in-place each step, never reallocated
        n_channels = env.observation_space.shape[0]
        self._obs_buf = torch.zeros(
            1,
            n_channels,
            obs_size,
            obs_size,
            dtype=torch.float32,
            device=device,
        )

    def _extract_endpoint_seeds(
        self,
        on_vessel_mask: np.ndarray,
        existing_frontier: List[Tuple[int, int]],
        min_distance: int = 8,
    ) -> List[Tuple[int, int]]:
        """Find dangling tips of the on-vessel skeleton.

        IMPORTANT: accepts `on_vessel_mask` — the on-vessel-only pixels from
        sub_paths — NOT the full combined_mask which includes bridge segments.
        Using bridge tips as seeds would extend false connections further.

        Returns endpoints sorted farthest-from-covered first, suppressing
        near-duplicates from the existing frontier.
        """
        if not on_vessel_mask.any():
            return []

        h, w = on_vessel_mask.shape
        margin = self.half + 5
        skel = (on_vessel_mask > 0).astype(np.uint8)

        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
        neighbour_count = convolve(skel, kernel, mode='constant', cval=0)
        endpoint_mask = (skel > 0) & (neighbour_count == 1)
        endpoints = np.argwhere(endpoint_mask)

        if len(endpoints) == 0:
            return []

        frontier_set = set(
            (int(s[0]), int(s[1])) for s in existing_frontier
        )

        # Distance transform from covered pixels for re-ranking
        covered_bin = (on_vessel_mask > 0).astype(np.uint8)
        dist_from_covered = cv2.distanceTransform(1 - covered_bin, cv2.DIST_L2, 5)

        new_seeds = []
        for ep in endpoints:
            ey, ex = int(ep[0]), int(ep[1])
            if not (margin <= ey < h - margin and margin <= ex < w - margin):
                continue
            if (ey, ex) in frontier_set:
                continue
            too_close = any(
                abs(ey - fy) + abs(ex - fx) < min_distance
                for fy, fx in frontier_set
            )
            if not too_close:
                new_seeds.append((ey, ex))

        new_seeds.sort(
            key=lambda s: dist_from_covered[s[0], s[1]], reverse=True
        )
        return new_seeds

    def _execute_single_trace(
        self, start_pos: Tuple[int, int], combined_mask: np.ndarray
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Executes a single continuous trace until the agent stops or terminates."""
        # Expose accumulated coverage from all previous traces so the agent can
        # observe what has already been traced via the prior_coverage channel.
        self.env.prior_coverage = combined_mask
        obs, _ = self.env.reset(start_position=start_pos)
        path = [start_pos]
        done = False
        alternate_branches = []

        self.model.eval()
        with torch.no_grad():
            while not done:
                self._obs_buf.copy_(torch.from_numpy(obs))
                logits, _, _ = self.model(self._obs_buf)
                action = logits.argmax(dim=-1).item()

                obs, _, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                y, x = self.env.position
                path.append((y, x))
                combined_mask[y, x] = 1.0

        return path, alternate_branches

    def trace_from_seeds(
        self,
        sample: Dict[str, Any],
        initial_seeds: List[Tuple[int, int]],
        min_coverage_gain: float = 0.0005,
        max_low_gain_traces: int = 5,
    ) -> Tuple[np.ndarray, List[List[Tuple[int, int]]]]:
        """End-to-end inference: stack-based frontier with dynamic endpoint growth.

        Improvements:
        - Fix B (skip covered starts): skips seeds whose start pixel is covered.
        - Fix A (endpoint growth): after each trace, extracts endpoints from the
          ON-VESSEL skeleton only (sub_paths, not the full path including bridges).
          This prevents bridge tips from seeding further false connections.
        - Early stopping: halts after consecutive low-gain traces.
        """
        self._setup_env(sample)
        h, w = sample["image"].shape[:2]
        combined_mask = np.zeros((h, w), dtype=np.float32)
        all_paths = []
        gt_total = float(max(sample["centerline"].sum(), 1))
        low_gain_streak = 0
        traces_run = 0

        frontier = list(initial_seeds)
        pbar = tqdm(total=len(frontier), desc="Tracing Seeds", unit="seed", leave=False)

        while frontier:
            start_pos = frontier.pop()
            pbar.update(1)

            # Skip seeds whose start pixel is already covered
            sy, sx = int(start_pos[0]), int(start_pos[1])
            if combined_mask[sy, sx] > 0:
                continue

            covered_before = combined_mask.sum()
            path, alternate_branches = self._execute_single_trace(
                start_pos, combined_mask
            )
            traces_run += 1

            # Split trace at off-vessel bridges
            dt = self.env.distance_transform
            tol = self.env.tolerance
            sub_paths = self._split_trace_at_bridges(path, dt, tol)

            # Build on-vessel-only mask for this trace (used for endpoint extraction)
            on_vessel_mask = np.zeros((h, w), dtype=np.float32)
            for sp in sub_paths:
                all_paths.append(sp)
                coords = np.array(sp, dtype=np.intp)
                combined_mask[coords[:, 0], coords[:, 1]] = 1.0
                on_vessel_mask[coords[:, 0], coords[:, 1]] = 1.0

            # Clear off-vessel bridge pixels from combined_mask
            path_arr = np.array(path, dtype=np.intp)
            dt_vals = dt[path_arr[:, 0], path_arr[:, 1]]
            off_mask = dt_vals > tol
            combined_mask[path_arr[off_mask, 0], path_arr[off_mask, 1]] = 0.0

            # Coverage gain tracking for early stopping
            gain = (combined_mask.sum() - covered_before) / gt_total
            if gain < min_coverage_gain:
                low_gain_streak += 1
            else:
                low_gain_streak = 0

            if low_gain_streak >= max_low_gain_traces:
                tqdm.write(
                    f"    Early stop: {max_low_gain_traces} consecutive "
                    f"low-gain traces (gain < {min_coverage_gain:.4f})"
                )
                break

            # Fix A: extract endpoints from the ON-VESSEL mask only.
            # Using the full path (including bridges) would seed the agent
            # at bridge tips, causing further false connections downstream.
            new_endpoints = self._extract_endpoint_seeds(on_vessel_mask, frontier)
            if new_endpoints:
                frontier.extend(new_endpoints)
                pbar.total += len(new_endpoints)

            # Gap reseeder: when the frontier is nearly empty and no new
            # endpoints were found, inject seeds at uncovered vessel regions
            # that are far (>40px) from any covered pixel.  This prevents
            # starvation when disconnected subtrees were never reached by any
            # trace endpoint.
            if not new_endpoints and len(frontier) < 3:
                gap_seeds = self._gap_reseeder(combined_mask, sample)
                if gap_seeds:
                    frontier.extend(gap_seeds)
                    pbar.total += len(gap_seeds)
                    tqdm.write(
                        f"    Gap reseeder: injected {len(gap_seeds)} seeds "
                        f"(coverage={combined_mask.sum() / gt_total:.3f})"
                    )

            for branch_pos in alternate_branches:
                if combined_mask[branch_pos[0], branch_pos[1]] == 0:
                    frontier.append(branch_pos)
                    pbar.total += 1

        pbar.close()
        tqdm.write(
            f"    Frontier tracer: {traces_run} traces, "
            f"coverage={combined_mask.sum() / gt_total:.3f}"
        )
        return combined_mask, all_paths

    def trace_with_gt_gaps(
        self,
        sample: Dict[str, Any],
        max_traces: int = 50,
        min_coverage_gain: float = 0.005,
    ) -> Tuple[np.ndarray, List[List[Tuple[int, int]]]]:
        """Evaluation method: Iteratively forces the agent into ground-truth gaps."""
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

            # Split trace at off-vessel bridges
            dt = self.env.distance_transform
            tol = self.env.tolerance
            sub_paths = self._split_trace_at_bridges(path, dt, tol)

            # Re-stamp combined_mask with only on-vessel segments (Vectorized)
            for sp in sub_paths:
                all_paths.append(sp)
                coords = np.array(sp, dtype=np.intp)
                combined_mask[coords[:, 0], coords[:, 1]] = 1.0

            # Clear off-vessel bridge pixels from combined_mask (Vectorized)
            path_arr = np.array(path, dtype=np.intp)
            dt_vals = dt[path_arr[:, 0], path_arr[:, 1]]
            off_mask = dt_vals > tol
            combined_mask[path_arr[off_mask, 0], path_arr[off_mask, 1]] = 0.0

            # all_paths.append(path)

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

    def _gap_reseeder(
        self,
        combined_mask: np.ndarray,
        sample: Dict[str, Any],
        n_gap_seeds: int = 40,
        gap_threshold: int = 25,
    ) -> List[Tuple[int, int]]:
        """Find vessel pixels far from covered regions → new frontier seeds.

        Prevents starvation when the frontier runs dry while large uncovered
        vessel segments remain (e.g. disconnected subtrees or peripheral
        branches that no trace endpoint reached).

        Uses the GT centerline as the vessel proxy.  gap_threshold controls
        the minimum distance from any covered pixel before a vessel point is
        considered a gap.  Seeds are spaced at least gap_threshold//2 apart.
        """
        from scipy.ndimage import distance_transform_edt

        centerline = sample["centerline"]
        h, w = centerline.shape
        margin = self.half + 5

        vessel_uncovered = (centerline > 0) & (combined_mask == 0)
        if not vessel_uncovered.any():
            return []

        if combined_mask.any():
            dist_from_covered = distance_transform_edt(combined_mask == 0).astype(np.float32)
        else:
            dist_from_covered = np.full((h, w), float(max(h, w)), dtype=np.float32)

        gap_mask = vessel_uncovered & (dist_from_covered > gap_threshold)
        gap_pts = np.argwhere(gap_mask)
        if len(gap_pts) == 0:
            return []

        valid = [
            (int(y), int(x)) for y, x in gap_pts
            if margin <= y < h - margin and margin <= x < w - margin
        ]
        if not valid:
            return []

        valid.sort(key=lambda yx: dist_from_covered[yx[0], yx[1]], reverse=True)

        min_half = max(4, gap_threshold // 4)
        selected = []
        occupied = np.zeros((h, w), dtype=bool)
        for y, x in valid:
            if len(selected) >= n_gap_seeds:
                break
            if occupied[max(0, y - min_half):y + min_half + 1,
                        max(0, x - min_half):x + min_half + 1].any():
                continue
            selected.append((y, x))
            occupied[max(0, y - min_half):y + min_half + 1,
                     max(0, x - min_half):x + min_half + 1] = True

        return selected

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

    def _split_trace_at_bridges(
        self,
        path: List[Tuple[int, int]],
        distance_transform: np.ndarray,
        tolerance: float,
    ) -> List[List[Tuple[int, int]]]:
        """Split a single trace into sub-traces wherever it went off-vessel.

        Removes bridge segments that cross non-vessel regions.
        Only keeps sub-traces with >= 3 on-vessel points.
        """
        if len(path) < 3:
            return [path]

        coords = np.array(path, dtype=np.intp)
        on_vessel = distance_transform[coords[:, 0], coords[:, 1]] <= tolerance

        # Find boundaries where on_vessel changes
        changes = np.diff(on_vessel.astype(np.int8))
        split_indices = np.where(changes != 0)[0] + 1
        chunks = np.split(np.arange(len(path)), split_indices)

        segments = []
        for chunk in chunks:
            if len(chunk) >= 3 and on_vessel[chunk[0]]:
                segments.append([tuple(coords[i]) for i in chunk])

        return segments if segments else [path]
