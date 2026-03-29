# # environment/frontier_tracer.py
# """Branch Coverage Manager for Retinal Vessel Tracing.
# Implements the Frontier-Based Coverage (Algorithm 2) to trace the full
# connected vascular tree.
# """

# from typing import Any, Dict, List, Optional, Tuple

# import cv2
# import numpy as np
# import torch
# from tqdm import tqdm


# class FrontierTracer:
#     """Single source of truth for Frontier-Based Coverage (Algorithm 2)."""

#     def __init__(self, env, policy_model, device, obs_size: int = 65):
#         self.env = env
#         self.model = policy_model
#         self.device = device
#         self.obs_size = obs_size
#         self.half = obs_size // 2

#         # Preallocated inference buffer — filled in-place each step, never reallocated
#         n_channels = env.observation_space.shape[0]
#         self._obs_buf = torch.zeros(
#             1,
#             n_channels,
#             obs_size,
#             obs_size,
#             dtype=torch.float32,
#             device=device,
#         )

#     def _execute_single_trace(
#         self, start_pos: Tuple[int, int], combined_mask: np.ndarray
#     ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
#         """Executes a single continuous trace until the agent stops or terminates."""
#         obs, _ = self.env.reset(start_position=start_pos)
#         path = [start_pos]
#         done = False
#         alternate_branches = []

#         self.model.eval()
#         with torch.no_grad():
#             while not done:
#                 self._obs_buf.copy_(torch.from_numpy(obs))
#                 logits, _, _ = self.model(self._obs_buf)
#                 action = logits.argmax(dim=-1).item()

#                 obs, _, terminated, truncated, _ = self.env.step(action)
#                 done = terminated or truncated

#                 y, x = self.env.position
#                 path.append((y, x))
#                 combined_mask[y, x] = 1.0

#         return path, alternate_branches

#     def trace_from_seeds(
#         self, sample: Dict[str, Any], initial_seeds: List[Tuple[int, int]]
#     ) -> Tuple[np.ndarray, List[List[Tuple[int, int]]]]:
#         """True End-to-End Inference: Algorithm 2 using a stack-based frontier."""
#         self._setup_env(sample)
#         h, w = sample["image"].shape[:2]
#         combined_mask = np.zeros((h, w), dtype=np.float32)
#         all_paths = []

#         frontier = list(initial_seeds)

#         # Added progress bar for seed exploration
#         pbar = tqdm(total=len(frontier), desc="Tracing Seeds", unit="seed", leave=False)

#         while frontier:
#             start_pos = frontier.pop()
#             pbar.update(1)  # Manually update since the stack size changes dynamically

#             if combined_mask[start_pos[0], start_pos[1]] > 0:
#                 continue

#             path, alternate_branches = self._execute_single_trace(
#                 start_pos, combined_mask
#             )

#             # Split trace at off-vessel bridges
#             dt = self.env.distance_transform
#             tol = self.env.tolerance
#             sub_paths = self._split_trace_at_bridges(path, dt, tol)

#             # Re-stamp combined_mask with only on-vessel segments
#             for sp in sub_paths:
#                 all_paths.append(sp)
#                 for y, x in sp:
#                     combined_mask[y, x] = 1.0

#             # Clear off-vessel bridge pixels from combined_mask
#             for y, x in path:
#                 if dt[y, x] > tol:
#                     combined_mask[y, x] = 0.0

#             # all_paths.append(path)

#             for branch_pos in alternate_branches:
#                 if combined_mask[branch_pos[0], branch_pos[1]] == 0:
#                     frontier.append(branch_pos)
#                     pbar.total += 1  # Increase total if new branches are found

#         pbar.close()
#         return combined_mask, all_paths

#     def trace_with_gt_gaps(
#         self,
#         sample: Dict[str, Any],
#         max_traces: int = 50,
#         min_coverage_gain: float = 0.005,
#     ) -> Tuple[np.ndarray, List[List[Tuple[int, int]]]]:
#         """Evaluation method: Iteratively forces the agent into ground-truth gaps."""
#         self._setup_env(sample)
#         h, w = sample["image"].shape[:2]
#         combined_mask = np.zeros((h, w), dtype=np.float32)
#         all_paths = []
#         gt_total = float(max(sample["centerline"].sum(), 1))

#         # Added tqdm to the GT evaluation loop
#         for trace_idx in tqdm(range(max_traces), desc="GT Gap Tracing", unit="trace"):
#             start_pos = self._pick_frontier_seed_from_gt(
#                 sample["centerline"], combined_mask
#             )

#             if start_pos is None:
#                 tqdm.write(f"    Full coverage after {trace_idx} traces.")
#                 break

#             covered_before = combined_mask.sum()
#             path, _ = self._execute_single_trace(start_pos, combined_mask)

#             # Split trace at off-vessel bridges
#             dt = self.env.distance_transform
#             tol = self.env.tolerance
#             sub_paths = self._split_trace_at_bridges(path, dt, tol)

#             # Re-stamp combined_mask with only on-vessel segments
#             for sp in sub_paths:
#                 all_paths.append(sp)
#                 for y, x in sp:
#                     combined_mask[y, x] = 1.0

#             # Clear off-vessel bridge pixels from combined_mask
#             for y, x in path:
#                 if dt[y, x] > tol:
#                     combined_mask[y, x] = 0.0

#             # all_paths.append(path)

#             gain = (combined_mask.sum() - covered_before) / gt_total
#             coverage_pct = combined_mask.sum() / gt_total

#             # Using tqdm.write instead of print
#             tqdm.write(
#                 f"    Trace {trace_idx+1:3d} from {start_pos} -> "
#                 f"{len(path)} steps  gain={gain:.3f}  coverage={coverage_pct:.3f}"
#             )

#             if trace_idx >= 3 and gain < min_coverage_gain:
#                 tqdm.write(f"    Early stop: gain {gain:.4f} < {min_coverage_gain}")
#                 break

#         return combined_mask, all_paths

#     def _setup_env(self, sample: Dict[str, Any]):
#         self.env.set_data(
#             image=sample["image"],
#             centerline=sample["centerline"],
#             distance_transform=sample["distance_transform"],
#             fov_mask=sample["fov_mask"],
#         )

#     def _pick_frontier_seed_from_gt(
#         self, gt_centerline: np.ndarray, covered: np.ndarray
#     ) -> Optional[Tuple[int, int]]:
#         uncovered = (gt_centerline > 0) & (covered == 0)
#         if not uncovered.any():
#             return None

#         uncovered_pts = np.argwhere(uncovered)
#         h, w = gt_centerline.shape

#         covered_bin = (covered > 0).astype(np.uint8)
#         if covered_bin.any():
#             dist = cv2.distanceTransform(1 - covered_bin, cv2.DIST_L2, 5)
#             scores = dist[uncovered_pts[:, 0], uncovered_pts[:, 1]]
#             best = uncovered_pts[np.argmax(scores)]
#         else:
#             centre = np.array([h // 2, w // 2])
#             dists = np.linalg.norm(uncovered_pts - centre, axis=1)
#             best = uncovered_pts[np.argmin(dists)]

#         y = int(np.clip(best[0], self.half, h - self.half - 1))
#         x = int(np.clip(best[1], self.half, w - self.half - 1))
#         return (y, x)

#     def _split_trace_at_bridges(
#         self,
#         path: List[Tuple[int, int]],
#         distance_transform: np.ndarray,
#         tolerance: float,
#     ) -> List[List[Tuple[int, int]]]:
#         """Split a single trace into sub-traces wherever it went off-vessel.

#         Removes bridge segments that cross non-vessel regions.
#         Only keeps sub-traces with >= 3 on-vessel points.
#         """
#         segments = []
#         current_segment = []

#         for y, x in path:
#             if distance_transform[y, x] <= tolerance:
#                 current_segment.append((y, x))
#             else:
#                 # Off-vessel point — close current segment
#                 if len(current_segment) >= 3:
#                     segments.append(current_segment)
#                 current_segment = []

#         # Don't forget the last segment
#         if len(current_segment) >= 3:
#             segments.append(current_segment)

#         return segments if segments else [path]  # fallback: keep original


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

    def _execute_single_trace(
        self, start_pos: Tuple[int, int], combined_mask: np.ndarray
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Executes a single continuous trace until the agent stops or terminates."""
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
        self, sample: Dict[str, Any], initial_seeds: List[Tuple[int, int]]
    ) -> Tuple[np.ndarray, List[List[Tuple[int, int]]]]:
        """True End-to-End Inference: Algorithm 2 using a stack-based frontier."""
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
