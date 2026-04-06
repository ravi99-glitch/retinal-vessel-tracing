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
        prev_action = None  

        self.model.eval()
        with torch.no_grad():
            while not done:
                self._obs_buf.copy_(torch.from_numpy(obs))
                logits, _, _ = self.model(self._obs_buf)
                
                # 1. Ban the "STOP" action
                logits[0, 8] = -float("inf")
                
                # 2. Ban the "REVERSE" action to prevent 1-pixel oscillation loops
                if prev_action is not None:
                    reverse_action = (prev_action + 4) % 8
                    logits[0, reverse_action] = -float("inf")

                action = logits.argmax(dim=-1).item()
                prev_action = action

                obs, _, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                y, x = self.env.position
                path.append((y, x))
                combined_mask[y, x] = 1.0

        return path, alternate_branches

    def trace_from_seeds(self, sample: dict, initial_seeds: list):
        """Traces paths starting from seeds, dynamically queuing new branches it discovers."""
        self._setup_env(sample)
        
        seeds_queue = list(initial_seeds)
        
        h, w = sample["image"].shape[:2]
        combined_mask = np.zeros((h, w), dtype=np.float32)
        
        # --- Spatial Anti-Spam Mask ---
        queued_mask = np.zeros((h, w), dtype=np.uint8)
        for sy, sx in initial_seeds:
            queued_mask[int(sy), int(sx)] = 1
        # -----------------------------------

        paths = []
        tolerance = int(self.env.tolerance)
        
        # Use a slightly fatter brush to hide thick artery walls from the radar
        brush_size = tolerance + 1 

        MAX_DYNAMIC_TRACES = 2000 
        traces_run = 0

        pbar = tqdm(total=len(seeds_queue), desc="Active Branch Tracing", leave=False)

        while len(seeds_queue) > 0 and traces_run < MAX_DYNAMIC_TRACES:
            start = seeds_queue.pop(0)
            
            if combined_mask[int(start[0]), int(start[1])] > 0:
                pbar.update(1)
                continue

            traces_run += 1
            obs, _ = self.env.reset(start_position=start)
            path = [tuple(start)]
            done = False
            prev_action = None  

            while not done:
                self._obs_buf.copy_(torch.from_numpy(obs))
                
                with torch.no_grad():
                    logits, _, _ = self.model(self._obs_buf)
                    logits[0, 8] = -float("inf")
                    if prev_action is not None:
                        reverse_action = (prev_action + 4) % 8
                        logits[0, reverse_action] = -float("inf")
                    action = logits.argmax(dim=-1).item()
                    prev_action = action

                obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                y, x = self.env.position
                path.append((y, x))
                
                # --- Use the fatter brush ---
                cv2.circle(combined_mask, (int(x), int(y)), brush_size, 1.0, -1)

                # ==================================================
                # ACTIVE BRANCH QUEUING
                # ==================================================
                # --- Cooldown (only check every 4 steps) ---
                if len(path) > 5 and len(path) % 4 == 0:
                    window = 15 
                    half_w = window // 2
                    y_min, y_max = max(0, y - half_w), min(h, y + half_w + 1)
                    x_min, x_max = max(0, x - half_w), min(w, x + half_w + 1)
                    
                    local_vessel = sample["vessel_mask"][y_min:y_max, x_min:x_max]
                    local_covered = combined_mask[y_min:y_max, x_min:x_max]
                    
                    untraced_y, untraced_x = np.where((local_vessel > 0) & (local_covered == 0))
                    
                    if len(untraced_y) > 0:
                        global_y = untraced_y + y_min
                        global_x = untraced_x + x_min
                        
                        local_dt = sample["distance_transform"][global_y, global_x]
                        best_idx = np.argmax(local_dt)
                        
                        new_seed = (int(global_y[best_idx]), int(global_x[best_idx]))
                        
                        # --- NEW: Check the spatial mask to ensure we haven't queued this junction ---
                        if queued_mask[new_seed[0], new_seed[1]] == 0 and combined_mask[new_seed[0], new_seed[1]] == 0:
                            seeds_queue.append(new_seed)
                            
                            # Mark a 3px radius as "Queued" so we don't spam 10 seeds on the same branch!
                            cv2.circle(queued_mask, (new_seed[1], new_seed[0]), 3, 1, -1)
                            pbar.total += 1 
                # ==================================================

            paths.append(path)
            pbar.update(1)

        pbar.close()
        return combined_mask, paths

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

        for trace_idx in tqdm(range(max_traces), desc="GT Gap Tracing", unit="trace"):
            start_pos = self._pick_frontier_seed_from_gt(
                sample["centerline"], combined_mask
            )

            if start_pos is None:
                tqdm.write(f"    Full coverage after {trace_idx} traces.")
                break

            covered_before = combined_mask.sum()
            path, _ = self._execute_single_trace(start_pos, combined_mask)

            dt = self.env.distance_transform
            tol = self.env.tolerance
            sub_paths = self._split_trace_at_bridges(path, dt, tol)

            for sp in sub_paths:
                all_paths.append(sp)
                coords = np.array(sp, dtype=np.intp)
                combined_mask[coords[:, 0], coords[:, 1]] = 1.0

            path_arr = np.array(path, dtype=np.intp)
            dt_vals = dt[path_arr[:, 0], path_arr[:, 1]]
            off_mask = dt_vals > tol
            combined_mask[path_arr[off_mask, 0], path_arr[off_mask, 1]] = 0.0

            gain = (combined_mask.sum() - covered_before) / gt_total
            coverage_pct = combined_mask.sum() / gt_total

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
            vessel_orientation=sample.get("vessel_orientation"),
            dt_gradient=sample.get("dt_gradient"),
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
        if len(path) < 3:
            return [path]

        coords = np.array(path, dtype=np.intp)
        on_vessel = distance_transform[coords[:, 0], coords[:, 1]] <= tolerance

        changes = np.diff(on_vessel.astype(np.int8))
        split_indices = np.where(changes != 0)[0] + 1
        chunks = np.split(np.arange(len(path)), split_indices)

        segments = []
        for chunk in chunks:
            if len(chunk) >= 3 and on_vessel[chunk[0]]:
                segments.append([tuple(coords[i]) for i in chunk])

        return segments if segments else [path]
