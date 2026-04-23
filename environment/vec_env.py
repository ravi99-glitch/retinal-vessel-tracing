"""Subprocess-based vectorized environment for parallel PPO rollouts.

Each VesselTracingEnv runs in its own process, stepping in parallel.
Communication via multiprocessing.Pipe — the standard SubprocVecEnv pattern.

Workers load their own dataset copy to avoid serializing large arrays
through the pipe on every episode reset — only the sample index is sent.
"""

import multiprocessing as mp
from typing import Dict, List, Optional, Tuple

import numpy as np


def _worker(conn, config: dict, tolerance: float):
    """Event loop running in a child process. Owns one VesselTracingEnv."""
    from data.dataloader import get_data
    from environment.vessel_env import VesselTracingEnv

    env = VesselTracingEnv(config)

    _env_cfg = config.get("environment", {})
    _use_vesselness = _env_cfg.get("use_vesselness", False)
    _use_unet_prior = _env_cfg.get("use_unet_prior", False)

    # Load own dataset copy for local sample access (avoids large IPC).
    # Only the torch Dataset metadata is loaded here; images load on demand.
    ds, _ = get_data(
        "rl_agent", "train", tolerance=tolerance, use_unet_prior=_use_unet_prior
    )
    _sample_cache: Dict[int, dict] = {}
    _MAX_CACHE = 32  # cap per-worker memory (~160 MB at 5 MB/sample)

    def _get_sample(idx: int) -> dict:
        if idx in _sample_cache:
            return _sample_cache[idx]
        s = ds[idx]
        sample = {
            "image": s["image"].permute(1, 2, 0).numpy(),
            "centerline": s["centerline"].squeeze(0).numpy(),
            "distance_transform": s["distance_transform"].squeeze(0).numpy(),
            "fov_mask": s["fov_mask"].squeeze(0).numpy(),
        }
        if "vessel_orientation" in s:
            sample["vessel_orientation"] = s["vessel_orientation"].numpy()
        if "dt_gradient" in s:
            sample["dt_gradient"] = s["dt_gradient"].numpy()
        # Cache Frangi vesselness once per sample. Otherwise the lazy
        # compute in env.set_data() runs on every episode reset (~0.5s),
        # which dominates wallclock when use_vesselness=True.
        if _use_vesselness:
            from skimage.filters import frangi
            img = sample["image"]
            gray = img[:, :, 1] if img.ndim == 3 else img
            sample["vesselness"] = frangi(
                gray.astype(np.float64),
                sigmas=np.linspace(1.0, 3.0, 5),
                black_ridges=True,
            ).astype(np.float32)
        if _use_unet_prior:
            if "unet_prior" in s:
                sample["unet_prior"] = s["unet_prior"].squeeze(0).numpy()
            else:
                from data.dataloader import compute_unet_prior
                up = compute_unet_prior(sample["image"])
                if up is not None:
                    sample["unet_prior"] = up
        if len(_sample_cache) >= _MAX_CACHE:
            _sample_cache.pop(next(iter(_sample_cache)))
        _sample_cache[idx] = sample
        return sample

    while True:
        cmd, data = conn.recv()

        if cmd == "step":
            result = env.step(data)  # (obs, reward, terminated, truncated, info)
            conn.send(result)

        elif cmd == "set_sample":
            # data is either an int (legacy) or (int, prior_coverage_or_None)
            if isinstance(data, tuple):
                sample_idx, prior_coverage = data
            else:
                sample_idx, prior_coverage = data, None
            s = _get_sample(sample_idx)
            env.set_data(
                image=s["image"],
                centerline=s["centerline"],
                distance_transform=s["distance_transform"],
                fov_mask=s["fov_mask"],
                vessel_orientation=s.get("vessel_orientation"),
                dt_gradient=s.get("dt_gradient"),
                vesselness=s.get("vesselness"),
                unet_prior=s.get("unet_prior"),
                prior_coverage=prior_coverage,
            )
            conn.send(None)

        elif cmd == "get_coverage":
            # Return the covered_centerline mask from the current episode.
            # Called after episode termination to accumulate multi-episode coverage.
            conn.send(
                env.covered_centerline.copy()
                if env.covered_centerline is not None
                else None
            )

        elif cmd == "set_data":
            # Fallback: receive full arrays (used for init if needed)
            env.set_data(**data)
            conn.send(None)

        elif cmd == "reset":
            obs, info = env.reset(**data)
            conn.send(obs)

        elif cmd == "apply_overrides":
            for key, val in data.items():
                if key == "smoothness_weight":
                    env.reward_calculator.smoothness_weight = val
                else:
                    setattr(env, key, val)
            conn.send(None)

        elif cmd == "close":
            conn.close()
            break


class SubprocVecEnv:
    """Run N VesselTracingEnv instances in separate processes."""

    def __init__(self, config: dict, n_envs: int):
        self.n_envs = n_envs
        self.tolerance = config.get("environment", {}).get("tolerance", 2.0)
        ctx = mp.get_context("forkserver")

        self.parent_conns = []
        self.processes = []

        for _ in range(n_envs):
            parent_conn, child_conn = ctx.Pipe()
            p = ctx.Process(
                target=_worker,
                args=(child_conn, config, self.tolerance),
                daemon=True,
            )
            p.start()
            child_conn.close()  # parent doesn't need the child end
            self.parent_conns.append(parent_conn)
            self.processes.append(p)

    def step(self, actions: List[int]):
        """Step all envs in parallel. Returns (obs_list, rewards, terminateds, truncateds, infos)."""
        for conn, action in zip(self.parent_conns, actions):
            conn.send(("step", action))

        results = [conn.recv() for conn in self.parent_conns]
        obs_list = [r[0] for r in results]
        rewards = [r[1] for r in results]
        terminateds = [r[2] for r in results]
        truncateds = [r[3] for r in results]
        infos = [r[4] for r in results]
        return obs_list, rewards, terminateds, truncateds, infos

    def set_sample(
        self,
        idx: int,
        sample_idx: int,
        prior_coverage: Optional[np.ndarray] = None,
    ):
        """Tell a specific env to load a sample by dataset index.

        If ``prior_coverage`` is provided (a (H, W) bool/float32 mask of
        centerline pixels already traced on this image by prior episodes),
        it is forwarded to env.set_data() so the gated connectivity bonus
        can fire during training — closing the train-eval gap where the
        frontier_tracer provides multi-episode context but training does not.
        """
        self.parent_conns[idx].send(("set_sample", (sample_idx, prior_coverage)))
        self.parent_conns[idx].recv()

    def get_coverage_mask(self, idx: int) -> Optional[np.ndarray]:
        """Retrieve the covered_centerline mask from a worker after episode end.

        Returns a (H, W) float32 array (copy) or None if the env has not
        been reset yet.  Used by PPOTrainer to accumulate multi-episode
        coverage per image so that subsequent episodes on the same image
        receive prior_coverage and can earn the gated connectivity bonus.
        """
        self.parent_conns[idx].send(("get_coverage", None))
        return self.parent_conns[idx].recv()

    def set_data(self, idx: int, **kwargs):
        """Set image/centerline/dt data on a specific env (full array transfer)."""
        self.parent_conns[idx].send(("set_data", kwargs))
        self.parent_conns[idx].recv()

    def reset(self, idx: int, **kwargs) -> np.ndarray:
        """Reset a specific env and return its initial observation."""
        self.parent_conns[idx].send(("reset", kwargs))
        return self.parent_conns[idx].recv()

    def apply_overrides(self, idx: int, overrides: dict):
        """Apply curriculum overrides to a specific env."""
        self.parent_conns[idx].send(("apply_overrides", overrides))
        self.parent_conns[idx].recv()

    def close(self):
        """Shut down all worker processes."""
        for conn in self.parent_conns:
            try:
                conn.send(("close", None))
                conn.close()
            except (BrokenPipeError, OSError):
                pass
        for p in self.processes:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
