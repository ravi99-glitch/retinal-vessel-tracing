# observation.py
"""Observation construction for vessel tracing environment."""

from typing import Any, Dict, Optional

import numpy as np


class ObservationBuilder:
    """Builds observation tensors for the RL agent.

    Base channels (always present):
    0-2 : RGB crop
    3   : visited mask crop
    4   : distance transform crop, normalised to [0, 1]
    5   : vessel gradient dy (from DT), normalised to [-1, 1]
    6   : vessel gradient dx (from DT), normalised to [-1, 1]
    7   : centerline binary mask
    8   : vessel tangent dy (along-vessel direction)
    9   : vessel tangent dx (along-vessel direction)

    Optional channels (in this order, gated by config flags):
        curvature   — magnitude of the gradient of the vessel-tangent field;
                      peaks at bends. Enabled by `use_curvature`.
        junction    — 1.0 at skeleton junctions, 0.5 at endpoints, 0 elsewhere
                      (dilated so the agent sees decision points before
                      reaching them). Enabled by `use_junction`.
        vesselness  — Frangi vesselness map. Enabled by `use_vesselness`.
        unet_prior  — Frozen Centerline-UNet probability map (full-image
                      vessel context). Enabled by `use_unet_prior`.
        prev_action — Two broadcast channels carrying (last_dy, last_dx) of
                      the most recently taken move. Enabled by `use_prev_action`.

    Channels 5-6 point TOWARD the centerline (perpendicular to vessel).
    Channels 8-9 point ALONG the vessel (tangent direction from structure tensor).
    """

    def __init__(self, config: Dict[str, Any]):
        env_config = config.get("environment", {})
        self.obs_size = env_config.get("observation_size", 65)
        self.half_size = self.obs_size // 2
        self.use_vesselness = env_config.get("use_vesselness", False)
        self.use_curvature = env_config.get("use_curvature", True)
        self.use_junction = env_config.get("use_junction", True)
        self.use_unet_prior = env_config.get("use_unet_prior", False)
        self.use_global_visited = env_config.get("use_global_visited", False)
        self.use_prior_coverage = env_config.get("use_prior_coverage", False)
        self.use_prev_action = env_config.get("use_prev_action", False)
        self.tolerance = env_config.get("tolerance", 2.0)

        # Pre-allocate observation buffer
        self._max_channels = 10
        if self.use_curvature:
            self._max_channels += 1
        if self.use_junction:
            self._max_channels += 1
        if self.use_vesselness:
            self._max_channels += 1
        if self.use_unet_prior:
            self._max_channels += 1
        if self.use_global_visited:
            self._max_channels += 1
        if self.use_prior_coverage:
            self._max_channels += 1
        if self.use_prev_action:
            self._max_channels += 2
        self._obs_buffer = np.zeros(
            (self._max_channels, self.obs_size, self.obs_size), dtype=np.float32
        )
        self._stacked_sources: Optional[np.ndarray] = None  # (H, W, K)
        self._copy_on_build: bool = True  # set False for zero-copy inference
        # Full-image junction/endpoint map (H, W) float32 — set by
        # prepare_stacked_sources() so VesselTracingEnv can look up the value
        # at the agent's current position for junction/endpoint bonuses.
        self.junction_map: Optional[np.ndarray] = None

        # Cached normalised direction lookup for prev-action channels.
        # 8 movement actions (N, NE, E, SE, S, SW, W, NW) — STOP has no direction.
        _RAW = np.array(
            [[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]],
            dtype=np.float32,
        )
        self._action_dy_dx = _RAW / np.linalg.norm(_RAW, axis=1, keepdims=True)

    def prepare_stacked_sources(
        self,
        distance_transform: np.ndarray,
        dt_gradient: np.ndarray,
        centerline: np.ndarray,
        vessel_orientation: np.ndarray,
        unet_prior: Optional[np.ndarray] = None,
    ) -> None:
        """Pre-stack static per-episode maps into one (H, W, K) float32 array.

        Call once per episode in set_data(), not per step.
        Base layout (K=6):
            0=DT  1=grad_y  2=grad_x  3=centerline  4=tangent_y  5=tangent_x
        Optional channels appended after, in order:
            curvature (use_curvature), junction (use_junction),
            unet_prior (use_unet_prior).

        Note: vesselness and prev_action are *not* stacked here — vesselness
        lives outside the stack for historical reasons; prev_action is dynamic.
        """
        H, W = distance_transform.shape[:2]
        # Channel count is driven by config flags only — never by whether the
        # caller happened to supply an optional map. This keeps the observation
        # width in lockstep with models.policy_network._compute_in_channels so
        # fallback paths (e.g. missing UNet checkpoint → unet_prior=None) still
        # produce an obs of the declared shape, just with a zero-filled slot.
        n_extra = (
            int(self.use_curvature)
            + int(self.use_junction)
            + int(self.use_unet_prior)
        )
        s = np.empty((H, W, 6 + n_extra), dtype=np.float32)
        s[:, :, 0] = distance_transform
        s[:, :, 1] = dt_gradient[:, :, 0]
        s[:, :, 2] = dt_gradient[:, :, 1]
        s[:, :, 3] = (centerline > 0).astype(np.float32)
        s[:, :, 4] = vessel_orientation[:, :, 0]
        s[:, :, 5] = vessel_orientation[:, :, 1]
        idx = 6
        if self.use_curvature:
            s[:, :, idx] = self.compute_curvature(vessel_orientation)
            idx += 1
        if self.use_junction:
            jmap = self.compute_junction_map(centerline)
            s[:, :, idx] = jmap
            # Expose full-image map so VesselTracingEnv can read positional values
            self.junction_map = jmap
            idx += 1
        if self.use_unet_prior:
            if unet_prior is not None:
                s[:, :, idx] = unet_prior.astype(np.float32, copy=False)
            else:
                # Predictor unavailable (checkpoint missing) — emit a zero
                # channel so obs dims still match _compute_in_channels.
                s[:, :, idx] = 0.0
            idx += 1
        self._stacked_sources = s

    @staticmethod
    def compute_dt_gradient(distance_transform: np.ndarray) -> np.ndarray:
        """Precompute full-image DT gradient. Call once per episode in set_data().

        Returns (H, W, 2) array of [grad_y, grad_x], negated and normalised
        so vectors point TOWARD the centerline.
        """
        dt = distance_transform.astype(np.float32)
        grad_y, grad_x = np.gradient(dt)
        grad_y, grad_x = -grad_y, -grad_x  # point toward centerline
        mag = np.sqrt(grad_y**2 + grad_x**2) + 1e-8
        grad_y = (grad_y / mag).astype(np.float32)
        grad_x = (grad_x / mag).astype(np.float32)
        return np.stack([grad_y, grad_x], axis=-1)  # (H, W, 2)

    def build(
        self,
        image: np.ndarray,
        visited_mask: np.ndarray,
        vesselness: Optional[np.ndarray],
        position: np.ndarray,
        prev_direction: Optional[int],
        distance_transform: Optional[np.ndarray] = None,
        centerline: Optional[np.ndarray] = None,
        vessel_orientation: Optional[np.ndarray] = None,
        dt_gradient: Optional[np.ndarray] = None,
        unet_prior: Optional[np.ndarray] = None,
        prior_coverage: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        y, x = int(position[0]), int(position[1])
        y_start = y - self.half_size
        y_end = y + self.half_size + 1
        x_start = x - self.half_size
        x_end = x + self.half_size + 1

        buf = self._obs_buffer

        # --- RGB (channels 0-2) ---
        image_crop = self._crop(image, y_start, y_end, x_start, x_end)
        buf[0:3] = image_crop.transpose(2, 0, 1)

        # --- Visited mask (channel 3) ---
        buf[3] = self._crop(visited_mask, y_start, y_end, x_start, x_end)

        # --- Static channels (DT, grads, centerline, tangent, [curv], [junc], [unet]) ---
        if self._stacked_sources is not None:
            n_static = self._stacked_sources.shape[2]
            static_crop = self._crop(
                self._stacked_sources, y_start, y_end, x_start, x_end
            )  # (obs, obs, n_static)
            buf[4 : 4 + n_static] = static_crop.transpose(2, 0, 1)
            # Normalise DT channel (always at static index 0) in-place
            buf[4] /= max(self.tolerance, 1e-6)
            np.clip(buf[4], 0.0, 1.0, out=buf[4])
            n = 4 + n_static
        else:
            # Fallback when prepare_stacked_sources() was not called.
            # Zero everything past the RGB+visited channels so optional
            # slots (curvature/junction/vesselness/unet) don't leak stale data.
            buf[4:] = 0
            if distance_transform is not None:
                dt_crop = self._crop(
                    distance_transform, y_start, y_end, x_start, x_end
                ).astype(np.float32)
                dt_crop /= max(self.tolerance, 1e-6)
                np.clip(dt_crop, 0.0, 1.0, out=dt_crop)
                buf[4] = dt_crop
                if dt_gradient is not None:
                    grad_crop = self._crop(dt_gradient, y_start, y_end, x_start, x_end)
                    buf[5] = grad_crop[:, :, 0]
                    buf[6] = grad_crop[:, :, 1]
                else:
                    raw_dt = self._crop(
                        distance_transform, y_start, y_end, x_start, x_end
                    ).astype(np.float32)
                    gy, gx = np.gradient(raw_dt)
                    gy, gx = -gy, -gx
                    mag = np.sqrt(gy**2 + gx**2) + 1e-8
                    buf[5] = gy / mag
                    buf[6] = gx / mag
            if centerline is not None:
                buf[7] = (
                    self._crop(centerline, y_start, y_end, x_start, x_end) > 0
                ).astype(np.float32)
            if vessel_orientation is not None:
                orient_crop = self._crop(
                    vessel_orientation, y_start, y_end, x_start, x_end
                )
                buf[8] = orient_crop[:, :, 0]
                buf[9] = orient_crop[:, :, 1]
            n = 10  # fallback path doesn't compute curvature/junction/unet

        # --- Vesselness (optional) ---
        if self.use_vesselness and vesselness is not None:
            buf[n] = self._crop(vesselness, y_start, y_end, x_start, x_end)
            n += 1

        # --- UNet prior fallback (only used when stacked sources are absent) ---
        # Always advance n when the flag is on so the channel count matches
        # _compute_in_channels even if unet_prior was not provided (e.g. missing
        # checkpoint); zero-fill the slot in that case.
        if self.use_unet_prior and self._stacked_sources is None:
            if unet_prior is not None:
                buf[n] = self._crop(unet_prior, y_start, y_end, x_start, x_end)
            else:
                buf[n] = 0.0
            n += 1

        # --- Global downsampled visited mask ---
        # Nearest-neighbor downsample of the full visited_mask to obs_size×obs_size.
        # Gives the agent a global view of where it has been (loop avoidance,
        # branch coverage awareness) that the local crop cannot provide.
        if self.use_global_visited:
            H, W = visited_mask.shape
            iy = (np.arange(self.obs_size) * H / self.obs_size).astype(np.intp)
            ix = (np.arange(self.obs_size) * W / self.obs_size).astype(np.intp)
            buf[n] = visited_mask[np.ix_(iy, ix)]
            n += 1

        # --- Prior coverage channel ---
        # Downsampled mask of all pixels covered by PREVIOUS traces in the
        # same inference run.  Tells the agent what has already been traced so
        # it can avoid re-tracing and actively seek uncovered subtrees.
        # Zero during single-episode training (prior_coverage=None).
        if self.use_prior_coverage:
            if prior_coverage is not None:
                H, W = prior_coverage.shape
                iy = (np.arange(self.obs_size) * H / self.obs_size).astype(np.intp)
                ix = (np.arange(self.obs_size) * W / self.obs_size).astype(np.intp)
                buf[n] = prior_coverage[np.ix_(iy, ix)]
            else:
                buf[n] = 0.0
            n += 1

        # --- Previous-action channels (dynamic, broadcast) ---
        if self.use_prev_action:
            if prev_direction is not None and 0 <= prev_direction < 8:
                dy, dx = self._action_dy_dx[prev_direction]
            else:
                dy, dx = 0.0, 0.0
            buf[n].fill(dy)
            buf[n + 1].fill(dx)
            n += 2

        # Copy out — buffer is reused across calls.
        # Callers that consume the observation immediately (e.g. inference)
        # can pass copy=False to avoid the allocation.
        if self._copy_on_build:
            return buf[:n].copy()
        return buf[:n]

    def _crop(
        self, array: np.ndarray, y_start: int, y_end: int, x_start: int, x_end: int
    ) -> np.ndarray:
        """Extract a crop with zero-padding at boundaries."""
        h, w = array.shape[:2]

        pad_top = max(0, -y_start)
        pad_bottom = max(0, y_end - h)
        pad_left = max(0, -x_start)
        pad_right = max(0, x_end - w)

        ys = max(0, y_start)
        ye = min(h, y_end)
        xs = max(0, x_start)
        xe = min(w, x_end)

        crop = array[ys:ye, xs:xe]

        if pad_top or pad_bottom or pad_left or pad_right:
            pw = ((pad_top, pad_bottom), (pad_left, pad_right))
            if array.ndim == 3:
                pw = pw + ((0, 0),)
            crop = np.pad(crop, pw, mode="constant", constant_values=0)

        return crop

    @staticmethod
    def compute_curvature(vessel_orientation: np.ndarray) -> np.ndarray:
        """Per-pixel curvature derived from the vessel-tangent field.

        The structure-tensor tangent already encodes vessel direction
        everywhere; the magnitude of its spatial gradient is a smooth
        proxy for local curvature (peaks at bends, ~0 on straight
        segments). Returns (H, W) float32 in roughly [0, 1].
        """
        ty = vessel_orientation[:, :, 0].astype(np.float32)
        tx = vessel_orientation[:, :, 1].astype(np.float32)
        gy_y, gy_x = np.gradient(ty)
        gx_y, gx_x = np.gradient(tx)
        curv = np.sqrt(gy_y ** 2 + gy_x ** 2 + gx_y ** 2 + gx_x ** 2)
        # Normalise: tangent components live in [-1, 1] so the gradient
        # magnitude is bounded; clip to [0, 1] for a stable input range.
        return np.clip(curv, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def compute_junction_map(
        centerline: np.ndarray, dilation_radius: int = 3
    ) -> np.ndarray:
        """Mark skeleton junctions and endpoints, dilated for visibility.

        Per centerline pixel, count 8-neighbours on the skeleton:
            >= 3 neighbours → junction (1.0)
            == 1 neighbour  → endpoint (0.5)
            else            → 0.0

        The result is dilated by ``dilation_radius`` so the agent can see
        an upcoming junction from a few pixels away. Returns (H, W) float32.
        """
        skel = (centerline > 0).astype(np.uint8)
        if skel.sum() == 0:
            return np.zeros_like(skel, dtype=np.float32)

        # 3×3 sum of skeleton minus the centre = neighbour count
        from scipy.ndimage import (
            convolve,
            grey_dilation,
        )

        kernel = np.ones((3, 3), dtype=np.uint8)
        nbr_count = convolve(skel, kernel, mode="constant", cval=0) - skel

        marker = np.zeros_like(skel, dtype=np.float32)
        marker[(skel > 0) & (nbr_count >= 3)] = 1.0
        marker[(skel > 0) & (nbr_count == 1)] = 0.5

        if dilation_radius > 0:
            size = 2 * dilation_radius + 1
            marker = grey_dilation(marker, size=(size, size))

        return marker.astype(np.float32)

    @staticmethod
    def compute_vessel_orientation(image: np.ndarray) -> np.ndarray:
        """Precompute vessel tangent direction from the image structure tensor.

        Uses the green channel (best vessel contrast in fundus images).
        Returns (H, W, 2) array of [tangent_y, tangent_x], normalised.

        Should be called once per image (in env.set_data), not per step.
        """
        # Use green channel for best vessel contrast
        if image.ndim == 3:
            gray = image[:, :, 1].astype(np.float64)
        else:
            gray = image.astype(np.float64)

        # Image gradients
        iy = np.gradient(gray, axis=0)
        ix = np.gradient(gray, axis=1)

        # Structure tensor components (Gaussian-weighted local averages)
        from scipy.ndimage import gaussian_filter

        sigma = 3.0  # integration scale — ~vessel width
        j_xx = gaussian_filter(ix * ix, sigma)
        j_xy = gaussian_filter(ix * iy, sigma)
        j_yy = gaussian_filter(iy * iy, sigma)

        # Eigendecomposition: smallest eigenvector = vessel tangent
        # For 2x2 symmetric matrix, analytic solution:
        # θ = 0.5 * atan2(2*Jxy, Jxx - Jyy)  gives the dominant orientation
        # The perpendicular direction (vessel tangent) is θ + π/2
        theta = 0.5 * np.arctan2(2.0 * j_xy, j_xx - j_yy + 1e-10)

        # Dominant eigenvector direction (perpendicular to vessel)
        # Rotate 90° to get vessel tangent
        tangent_y = -np.sin(theta).astype(np.float32)  # rotated by 90°
        tangent_x = np.cos(theta).astype(np.float32)

        orientation = np.stack([tangent_y, tangent_x], axis=-1)  # (H, W, 2)
        return orientation
