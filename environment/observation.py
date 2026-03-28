# observation.py
"""Observation construction for vessel tracing environment."""

from typing import Any, Dict, Optional

import numpy as np


class ObservationBuilder:
    """Builds observation tensors for the RL agent.

    Channels (use_vesselness=False):
    0-2 : RGB crop
    3   : visited mask crop
    4   : distance transform crop, normalised to [0, 1]
    5   : vessel gradient dy (from DT), normalised to [-1, 1]
    6   : vessel gradient dx (from DT), normalised to [-1, 1]
    7   : centerline binary mask                                  # NEW
    8   : vessel tangent dy (along-vessel direction)               # NEW
    9   : vessel tangent dx (along-vessel direction)               # NEW

    Total: 10 channels (11 with vesselness)

    Channels 5-6 point TOWARD the centerline (perpendicular to vessel).
    Channels 8-9 point ALONG the vessel (tangent direction from structure tensor).
    Together they give the agent full local geometry: where the centerline is,
    which way to approach it, and which way the vessel runs.
    """

    def __init__(self, config: Dict[str, Any]):
        env_config = config.get("environment", {})
        self.obs_size = env_config.get("observation_size", 65)
        self.half_size = self.obs_size // 2
        self.use_vesselness = env_config.get("use_vesselness", False)
        self.tolerance = env_config.get("tolerance", 2.0)

    def build(
        self,
        image: np.ndarray,
        visited_mask: np.ndarray,
        vesselness: Optional[np.ndarray],
        position: np.ndarray,
        prev_direction: Optional[int],  # kept for API compat, unused
        distance_transform: Optional[np.ndarray] = None,
        centerline: Optional[np.ndarray] = None,
        vessel_orientation: Optional[np.ndarray] = None,  # (H,W,2)
    ) -> np.ndarray:
        """Build observation tensor.

        Args:
            image:              Full RGB image (H, W, 3), float32 in [0, 1]
            visited_mask:       Full visited mask (H, W)
            vesselness:         Optional vesselness response (H, W)
            position:           Current position [y, x]
            prev_direction:     Ignored (kept for API compatibility)
            distance_transform: Distance-to-centerline map (H, W) — required

        Returns:
            Observation tensor (C, obs_size, obs_size), float32

        """
        y, x = int(position[0]), int(position[1])

        y_start = y - self.half_size
        y_end = y + self.half_size + 1
        x_start = x - self.half_size
        x_end = x + self.half_size + 1

        # --- RGB (3 channels) ---
        image_crop = self._crop(image, y_start, y_end, x_start, x_end)
        rgb = image_crop.transpose(2, 0, 1).astype(np.float32)  # (3,H,W)

        # --- Visited mask (1 channel) ---
        vis_crop = self._crop(
            visited_mask[:, :, np.newaxis], y_start, y_end, x_start, x_end
        )[:, :, 0]
        visited_ch = vis_crop[np.newaxis].astype(np.float32)  # (1,H,W)

        channels = [rgb, visited_ch]

        # --- Distance transform + vessel gradient (3 channels) ---
        if distance_transform is not None:
            dt_crop = self._crop(
                distance_transform[:, :, np.newaxis], y_start, y_end, x_start, x_end
            )[:, :, 0].astype(np.float32)

            # Channel 4: normalised DT, 0=on centerline, 1=at tolerance boundary
            dt_norm = np.clip(dt_crop / max(self.tolerance, 1e-6), 0.0, 1.0)
            channels.append(dt_norm[np.newaxis])  # (1,H,W)

            # Channels 5-6: local vessel direction from DT gradient
            # gradient points away from centerline; negate so it points TOWARD it
            grad_y, grad_x = np.gradient(dt_crop)
            grad_y = -grad_y
            grad_x = -grad_x
            mag = np.sqrt(grad_y**2 + grad_x**2) + 1e-8
            grad_y_norm = (grad_y / mag).astype(np.float32)  # [-1, 1]
            grad_x_norm = (grad_x / mag).astype(np.float32)  # [-1, 1]
            channels.append(grad_y_norm[np.newaxis])  # (1,H,W)
            channels.append(grad_x_norm[np.newaxis])  # (1,H,W)
        else:
            # Fallback: three zero channels so shape is always consistent
            zeros = np.zeros((1, self.obs_size, self.obs_size), dtype=np.float32)
            channels += [zeros, zeros, zeros]

        # Centerline binary mask (1 channel)
        if centerline is not None:
            cl_crop = self._crop(
                centerline[:, :, np.newaxis], y_start, y_end, x_start, x_end
            )[:, :, 0]
            cl_ch = (cl_crop > 0).astype(np.float32)
            channels.append(cl_ch[np.newaxis])  # (1,H,W)
        else:
            channels.append(
                np.zeros((1, self.obs_size, self.obs_size), dtype=np.float32)
            )

        # Vessel tangent direction (2 channels)
        if vessel_orientation is not None:
            orient_crop = self._crop(
                vessel_orientation, y_start, y_end, x_start, x_end
            )  # (obs_size, obs_size, 2)
            channels.append(
                orient_crop[:, :, 0][np.newaxis].astype(np.float32)
            )  # tangent_y
            channels.append(
                orient_crop[:, :, 1][np.newaxis].astype(np.float32)
            )  # tangent_x
        else:
            zeros = np.zeros((1, self.obs_size, self.obs_size), dtype=np.float32)
            channels.append(zeros)
            channels.append(zeros)

        # --- Vesselness (1 channel, optional) ---
        if self.use_vesselness and vesselness is not None:
            v_crop = self._crop(
                vesselness[:, :, np.newaxis], y_start, y_end, x_start, x_end
            )[:, :, 0]
            channels.append(v_crop[np.newaxis].astype(np.float32))  # (1,H,W)

        return np.concatenate(channels, axis=0)  # (C,H,W)

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

    def compute_vessel_orientation(self, image: np.ndarray) -> np.ndarray:
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
