"""Observation construction for vessel tracing environment."""

from typing import Any, Dict, Optional

import numpy as np


class ObservationBuilder:
    """Builds observation tensors for the RL agent.

    Channels:
    0-2 : RGB crop
    3   : visited mask crop
    4   : distance transform crop, normalised to [0, 1]
    5   : vessel gradient dy (from DT), normalised to [-1, 1]
    6   : vessel gradient dx (from DT), normalised to [-1, 1]
    7   : vessel tangent dy (along-vessel direction)                     # THE COMPASS
    8   : vessel tangent dx (along-vessel direction)                     # THE COMPASS

    Total: 9 channels

    Channels 5-6 point TOWARD the centerline (perpendicular to vessel).
    Channels 7-8 point ALONG the vessel (tangent direction from structure tensor).
    Together they give the agent full local geometry: where the vessel is,
    which way to approach it, and which way the river flows.
    """

    def __init__(self, config: Dict[str, Any]):
        env_config = config.get("environment", {})
        self.obs_size = env_config.get("observation_size", 65)
        self.half_size = self.obs_size // 2
        self.tolerance = env_config.get("tolerance", 2.0)

        # Pre-allocate observation buffer for extreme speed
        self._max_channels = 9
        self._obs_buffer = np.zeros(
            (self._max_channels, self.obs_size, self.obs_size), dtype=np.float32
        )
        self._stacked_sources: Optional[np.ndarray] = None  # (H, W, 5)

    def prepare_stacked_sources(
        self,
        distance_transform: np.ndarray,
        dt_gradient: np.ndarray,
        vessel_orientation: np.ndarray,
    ) -> None:
        """Pre-stack static per-episode maps into one (H, W, 5) float32 array.

        Call once per episode in set_data(), not per step.
        Layout: 0=DT  1=grad_y  2=grad_x  3=tangent_y  4=tangent_x
        """
        H, W = distance_transform.shape[:2]
        s = np.empty((H, W, 5), dtype=np.float32)
        s[:, :, 0] = distance_transform
        s[:, :, 1] = dt_gradient[:, :, 0]
        s[:, :, 2] = dt_gradient[:, :, 1]
        s[:, :, 3] = vessel_orientation[:, :, 0]
        s[:, :, 4] = vessel_orientation[:, :, 1]
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
        position: np.ndarray,
        prev_direction: Optional[int],
        distance_transform: Optional[np.ndarray] = None,
        vessel_orientation: Optional[np.ndarray] = None,
        dt_gradient: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        y, x = int(position[0]), int(position[1])
        y_start = y - self.half_size
        y_end = y + self.half_size + 1
        x_start = x - self.half_size
        x_end = x + self.half_size + 1

        buf = self._obs_buffer
        n = 9

        # --- RGB (channels 0-2) ---
        image_crop = self._crop(image, y_start, y_end, x_start, x_end)
        buf[0:3] = image_crop.transpose(2, 0, 1)

        # --- Visited mask (channel 3) ---
        buf[3] = self._crop(visited_mask, y_start, y_end, x_start, x_end)

        # --- Static channels 4-8 via single crop ---
        if self._stacked_sources is not None:
            static_crop = self._crop(
                self._stacked_sources, y_start, y_end, x_start, x_end
            )  # (obs, obs, 5)
            buf[4:9] = static_crop.transpose(2, 0, 1)
            # Normalise DT channel in-place
            buf[4] /= max(self.tolerance, 1e-6)
            np.clip(buf[4], 0.0, 1.0, out=buf[4])
        else:
            # Fallback when prepare_stacked_sources() was not called
            buf[4:9] = 0
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
            
            if vessel_orientation is not None:
                orient_crop = self._crop(
                    vessel_orientation, y_start, y_end, x_start, x_end
                )
                buf[7] = orient_crop[:, :, 0]
                buf[8] = orient_crop[:, :, 1]

        # Copy out — buffer is reused across calls
        return buf[:n].copy()

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
        theta = 0.5 * np.arctan2(2.0 * j_xy, j_xx - j_yy + 1e-10)

        # Dominant eigenvector direction (perpendicular to vessel)
        # Rotate 90° to get vessel tangent
        tangent_y = -np.sin(theta).astype(np.float32)  # rotated by 90°
        tangent_x = np.cos(theta).astype(np.float32)

        orientation = np.stack([tangent_y, tangent_x], axis=-1)  # (H, W, 2)
        return orientation
