# evaluation/visualization.py
"""Visualization utilities for vessel tracing.
"""

from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle


class TracingVisualizer:
    """Visualize tracing results."""

    DIRECTION_COLORS = [
        "#FF0000",  # N - red
        "#FF7F00",  # NE - orange
        "#FFFF00",  # E - yellow
        "#7FFF00",  # SE - chartreuse
        "#00FF00",  # S - green
        "#00FF7F",  # SW - spring green
        "#00FFFF",  # W - cyan
        "#007FFF",  # NW - azure
    ]

    def __init__(self, figsize: Tuple[int, int] = (15, 10)):
        self.figsize = figsize

    def visualize_episode(
        self,
        image: np.ndarray,
        gt_centerline: np.ndarray,
        trajectory: List[Tuple[int, int]],
        actions: Optional[List[int]] = None,
        title: str = "",
    ) -> plt.Figure:
        """Visualize a complete tracing episode.

        Args:
            image: RGB image
            gt_centerline: Ground truth centerline
            trajectory: List of (y, x) positions
            actions: Optional list of actions taken
            title: Figure title

        Returns:
            Matplotlib figure

        """
        fig, axes = plt.subplots(1, 3, figsize=self.figsize)

        # Original image with GT centerline
        axes[0].imshow(image)
        axes[0].contour(gt_centerline, colors="blue", linewidths=0.5)
        axes[0].set_title("Image + GT Centerline")
        axes[0].axis("off")

        # Trajectory overlay
        vis = image.copy()
        if vis.max() <= 1.0:
            vis = (vis * 255).astype(np.uint8)

        # Draw GT centerline in blue
        vis[gt_centerline > 0] = [0, 0, 255]

        # Draw trajectory
        for i, (y, x) in enumerate(trajectory):
            color = [255, 0, 0]  # Red
            if actions and i < len(actions):
                # Color by action direction
                color_hex = self.DIRECTION_COLORS[actions[i] % 8]
                color = [
                    int(color_hex[1:3], 16),
                    int(color_hex[3:5], 16),
                    int(color_hex[5:7], 16),
                ]

            cv2.circle(vis, (x, y), 1, color, -1)

        # Mark start and end
        if trajectory:
            cv2.circle(
                vis, (trajectory[0][1], trajectory[0][0]), 5, [0, 255, 0], -1
            )  # Start - green
            cv2.circle(
                vis, (trajectory[-1][1], trajectory[-1][0]), 5, [255, 255, 0], -1
            )  # End - yellow

        axes[1].imshow(vis)
        axes[1].set_title("Trajectory (green=start, yellow=end)")
        axes[1].axis("off")

        # Coverage visualization
        coverage = np.zeros_like(gt_centerline)
        for y, x in trajectory:
            if 0 <= y < coverage.shape[0] and 0 <= x < coverage.shape[1]:
                coverage[
                    max(0, y - 2) : min(coverage.shape[0], y + 3),
                    max(0, x - 2) : min(coverage.shape[1], x + 3),
                ] = 1

        covered = np.logical_and(gt_centerline > 0, coverage > 0)
        missed = np.logical_and(gt_centerline > 0, coverage == 0)
        extra = np.logical_and(gt_centerline == 0, coverage > 0)

        coverage_vis = np.zeros((*gt_centerline.shape, 3), dtype=np.uint8)
        coverage_vis[covered] = [0, 255, 0]  # Green - correctly covered
        coverage_vis[missed] = [255, 0, 0]  # Red - missed
        coverage_vis[extra] = [255, 255, 0]  # Yellow - extra

        axes[2].imshow(coverage_vis)
        axes[2].set_title("Coverage (green=hit, red=miss, yellow=extra)")
        axes[2].axis("off")

        plt.suptitle(title)
        plt.tight_layout()

        return fig

    def visualize_seeds(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        seeds: List[Tuple[int, int, float]],
        gt_endpoints: Optional[List[Tuple[int, int]]] = None,
        gt_junctions: Optional[List[Tuple[int, int]]] = None,
    ) -> plt.Figure:
        """Visualize seed detection results.

        Args:
            image: RGB image
            heatmap: Predicted seed heatmap
            seeds: Detected seeds [(y, x, confidence), ...]
            gt_endpoints: Optional GT endpoints
            gt_junctions: Optional GT junctions

        Returns:
            Matplotlib figure

        """
        fig, axes = plt.subplots(1, 3, figsize=self.figsize)

        # Original image
        axes[0].imshow(image)
        axes[0].set_title("Input Image")
        axes[0].axis("off")

        # Heatmap
        axes[1].imshow(heatmap, cmap="hot")
        axes[1].set_title("Seed Heatmap")
        axes[1].axis("off")

        # Detected seeds
        axes[2].imshow(image)

        for y, x, conf in seeds:
            color = plt.cm.Greens(conf)
            circle = Circle((x, y), radius=5, color=color, fill=False, linewidth=2)
            axes[2].add_patch(circle)

        if gt_endpoints:
            for y, x in gt_endpoints:
                axes[2].scatter(x, y, c="blue", marker="^", s=50, label="GT Endpoint")

        if gt_junctions:
            for y, x in gt_junctions:
                axes[2].scatter(x, y, c="red", marker="s", s=50, label="GT Junction")

        axes[2].set_title(f"Detected Seeds (n={len(seeds)})")
        axes[2].axis("off")

        plt.tight_layout()
        return fig

    def plot_training_history(
        self, history: Dict[str, List], save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot training history.

        Args:
            history: Dictionary of metric lists
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure

        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Losses
        if "policy_loss" in history:
            axes[0, 0].plot(history["policy_loss"], label="Policy Loss")
        if "value_loss" in history:
            axes[0, 0].plot(history["value_loss"], label="Value Loss")
        axes[0, 0].set_xlabel("Update")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()
        axes[0, 0].set_title("Training Losses")

        # Entropy
        if "entropy" in history:
            axes[0, 1].plot(history["entropy"])
            axes[0, 1].set_xlabel("Update")
            axes[0, 1].set_ylabel("Entropy")
            axes[0, 1].set_title("Policy Entropy")

        # Episode rewards
        if "episode_reward" in history:
            axes[1, 0].plot(history["episode_reward"])
            axes[1, 0].set_xlabel("Update")
            axes[1, 0].set_ylabel("Reward")
            axes[1, 0].set_title("Episode Reward")

        # Coverage
        if "coverage_ratio" in history:
            axes[1, 1].plot(history["coverage_ratio"])
            axes[1, 1].set_xlabel("Update")
            axes[1, 1].set_ylabel("Coverage")
            axes[1, 1].set_title("Coverage Ratio")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def create_video(self, frames: List[np.ndarray], output_path: str, fps: int = 10):
        """Create video from frames.

        Args:
            frames: List of frame images
            output_path: Output video path
            fps: Frames per second

        """
        if len(frames) == 0:
            return

        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        for frame in frames:
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame)

        out.release()
