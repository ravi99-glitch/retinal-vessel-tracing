from typing import List, Optional, Tuple, Union

import cv2
import numpy as np


class FundusPreprocessor:
    """Preprocessing pipeline for blood vessel centerline extraction:
    1. Green channel extraction
    2. Gamma Correction
    3. Median Blur (Denoising)
    4. FOV mask (external or internally created)
    5. Apply mask BEFORE CLAHE → prevents FOV border being detected as vessel
    6. CLAHE + ROI-based normalization to 0-1
    """

    def __init__(
        self,
        clahe_clip_limit: float = 2.5,
        clahe_tile_size: int = 8,
        gamma: float = 0.8,
        median_kernel: int = 3,
    ):

        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_size = clahe_tile_size
        self.gamma = gamma
        self.median_kernel = median_kernel
        self.clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=(self.clahe_tile_size, self.clahe_tile_size),
        )

    # --------------------------------------------------
    # CHANNEL EXTRACTION
    # --------------------------------------------------
    def extract_green_channel(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 2:
            return image
        return image[:, :, 1]

    # --------------------------------------------------
    # GAMMA CORRECTION
    # --------------------------------------------------
    def apply_gamma_correction(self, image: np.ndarray) -> np.ndarray:
        # Safety cast: If the image is float (0-1), scale it back to 0-255 uint8
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

        invGamma = 1.0 / self.gamma
        table = np.array(
            [((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]
        ).astype("uint8")
        return cv2.LUT(image, table)

    # --------------------------------------------------
    # NOISE REDUCTION
    # --------------------------------------------------
    def apply_median_blur(self, image: np.ndarray) -> np.ndarray:
        """Removes salt-and-pepper noise while preserving vessel edges."""
        if self.median_kernel > 0:
            return cv2.medianBlur(image, self.median_kernel)
        return image

    # --------------------------------------------------
    # DYNAMIC SCALING
    # --------------------------------------------------
    def _get_dynamic_kernel_size(self, image: np.ndarray, base_size: int = 5) -> int:
        """Scales kernel sizes based on image resolution (normalized to DRIVE)."""
        diag = np.sqrt(image.shape[0] ** 2 + image.shape[1] ** 2)
        scale = diag / 812.0  # DRIVE diagonal is approx 812px
        return int(max(1, round(base_size * scale)))

    # --------------------------------------------------
    # INTERNAL FOV CREATION
    # --------------------------------------------------
    def create_fov_mask(
        self,
        image: np.ndarray,
        block_size: int = 51,
        C: int = 10,
        erosion_size: Optional[int] = None,
    ) -> np.ndarray:
        """Create FOV mask with robust fallback for challenging images."""
        
        # Ensure uint8
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        # Apply slight blur to reduce noise
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        
        # Try adaptive thresholding first
        try:
            binary = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, -C
            )
        except cv2.error:
            # Fallback to Otsu if adaptive fails
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # If adaptive threshold produces too little/too much, fallback to Otsu
        coverage = binary.sum() / (binary.size * 255)
        if coverage < 0.05 or coverage > 0.95:
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        kernel_size = self._get_dynamic_kernel_size(image, 5)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )

        mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest = max(contours, key=cv2.contourArea)
            mask = np.zeros_like(mask)
            cv2.drawContours(mask, [largest], -1, 255, -1)
        else:
            # CRITICAL FALLBACK: No contours found, create circular mask
            h, w = image.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            center = (w // 2, h // 2)
            radius = int(min(h, w) * 0.45)  # Conservative 90% diameter
            cv2.circle(mask, center, radius, 255, -1)
        
        # Final check: if mask is still too small, replace with circular fallback
        final_coverage = mask.sum() / (mask.size * 255)
        if final_coverage < 0.10:  # Less than 10% of image
            h, w = image.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            center = (w // 2, h // 2)
            radius = int(min(h, w) * 0.45)
            cv2.circle(mask, center, radius, 255, -1)

        e_size = erosion_size if erosion_size is not None else kernel_size
        if e_size > 0:
            erosion_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (e_size * 2 + 1, e_size * 2 + 1)
            )
            mask = cv2.erode(mask, erosion_kernel, iterations=1)

        return mask


    # def create_fov_mask(
    #     self,
    #     image: np.ndarray,
    #     block_size: int = 51,
    #     C: int = 10,
    #     erosion_size: Optional[int] = None,
    # ) -> np.ndarray:

    #     binary = cv2.adaptiveThreshold(
    #         image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, -C
    #     )

    #     kernel_size = self._get_dynamic_kernel_size(image, 5)
    #     kernel = cv2.getStructuringElement(
    #         cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    #     )

    #     mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    #     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    #     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #     if contours:
    #         largest = max(contours, key=cv2.contourArea)
    #         mask = np.zeros_like(mask)
    #         cv2.drawContours(mask, [largest], -1, 255, -1)

    #     e_size = erosion_size if erosion_size is not None else kernel_size
    #     if e_size > 0:
    #         erosion_kernel = cv2.getStructuringElement(
    #             cv2.MORPH_ELLIPSE, (e_size * 2 + 1, e_size * 2 + 1)
    #         )
    #         mask = cv2.erode(mask, erosion_kernel, iterations=1)

    #     return mask

    # --------------------------------------------------
    # EXTERNAL FOV HANDLING
    # --------------------------------------------------
    def load_external_mask(
        self, mask: np.ndarray, erosion_size: Optional[int] = None
    ) -> np.ndarray:
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # Binarize
        mask = (mask > 128).astype(np.uint8) * 255

        # Eroding external masks ensures we avoid the sharp high-contrast boundary
        e_size = (
            erosion_size
            if erosion_size is not None
            else self._get_dynamic_kernel_size(mask, 5)
        )
        if e_size > 0:
            erosion_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (e_size * 2 + 1, e_size * 2 + 1)
            )
            mask = cv2.erode(mask, erosion_kernel, iterations=1)

        return mask

    # --------------------------------------------------
    # APPLY MASK
    # --------------------------------------------------
    def apply_mask(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return cv2.bitwise_and(image, image, mask=mask)

    # --------------------------------------------------
    # MAIN PIPELINE
    # --------------------------------------------------
    def preprocess(
        self,
        image: np.ndarray,
        external_mask: Optional[np.ndarray] = None,
        return_intermediate: bool = False,
    ) -> Union[np.ndarray, Tuple]:

        # 1. Green channel
        green = self.extract_green_channel(image)

        # 2. Gamma
        gamma_corrected = self.apply_gamma_correction(green)

        # 3. Median Blur (Denoising)
        denoised = self.apply_median_blur(gamma_corrected)

        # 4. Mask selection
        if external_mask is not None:
            mask = self.load_external_mask(external_mask)
        else:
            mask = self.create_fov_mask(gamma_corrected)

        # 5. Apply mask BEFORE CLAHE
        gamma_masked = self.apply_mask(denoised, mask)

        # 6. CLAHE
        clahe_enhanced = self.clahe.apply(gamma_masked)

        # 7. ROI-Aware Normalize to 0–1
        roi_pixels = clahe_enhanced[mask > 0]
        if roi_pixels.size > 0:
            vmin, vmax = np.percentile(roi_pixels, [1.0, 99.0])
            preprocessed = np.clip((clahe_enhanced - vmin) / (vmax - vmin + 1e-8), 0, 1)
        else:
            preprocessed = clahe_enhanced.astype(np.float32) / 255.0

        preprocessed = preprocessed.astype(np.float32)

        if return_intermediate:
            return (preprocessed, green, gamma_corrected, clahe_enhanced, mask)

        return preprocessed

    # --------------------------------------------------
    # BATCH
    # --------------------------------------------------
    def preprocess_batch(
        self, images: List[np.ndarray], masks: Optional[List[np.ndarray]] = None
    ) -> List[np.ndarray]:

        results = []
        if masks is not None:
            for img, m in zip(images, masks):
                results.append(self.preprocess(img, external_mask=m))
        else:
            for img in images:
                results.append(self.preprocess(img))

        return results
