"""Spectral preprocessing for MS/MS inference pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch

LOGGER = logging.getLogger(__name__)


@dataclass
class SpectrumPreprocessorConfig:
    """Configuration for spectrum preprocessing."""

    mz_min: float = 0.0
    mz_max: float = 2000.0
    mz_bin_size: float = 0.1
    max_peaks: int = 1000
    entropy_threshold: float = 0.5
    normalize_intensity: bool = True
    apply_augmentation: bool = False


class SpectrumPreprocessor:
    """Preprocesses raw MS/MS spectra for model inference."""

    def __init__(self, config: Optional[SpectrumPreprocessorConfig] = None):
        """Initialize preprocessor with configuration."""
        self.config = config or SpectrumPreprocessorConfig()
        self._mz_bins = np.arange(
            self.config.mz_min,
            self.config.mz_max + self.config.mz_bin_size,
            self.config.mz_bin_size,
        )

    def preprocess(
        self,
        mz_values: np.ndarray,
        intensity_values: np.ndarray,
        precursor_mz: Optional[float] = None,
        retention_time: Optional[float] = None,
    ) -> torch.Tensor:
        """Preprocess a single spectrum into model-ready tensor.

        Args:
            mz_values: Array of m/z values
            intensity_values: Array of intensity values
            precursor_mz: Optional precursor m/z value
            retention_time: Optional retention time value

        Returns:
            Tensor of shape (N, D) where N is sequence length and D is feature dim
        """
        if len(mz_values) == 0:
            return self._empty_spectrum()

        mz_values = np.asarray(mz_values, dtype=np.float32)
        intensity_values = np.asarray(intensity_values, dtype=np.float32)

        if self.config.normalize_intensity:
            intensity_values = self._normalize_intensity(intensity_values)

        if self._should_filter_by_entropy(mz_values, intensity_values):
            mz_values, intensity_values = self._filter_by_entropy(mz_values, intensity_values)

        binned_spectrum = self._bin_spectrum(mz_values, intensity_values)
        padded_spectrum = self._pad_spectrum(binned_spectrum)

        if self.config.apply_augmentation:
            padded_spectrum = self._apply_augmentation(padded_spectrum)

        return torch.from_numpy(padded_spectrum).float()

    def _normalize_intensity(self, intensity: np.ndarray) -> np.ndarray:
        """Normalize intensity values to [0, 1] range."""
        max_intensity = intensity.max()
        if max_intensity > 0:
            return intensity / max_intensity
        return intensity

    def _should_filter_by_entropy(self, mz_values: np.ndarray, intensity_values: np.ndarray) -> bool:
        """Check if spectrum should be filtered by entropy."""
        if len(intensity_values) < 10:
            return False
        normalized = intensity_values / intensity_values.sum()
        entropy = -np.sum(normalized * np.log(normalized + 1e-10))
        return entropy < self.config.entropy_threshold

    def _filter_by_entropy(self, mz_values: np.ndarray, intensity_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Filter spectrum peaks by entropy threshold."""
        if len(intensity_values) < 10:
            return mz_values, intensity_values

        normalized = intensity_values / intensity_values.sum()
        entropy = -np.sum(normalized * np.log(normalized + 1e-10))

        if entropy >= self.config.entropy_threshold:
            return mz_values, intensity_values

        top_k = min(self.config.max_peaks, len(intensity_values))
        top_indices = np.argsort(intensity_values)[-top_k:]
        return mz_values[top_indices], intensity_values[top_indices]

    def _bin_spectrum(self, mz_values: np.ndarray, intensity_values: np.ndarray) -> np.ndarray:
        """Bin spectrum into fixed m/z bins."""
        binned = np.zeros(len(self._mz_bins) - 1, dtype=np.float32)
        bin_indices = np.digitize(mz_values, self._mz_bins) - 1
        bin_indices = np.clip(bin_indices, 0, len(binned) - 1)

        for idx, intensity in zip(bin_indices, intensity_values):
            binned[idx] += intensity

        return binned

    def _pad_spectrum(self, spectrum: np.ndarray) -> np.ndarray:
        """Pad or truncate spectrum to fixed length."""
        target_length = len(self._mz_bins) - 1
        if len(spectrum) < target_length:
            padded = np.zeros(target_length, dtype=np.float32)
            padded[: len(spectrum)] = spectrum
            return padded
        return spectrum[:target_length]

    def _apply_augmentation(self, spectrum: np.ndarray) -> np.ndarray:
        """Apply data augmentation (disabled in inference mode)."""
        return spectrum

    def _empty_spectrum(self) -> torch.Tensor:
        """Return empty spectrum tensor."""
        target_length = len(self._mz_bins) - 1
        return torch.zeros(target_length, dtype=torch.float32)

    def batch_preprocess(
        self,
        spectra: list[tuple[np.ndarray, np.ndarray]] | list[dict[str, Any]],
        precursor_mzs: Optional[list[float]] = None,
        retention_times: Optional[list[float]] = None,
    ) -> torch.Tensor:
        """Preprocess a batch of spectra.

        Args:
            spectra: List of (mz_values, intensity_values) tuples or dicts with 'mz'/'intensity' keys
            precursor_mzs: Optional list of precursor m/z values
            retention_times: Optional list of retention times

        Returns:
            Tensor of shape (B, N, D) where B is batch size
        """
        batch_tensors = []
        for i, spectrum in enumerate(spectra):
            if isinstance(spectrum, dict):
                mz_values = np.array(spectrum.get("mz", []))
                intensity_values = np.array(spectrum.get("intensity", []))
                precursor_mz = spectrum.get("precursor_mz") or (precursor_mzs[i] if precursor_mzs else None)
                rt = spectrum.get("retention_time") or (retention_times[i] if retention_times else None)
            else:
                mz_values, intensity_values = spectrum
                precursor_mz = precursor_mzs[i] if precursor_mzs else None
                rt = retention_times[i] if retention_times else None

            tensor = self.preprocess(mz_values, intensity_values, precursor_mz, rt)
            batch_tensors.append(tensor)

        return torch.stack(batch_tensors)

    def __call__(self, raw_spectrum: dict[str, Any]) -> torch.Tensor:
        """Callable interface for dictionary-based spectrum input.

        Args:
            raw_spectrum: Dictionary with 'mz' and 'intensity' keys

        Returns:
            Preprocessed tensor
        """
        mz_values = np.array(raw_spectrum.get("mz", []))
        intensity_values = np.array(raw_spectrum.get("intensity", []))
        precursor_mz = raw_spectrum.get("precursor_mz")
        retention_time = raw_spectrum.get("retention_time")
        return self.preprocess(mz_values, intensity_values, precursor_mz, retention_time)

