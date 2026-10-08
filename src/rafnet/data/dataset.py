import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Callable, Union
import numpy as np
from dataclasses import dataclass, asdict
import os
from pathlib import Path

from .manifest import Manifest, SampleManifest, check_leakage, create_synthetic_manifest


@dataclass
class AudioWindowConfig:
    window_size: int = 16000
    hop_size: int = 8000
    min_window_size: int = 8000
    pad_mode: str = "constant"


@dataclass
class NetworkWindowConfig:
    window_size: int = 10
    hop_size: int = 5
    min_window_size: int = 5


class Normalizer:
    def __init__(self):
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.fitted = False
    
    def fit(self, data: np.ndarray) -> "Normalizer":
        self.mean = data.mean(axis=0)
        self.std = data.std(axis=0)
        self.std[self.std < 1e-8] = 1.0
        self.fitted = True
        return self
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")
        return (data - self.mean) / self.std
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        return self.fit(data).transform(data)
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Normalizer not fitted.")
        return data * self.std + self.mean
    
    def save(self, path: str) -> None:
        np.savez(path, mean=self.mean, std=self.std, fitted=self.fitted)
    
    @classmethod
    def load(cls, path: str) -> "Normalizer":
        normalizer = cls()
        data = np.load(path)
        normalizer.mean = data["mean"]
        normalizer.std = data["std"]
        normalizer.fitted = data["fitted"].item()
        return normalizer


class NetworkTelemetryEncoder:
    KNOWN_CODECS = ["opus", "g722", "pcmu", "pcma", "g729", "amr", "silk"]
    
    def __init__(self, network_dim: int = 32, max_packets: int = 100):
        self.network_dim = network_dim
        self.max_packets = max_packets
        self.codec_to_idx = {codec: i for i, codec in enumerate(self.KNOWN_CODECS)}
        self.codec_unk_idx = len(self.KNOWN_CODECS)
        
    def encode_codec(self, codec: str) -> Tuple[int, bool]:
        codec_lower = codec.lower() if codec else "unknown"
        if codec_lower in self.codec_to_idx:
            return self.codec_to_idx[codec_lower], True
        return self.codec_unk_idx, False
    
    def compute_packet_timing_stats(self, packet_timing: Optional[List[float]]) -> Dict[str, float]:
        if not packet_timing or len(packet_timing) < 2:
            return {
                "mean_iat": 0.0,
                "std_iat": 0.0,
                "min_iat": 0.0,
                "max_iat": 0.0,
                "cv_iat": 0.0,
            }
        
        iats = np.diff(packet_timing)
        mean_iat = float(np.mean(iats))
        std_iat = float(np.std(iats))
        
        return {
            "mean_iat": mean_iat,
            "std_iat": std_iat,
            "min_iat": float(np.min(iats)),
            "max_iat": float(np.max(iats)),
            "cv_iat": std_iat / mean_iat if mean_iat > 0 else 0.0,
        }
    
    def compute_packet_size_stats(self, packet_size: Optional[List[float]]) -> Dict[str, float]:
        if not packet_size or len(packet_size) == 0:
            return {
                "mean_size": 0.0,
                "std_size": 0.0,
                "min_size": 0.0,
                "max_size": 0.0,
            }
        
        sizes = np.array(packet_size)
        return {
            "mean_size": float(np.mean(sizes)),
            "std_size": float(np.std(sizes)),
            "min_size": float(np.min(sizes)),
            "max_size": float(np.max(sizes)),
        }
    
    def extract_features(self, sample: SampleManifest) -> Tuple[np.ndarray, np.ndarray]:
        features = []
        mask = []
        
        # Core network features (always present)
        features.append(sample.packet_loss)
        mask.append(1.0)
        
        features.append(sample.jitter)
        mask.append(1.0)
        
        features.append(sample.rtt)
        mask.append(1.0)
        
        features.append(sample.bitrate / 100000.0)
        mask.append(1.0)
        
        # Packet timing statistics
        timing_stats = self.compute_packet_timing_stats(sample.packet_timing)
        features.extend([
            timing_stats["mean_iat"],
            timing_stats["std_iat"],
            timing_stats["cv_iat"],
        ])
        has_timing = sample.packet_timing is not None and len(sample.packet_timing) >= 2
        mask.extend([1.0 if has_timing else 0.0] * 3)
        
        # Packet size statistics
        size_stats = self.compute_packet_size_stats(sample.packet_size)
        features.extend([
            size_stats["mean_size"] / 1500.0,
            size_stats["std_size"] / 1500.0,
        ])
        has_size = sample.packet_size is not None and len(sample.packet_size) > 0
        mask.extend([1.0 if has_size else 0.0] * 2)
        
        # Burst loss statistics
        features.append(float(sample.burst_loss_count or 0) / 10.0)
        features.append(float(sample.burst_loss_duration or 0.0))
        features.append(float(sample.loss_burstiness or 0.0))
        has_burst = sample.burst_loss_count is not None
        mask.extend([1.0 if has_burst else 0.0] * 3)
        
        # Codec embedding (one-hot)
        codec_idx, is_known = self.encode_codec(sample.codec)
        codec_onehot = np.zeros(len(self.KNOWN_CODECS) + 1, dtype=np.float32)
        codec_onehot[codec_idx] = 1.0
        features.extend(codec_onehot.tolist())
        mask.extend([1.0 if is_known else 0.0] * (len(self.KNOWN_CODECS) + 1))
        
        features = np.array(features, dtype=np.float32)
        mask = np.array(mask, dtype=np.float32)
        
        # Pad or truncate to network_dim
        if len(features) < self.network_dim:
            padding = np.zeros(self.network_dim - len(features), dtype=np.float32)
            mask_padding = np.zeros(self.network_dim - len(mask), dtype=np.float32)
            features = np.concatenate([features, padding])
            mask = np.concatenate([mask, mask_padding])
        elif len(features) > self.network_dim:
            features = features[:self.network_dim]
            mask = mask[:self.network_dim]
        
        return features, mask
    
    def get_feature_names(self) -> List[str]:
        names = [
            "packet_loss", "jitter", "rtt", "bitrate_norm",
            "mean_iat", "std_iat", "cv_iat",
            "mean_size_norm", "std_size_norm",
            "burst_count_norm", "burst_duration", "loss_burstiness",
        ]
        names.extend([f"codec_{c}" for c in self.KNOWN_CODECS])
        names.append("codec_unknown")
        return names[:self.network_dim]


class AudioNetworkAligner:
    def __init__(
        self,
        sample_rate: int = 16000,
        network_fps: int = 50,
        audio_window: AudioWindowConfig = None,
        network_window: NetworkWindowConfig = None,
    ):
        self.sample_rate = sample_rate
        self.network_fps = network_fps
        self.audio_window = audio_window or AudioWindowConfig()
        self.network_window = network_window or NetworkWindowConfig()
    
    def align(self, audio: torch.Tensor, network: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        audio_duration = audio.shape[-1] / self.sample_rate
        network_duration = network.shape[0] / self.network_fps
        
        min_duration = min(audio_duration, network_duration)
        
        audio_samples = int(min_duration * self.sample_rate)
        network_frames = int(min_duration * self.network_fps)
        
        audio = audio[..., :audio_samples]
        network = network[:network_frames]
        
        return audio, network
    
    def create_windows(
        self,
        audio: torch.Tensor,
        network: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        audio, network = self.align(audio, network)
        
        audio_windows = []
        network_windows = []
        
        audio_step = self.audio_window.hop_size
        audio_win = self.audio_window.window_size
        
        network_step = self.network_window.hop_size
        network_win = self.network_window.window_size
        
        for i in range(0, audio.shape[-1] - audio_win + 1, audio_step):
            audio_win_data = audio[..., i:i + audio_win]
            audio_windows.append(audio_win_data)
        
        for i in range(0, network.shape[0] - network_win + 1, network_step):
            network_win_data = network[i:i + network_win]
            network_windows.append(network_win_data)
        
        min_windows = min(len(audio_windows), len(network_windows))
        return audio_windows[:min_windows], network_windows[:min_windows]


class SyntheticVoiceDataset(Dataset):
    def __init__(
        self,
        manifest: Manifest,
        audio_root: str = "",
        sample_rate: int = 16000,
        network_dim: int = 32,
        network_normalizer: Optional[Normalizer] = None,
        audio_transform: Optional[Callable] = None,
        network_transform: Optional[Callable] = None,
        aligner: Optional[AudioNetworkAligner] = None,
        return_windows: bool = False,
        max_samples: Optional[int] = None,
        telemetry_encoder: Optional["NetworkTelemetryEncoder"] = None,
    ):
        self.manifest = manifest
        self.audio_root = audio_root
        self.sample_rate = sample_rate
        self.network_dim = network_dim
        self.network_normalizer = network_normalizer
        self.audio_transform = audio_transform
        self.network_transform = network_transform
        self.aligner = aligner or AudioNetworkAligner(sample_rate=sample_rate)
        self.return_windows = return_windows
        self.max_samples = max_samples
        self.telemetry_encoder = telemetry_encoder or NetworkTelemetryEncoder(network_dim=network_dim)
        
        self.samples = manifest.samples[:max_samples] if max_samples else manifest.samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        audio_path = Path(self.audio_root) / sample.audio_path
        
        if audio_path.exists():
            audio, sr = self._load_audio(audio_path)
            if sr != self.sample_rate:
                audio = self._resample(audio, sr, self.sample_rate)
        else:
            audio = torch.randn(self.sample_rate)
        
        network_features, network_mask = self.telemetry_encoder.extract_features(sample)
        
        network_features = torch.from_numpy(network_features).float()
        network_mask = torch.from_numpy(network_mask).float()
        
        if self.network_normalizer is not None:
            network_features = torch.from_numpy(
                self.network_normalizer.transform(network_features.numpy().reshape(1, -1))
            ).float().squeeze(0)
        
        if self.return_windows:
            audio_windows, network_windows = self.aligner.create_windows(audio, network_features)
            audio = torch.stack(audio_windows)
            network_features = torch.stack(network_windows)
            # For windows, we'll need to handle masks appropriately
            network_mask = network_mask.unsqueeze(0).expand(len(network_windows), -1)
        
        if self.audio_transform:
            audio = self.audio_transform(audio)
        if self.network_transform:
            network_features = self.network_transform(network_features)
        
        item = {
            "sample_id": sample.sample_id,
            "audio": audio,
            "network": network_features,
            "network_mask": network_mask,
            "label": torch.tensor(sample.class_label, dtype=torch.long),
            "speaker_id": sample.speaker_id,
            "generator_id": sample.generator_id,
            "condition_id": sample.communication_condition_id,
            "codec": sample.codec,
            "packet_loss": sample.packet_loss,
            "jitter": sample.jitter,
            "rtt": sample.rtt,
            "bitrate": sample.bitrate,
        }
        
        return item
    
    def _load_audio(self, path: Path) -> Tuple[torch.Tensor, int]:
        import torchaudio
        waveform, sr = torchaudio.load(str(path))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        return waveform.squeeze(0), sr
    
    def _resample(self, audio: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
        import torchaudio.transforms as T
        if orig_sr == target_sr:
            return audio
        resampler = T.Resample(orig_sr, target_sr)
        return resampler(audio)
    
    def _generate_network_features(self, sample: SampleManifest) -> torch.Tensor:
        features = np.array([
            sample.packet_loss,
            sample.jitter,
            sample.rtt,
            sample.bitrate / 100000.0,
        ], dtype=np.float32)
        
        if len(features) < self.network_dim:
            padding = np.zeros(self.network_dim - len(features), dtype=np.float32)
            features = np.concatenate([features, padding])
        elif len(features) > self.network_dim:
            features = features[:self.network_dim]
        
        return torch.from_numpy(features)


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    if not batch:
        return {}
    
    sample_ids = [item["sample_id"] for item in batch]
    labels = torch.stack([item["label"] for item in batch])
    speaker_ids = [item["speaker_id"] for item in batch]
    generator_ids = [item["generator_id"] for item in batch]
    condition_ids = [item["condition_id"] for item in batch]
    codecs = [item["codec"] for item in batch]
    
    audio = batch[0]["audio"]
    network = batch[0]["network"]
    network_mask = batch[0].get("network_mask", torch.ones_like(network))
    
    if audio.dim() == 2 and audio.shape[1] == audio.shape[-1]:
        audio = torch.stack([item["audio"] for item in batch])
    elif audio.dim() == 3:
        audio = torch.stack([item["audio"] for item in batch])
    else:
        audio = torch.stack([item["audio"] for item in batch])
    
    network = torch.stack([item["network"] for item in batch])
    network_mask = torch.stack([item.get("network_mask", torch.ones_like(item["network"])) for item in batch])
    
    result = {
        "sample_ids": sample_ids,
        "audio": audio,
        "network": network,
        "network_mask": network_mask,
        "labels": labels,
        "speaker_ids": speaker_ids,
        "generator_ids": generator_ids,
        "condition_ids": condition_ids,
        "codecs": codecs,
    }
    
    return result


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )


def fit_normalizers(
    train_dataset: SyntheticVoiceDataset,
    val_dataset: Optional[SyntheticVoiceDataset] = None,
    test_dataset: Optional[SyntheticVoiceDataset] = None,
) -> Tuple[Normalizer, Normalizer]:
    audio_normalizer = Normalizer()
    network_normalizer = Normalizer()
    
    all_network = []
    for i in range(len(train_dataset)):
        item = train_dataset[i]
        all_network.append(item["network"].numpy())
    
    if all_network:
        network_normalizer.fit(np.stack(all_network))
    
    train_dataset.network_normalizer = network_normalizer
    if val_dataset is not None:
        val_dataset.network_normalizer = network_normalizer
    if test_dataset is not None:
        test_dataset.network_normalizer = network_normalizer
    
    return audio_normalizer, network_normalizer


def _generate_network_features_legacy(sample: SampleManifest, network_dim: int = 20) -> torch.Tensor:
    """Legacy feature generation for backward compatibility."""
    features = np.array([
        sample.packet_loss,
        sample.jitter,
        sample.rtt,
        sample.bitrate / 100000.0,
    ], dtype=np.float32)
    
    if len(features) < network_dim:
        padding = np.zeros(network_dim - len(features), dtype=np.float32)
        features = np.concatenate([features, padding])
    elif len(features) > network_dim:
        features = features[:network_dim]
    
    return torch.from_numpy(features)


def create_datasets_from_manifest(
    manifest: Manifest,
    audio_root: str = "",
    sample_rate: int = 16000,
    network_dim: int = 20,
    max_samples_per_split: Optional[int] = None,
    return_windows: bool = False,
) -> Tuple[SyntheticVoiceDataset, SyntheticVoiceDataset, SyntheticVoiceDataset]:
    train_manifest = manifest.get_split("train")
    val_manifest = manifest.get_split("val")
    test_manifest = manifest.get_split("test")
    
    train_dataset = SyntheticVoiceDataset(
        train_manifest, audio_root=audio_root, sample_rate=sample_rate,
        network_dim=network_dim, max_samples=max_samples_per_split,
        return_windows=return_windows
    )
    val_dataset = SyntheticVoiceDataset(
        val_manifest, audio_root=audio_root, sample_rate=sample_rate,
        network_dim=network_dim, max_samples=max_samples_per_split,
        return_windows=return_windows
    )
    test_dataset = SyntheticVoiceDataset(
        test_manifest, audio_root=audio_root, sample_rate=sample_rate,
        network_dim=network_dim, max_samples=max_samples_per_split,
        return_windows=return_windows
    )
    
    audio_norm, network_norm = fit_normalizers(train_dataset, val_dataset, test_dataset)
    
    return train_dataset, val_dataset, test_dataset


def detect_dataset_leakage(
    train_manifest: Manifest,
    val_manifest: Manifest,
    test_manifest: Manifest,
) -> Dict[str, any]:
    full_manifest = Manifest(train_manifest.samples + val_manifest.samples + test_manifest.samples)
    return check_leakage(full_manifest)


def create_synthetic_fixture(
    n_samples: int = 100,
    n_speakers: int = 20,
    n_generators: int = 10,
    n_conditions: int = 5,
    seed: int = 42,
) -> Manifest:
    return create_synthetic_manifest(
        n_samples=n_samples,
        n_speakers=n_speakers,
        n_generators=n_generators,
        n_conditions=n_conditions,
        seed=seed,
    )


def generate_synthetic_data(
    n_samples: int = 1000,
    audio_length: int = 16000,
    network_dim: int = 20,
    n_speakers: int = 10,
    n_generators: int = 5,
    n_conditions: int = 3,
    seed: int = 42,
):
    from .manifest import create_synthetic_manifest
    manifest = create_synthetic_manifest(
        n_samples=n_samples,
        n_speakers=n_speakers,
        n_generators=n_generators,
        n_conditions=n_conditions,
        seed=seed,
    )
    return manifest.samples


def split_dataset(
    samples: list,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    speaker_disjoint: bool = True,
    seed: int = 42,
):
    from .manifest import Manifest, assign_splits
    manifest = Manifest(samples)
    split_manifest = assign_splits(
        manifest,
        ratios=(train_ratio, val_ratio, test_ratio),
        strategy="speaker" if speaker_disjoint else "random",
        seed=seed,
    )
    return (
        [s for s in split_manifest if s.split == "train"],
        [s for s in split_manifest if s.split == "val"],
        [s for s in split_manifest if s.split == "test"],
    )