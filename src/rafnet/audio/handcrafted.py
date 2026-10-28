import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from scipy.fft import dct


@dataclass
class HandcraftedConfig:
    sample_rate: int = 16000
    n_fft: int = 512
    hop_length: int = 160
    win_length: int = 400
    n_mels: int = 80
    n_mfcc: int = 40
    n_lfcc: int = 60
    n_cqt: int = 84
    fmin: float = 50.0
    fmax: float = 8000.0


class LFCCExtractor(nn.Module):
    def __init__(self, config: HandcraftedConfig):
        super().__init__()
        self.config = config
        self.register_buffer("mel_basis", torch.from_numpy(
            librosa.filters.mel(sr=config.sample_rate, n_fft=config.n_fft, n_mels=config.n_lfcc, fmin=config.fmin, fmax=config.fmax)
        ).float())
        # Precomputed once (identical values to the per-call construction).
        self.register_buffer("hann_window", torch.hann_window(config.win_length))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spec = torch.stft(
            x, n_fft=self.config.n_fft, hop_length=self.config.hop_length,
            win_length=self.config.win_length, window=self.hann_window.to(x.device),
            return_complex=True
        )
        power_spec = spec.abs().pow(2)
        mel_spec = torch.matmul(self.mel_basis.to(x.device), power_spec)
        log_mel_spec = torch.log(mel_spec + 1e-10)
        lfcc = torch.from_numpy(dct(log_mel_spec.cpu().numpy(), norm='ortho', axis=1)[:, :self.config.n_lfcc]).float().to(x.device)
        return lfcc


class MFCCExtractor(nn.Module):
    def __init__(self, config: HandcraftedConfig):
        super().__init__()
        self.config = config
        self.register_buffer("mel_basis", torch.from_numpy(
            librosa.filters.mel(sr=config.sample_rate, n_fft=config.n_fft, n_mels=config.n_mels, fmin=config.fmin, fmax=config.fmax)
        ).float())
        # Precomputed once (identical values to the per-call construction).
        self.register_buffer("hann_window", torch.hann_window(config.win_length))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spec = torch.stft(
            x, n_fft=self.config.n_fft, hop_length=self.config.hop_length,
            win_length=self.config.win_length, window=self.hann_window.to(x.device),
            return_complex=True
        )
        power_spec = spec.abs().pow(2)
        mel_spec = torch.matmul(self.mel_basis.to(x.device), power_spec)
        log_mel_spec = torch.log(mel_spec + 1e-10)
        mfcc = torch.from_numpy(dct(log_mel_spec.cpu().numpy(), norm='ortho', axis=1)[:, :self.config.n_mfcc]).float().to(x.device)
        return mfcc


class CQTExtractor(nn.Module):
    def __init__(self, config: HandcraftedConfig):
        super().__init__()
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cqt_list = []
        for i in range(x.shape[0]):
            cqt = librosa.cqt(
                x[i].cpu().numpy(), sr=self.config.sample_rate,
                n_bins=self.config.n_cqt, fmin=self.config.fmin,
                hop_length=self.config.hop_length
            )
            cqt_mag = np.abs(cqt)
            # Aggregate over time: mean, std, max
            cqt_stats = np.stack([
                cqt_mag.mean(axis=1),
                cqt_mag.std(axis=1),
                cqt_mag.max(axis=1)
            ], axis=1)  # (n_cqt, 3)
            cqt_list.append(torch.from_numpy(cqt_stats).float())
        return torch.stack(cqt_list).to(x.device)  # (batch, n_cqt, 3)


class SpectralExtractor(nn.Module):
    def __init__(self, config: HandcraftedConfig):
        super().__init__()
        self.config = config
        # Precomputed once (identical values to the per-call construction).
        self.register_buffer("hann_window", torch.hann_window(config.win_length))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spec = torch.stft(
            x, n_fft=self.config.n_fft, hop_length=self.config.hop_length,
            win_length=self.config.win_length, window=self.hann_window.to(x.device),
            return_complex=True
        )
        magnitude = spec.abs()  # (batch, freq, time)
        
        # Compute per-frame features then aggregate over time
        freq_axis = torch.arange(magnitude.shape[1]).float().to(x.device).view(1, -1, 1)
        spectral_centroid_per_frame = torch.sum(magnitude * freq_axis, dim=1) / (magnitude.sum(dim=1) + 1e-10)  # (batch, time)
        spectral_centroid = spectral_centroid_per_frame.mean(dim=1, keepdim=True)  # (batch, 1)
        
        spectral_rolloff_per_frame = torch.quantile(magnitude.cumsum(dim=1) / (magnitude.sum(dim=1, keepdim=True) + 1e-10), 0.85, dim=1)  # (batch, time)
        spectral_rolloff = spectral_rolloff_per_frame.mean(dim=1, keepdim=True)  # (batch, 1)
        
        spectral_flux_per_frame = torch.cat([magnitude[:, :, :1], magnitude[:, :, 1:] - magnitude[:, :, :-1]], dim=2).abs().mean(dim=1)  # (batch, time)
        spectral_flux = spectral_flux_per_frame.mean(dim=1, keepdim=True)  # (batch, 1)
        
        zero_crossing = ((x[:, 1:] * x[:, :-1]) < 0).float().mean(dim=1, keepdim=True)  # (batch, 1)
        
        features = torch.cat([
            spectral_centroid,
            spectral_rolloff,
            spectral_flux,
            zero_crossing
        ], dim=-1)  # (batch, 4)
        return features


class ProsodicExtractor(nn.Module):
    def __init__(self, config: HandcraftedConfig):
        super().__init__()
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f0_list = []
        energy_list = []
        
        for i in range(x.shape[0]):
            f0, _, _ = librosa.pyin(
                x[i].cpu().numpy(), fmin=self.config.fmin, fmax=self.config.fmax,
                sr=self.config.sample_rate, hop_length=self.config.hop_length
            )
            f0 = np.nan_to_num(f0)
            f0_list.append(torch.from_numpy(f0).float())
            
            frame_energy = librosa.feature.rms(
                y=x[i].cpu().numpy(), frame_length=self.config.win_length,
                hop_length=self.config.hop_length
            )[0]
            energy_list.append(torch.from_numpy(frame_energy).float())
        
        f0_tensor = torch.stack(f0_list).to(x.device)
        energy_tensor = torch.stack(energy_list).to(x.device)
        
        f0_stats = torch.stack([
            f0_tensor.mean(dim=1), f0_tensor.std(dim=1),
            f0_tensor.median(dim=1).values, f0_tensor.max(dim=1).values,
        ], dim=-1)
        
        energy_stats = torch.stack([
            energy_tensor.mean(dim=1), energy_tensor.std(dim=1),
            energy_tensor.median(dim=1).values, energy_tensor.max(dim=1).values,
        ], dim=-1)
        
        return torch.cat([f0_stats, energy_stats], dim=-1)


def _autocorr_full_fft(y: np.ndarray) -> np.ndarray:
    """Full autocorrelation via FFT (Wiener-Khinchin), matching np.correlate(y, y, mode='full').

    Direct correlation is O(N^2) — minutes on a 30 s clip. The FFT form is
    O(N log N) (~ms) with the same values up to floating-point noise (~1e-7
    relative in float64), which the downstream rounded outputs absorb.
    """
    n = int(len(y))
    if n == 0:
        return np.zeros(1, dtype=np.float64)
    size = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(np.asarray(y, dtype=np.float64), n=size)
    circ = np.fft.irfft(np.abs(spectrum) ** 2, n=size)
    # Circular lags 0..n-1 are first; negative lags -(n-1)..-1 wrap at the
    # end (unpolluted: zero padding covers indices >= n). Reorder to the
    # np.correlate(..., mode='full') layout: [-(n-1) .. +(n-1)].
    return np.concatenate([circ[size - (n - 1):], circ[:n]])


class VoiceQualityExtractor(nn.Module):
    def __init__(self, config: HandcraftedConfig):
        super().__init__()
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hnr_list = []
        jitter_list = []
        shimmer_list = []
        
        for i in range(x.shape[0]):
            y = x[i].cpu().numpy()
            
            f0, _, _ = librosa.pyin(y, fmin=self.config.fmin, fmax=self.config.fmax,
                                     sr=self.config.sample_rate, hop_length=self.config.hop_length)
            f0 = np.nan_to_num(f0)
            
            if len(f0) > 1 and f0.mean() > 0:
                periods = self.config.sample_rate / (f0 + 1e-10)
                jitter = np.abs(np.diff(periods)).mean() / (periods.mean() + 1e-10)
                jitter_list.append(jitter)
            else:
                jitter_list.append(0.0)
            
            frame_energy = librosa.feature.rms(y=y, frame_length=self.config.win_length, hop_length=self.config.hop_length)[0]
            if len(frame_energy) > 1:
                shimmer = np.abs(np.diff(frame_energy)).mean() / (frame_energy.mean() + 1e-10)
                shimmer_list.append(shimmer)
            else:
                shimmer_list.append(0.0)
            
            autocorr = _autocorr_full_fft(y)
            autocorr = autocorr[len(autocorr)//2:]
            peaks = np.where((autocorr[1:-1] > autocorr[:-2]) & (autocorr[1:-1] > autocorr[2:]))[0] + 1
            if len(peaks) > 1:
                hnr = autocorr[peaks[0]] / (autocorr[peaks[1]] + 1e-10)
                hnr_list.append(hnr)
            else:
                hnr_list.append(0.0)
        
        return torch.tensor([
            hnr_list, jitter_list, shimmer_list
        ], device=x.device).T.float()


class HandcraftedFeatureExtractor(nn.Module):
    def __init__(self, config: HandcraftedConfig, enabled_features: List[str]):
        super().__init__()
        self.config = config
        self.enabled_features = enabled_features
        
        self.extractors = nn.ModuleDict()
        if "lfcc" in enabled_features:
            self.extractors["lfcc"] = LFCCExtractor(config)
        if "mfcc" in enabled_features:
            self.extractors["mfcc"] = MFCCExtractor(config)
        if "cqt" in enabled_features:
            self.extractors["cqt"] = CQTExtractor(config)
        if "spectral" in enabled_features:
            self.extractors["spectral"] = SpectralExtractor(config)
        if "prosodic" in enabled_features:
            self.extractors["prosodic"] = ProsodicExtractor(config)
        if "voice_quality" in enabled_features:
            self.extractors["voice_quality"] = VoiceQualityExtractor(config)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = {}
        for name, extractor in self.extractors.items():
            feat = extractor(x)
            # Ensure all features are 2D: (batch, feat_dim)
            # Aggregate over time dimension to get fixed-size features
            if feat.dim() > 2:
                # Aggregate over time: mean, std, max
                feat_mean = feat.mean(dim=-1)
                feat_std = feat.std(dim=-1)
                feat_max = feat.max(dim=-1).values
                feat = torch.cat([feat_mean, feat_std, feat_max], dim=-1)
            features[name] = feat
        return features

    def get_feature_dim(self) -> int:
        total = 0
        with torch.no_grad():
            dummy = torch.randn(1, self.config.sample_rate)
            feats = self.forward(dummy)
            for v in feats.values():
                total += v.numel() // v.shape[0]
        return total