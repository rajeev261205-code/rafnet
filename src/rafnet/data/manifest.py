from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Set, Literal
import pandas as pd
import numpy as np
from pathlib import Path
import json
import hashlib
from collections import defaultdict


@dataclass
class SampleManifest:
    sample_id: str
    speaker_id: str
    generator_id: str
    class_label: int
    audio_path: str
    communication_condition_id: str
    codec: str
    packet_loss: float
    jitter: float
    rtt: float
    bitrate: float
    packet_timing: Optional[List[float]] = None
    packet_size: Optional[List[float]] = None
    burst_loss_count: Optional[int] = None
    burst_loss_duration: Optional[float] = None
    loss_burstiness: Optional[float] = None
    split: Literal["train", "val", "test"] = "unassigned"
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict) -> "SampleManifest":
        return cls(**d)


class Manifest:
    REQUIRED_FIELDS = [
        "sample_id", "speaker_id", "generator_id", "class_label",
        "audio_path", "communication_condition_id", "codec",
        "packet_loss", "jitter", "rtt", "bitrate", "split"
    ]
    
    OPTIONAL_TELEMETRY_FIELDS = [
        "packet_timing", "packet_size", "burst_loss_count",
        "burst_loss_duration", "loss_burstiness"
    ]
    
    VALID_SPLITS = {"train", "val", "test"}
    VALID_LABELS = {0, 1}
    
    def __init__(self, samples: List[SampleManifest] = None):
        self.samples = samples or []
        self._df: Optional[pd.DataFrame] = None
        self._index_by_field: Dict[str, Dict] = {}
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> SampleManifest:
        return self.samples[idx]
    
    def __iter__(self):
        return iter(self.samples)
    
    def add(self, sample: SampleManifest) -> None:
        self.samples.append(sample)
        self._df = None
        self._index_by_field = {}
    
    def to_dataframe(self) -> pd.DataFrame:
        if self._df is None:
            self._df = pd.DataFrame([s.to_dict() for s in self.samples])
        return self._df
    
    def validate(self) -> Dict[str, List[str]]:
        errors = defaultdict(list)
        df = self.to_dataframe()
        
        if len(df) == 0:
            errors["general"].append("Manifest is empty")
            return dict(errors)
        
        missing_fields = set(self.REQUIRED_FIELDS) - set(df.columns)
        if missing_fields:
            errors["missing_fields"].extend(list(missing_fields))
        
        for field_name in self.REQUIRED_FIELDS:
            if field_name in df.columns:
                null_count = df[field_name].isnull().sum()
                if null_count > 0:
                    errors["null_values"].append(f"{field_name}: {null_count} nulls")
        
        # Validate optional telemetry fields if present (don't treat nulls as errors)
        # Optional fields are allowed to have nulls, so we don't add them to errors
        
        if "class_label" in df.columns:
            invalid_labels = df[~df["class_label"].isin(self.VALID_LABELS)]
            if len(invalid_labels) > 0:
                errors["invalid_labels"].append(f"Found {len(invalid_labels)} samples with invalid class_label")
        
        if "split" in df.columns:
            invalid_splits = df[~df["split"].isin(self.VALID_SPLITS)]
            if len(invalid_splits) > 0:
                errors["invalid_splits"].append(f"Found {len(invalid_splits)} samples with invalid split")
        
        if "packet_loss" in df.columns:
            invalid_pl = df[(df["packet_loss"] < 0) | (df["packet_loss"] > 1)]
            if len(invalid_pl) > 0:
                errors["invalid_packet_loss"].append(f"Found {len(invalid_pl)} samples with packet_loss outside [0,1]")
        
        if "jitter" in df.columns:
            invalid_jitter = df[df["jitter"] < 0]
            if len(invalid_jitter) > 0:
                errors["invalid_jitter"].append(f"Found {len(invalid_jitter)} samples with negative jitter")
        
        if "rtt" in df.columns:
            invalid_rtt = df[df["rtt"] < 0]
            if len(invalid_rtt) > 0:
                errors["invalid_rtt"].append(f"Found {len(invalid_rtt)} samples with negative rtt")
        
        if "bitrate" in df.columns:
            invalid_bitrate = df[df["bitrate"] <= 0]
            if len(invalid_bitrate) > 0:
                errors["invalid_bitrate"].append(f"Found {len(invalid_bitrate)} samples with non-positive bitrate")
        
        return dict(errors)
    
    def detect_duplicates(self, fields: List[str] = None) -> List[List[int]]:
        if fields is None:
            fields = ["audio_path"]
        
        df = self.to_dataframe()
        dup_groups = df.groupby(fields).apply(lambda x: list(x.index))
        duplicates = [group for group in dup_groups if len(group) > 1]
        return duplicates
    
    def check_audio_files_exist(self, base_path: str = "") -> List[int]:
        missing = []
        for i, sample in enumerate(self.samples):
            path = Path(base_path) / sample.audio_path if base_path else Path(sample.audio_path)
            if not path.exists():
                missing.append(i)
        return missing
    
    def get_split(self, split: str) -> "Manifest":
        return Manifest([s for s in self.samples if s.split == split])
    
    def get_by_speaker(self, speaker_id: str) -> "Manifest":
        return Manifest([s for s in self.samples if s.speaker_id == speaker_id])
    
    def get_by_generator(self, generator_id: str) -> "Manifest":
        return Manifest([s for s in self.samples if s.generator_id == generator_id])
    
    def get_by_condition(self, condition_id: str) -> "Manifest":
        return Manifest([s for s in self.samples if s.communication_condition_id == condition_id])
    
    def speakers(self) -> Set[str]:
        return set(s.speaker_id for s in self.samples)
    
    def generators(self) -> Set[str]:
        return set(s.generator_id for s in self.samples)
    
    def conditions(self) -> Set[str]:
        return set(s.communication_condition_id for s in self.samples)
    
    def class_distribution(self) -> Dict[int, int]:
        df = self.to_dataframe()
        return df["class_label"].value_counts().to_dict()
    
    def condition_class_distribution(self) -> Dict[str, Dict[int, int]]:
        df = self.to_dataframe()
        result = {}
        for cond in df["communication_condition_id"].unique():
            cond_df = df[df["communication_condition_id"] == cond]
            result[cond] = cond_df["class_label"].value_counts().to_dict()
        return result
    
    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump([s.to_dict() for s in self.samples], f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "Manifest":
        with open(path, "r") as f:
            data = json.load(f)
        samples = [SampleManifest.from_dict(d) for d in data]
        return cls(samples)
    
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "Manifest":
        samples = [SampleManifest.from_dict(row.to_dict()) for _, row in df.iterrows()]
        return cls(samples)
    
    def stratify_by_condition(self, n_splits: int = 5, seed: int = 42) -> List["Manifest"]:
        np.random.seed(seed)
        df = self.to_dataframe()
        conditions = df["communication_condition_id"].unique()
        np.random.shuffle(conditions)
        
        splits = []
        for i in range(n_splits):
            cond_subset = conditions[i::n_splits]
            split_samples = df[df["communication_condition_id"].isin(cond_subset)]
            splits.append(Manifest.from_dataframe(split_samples))
        return splits
    
    def compute_hash(self) -> str:
        df = self.to_dataframe()
        content = df.to_csv(index=False).encode()
        return hashlib.sha256(content).hexdigest()[:16]


def create_synthetic_manifest(
    n_samples: int = 1000,
    audio_length: int = 16000,
    n_speakers: int = 10,
    n_generators: int = 5,
    n_conditions: int = 3,
    seed: int = 42,
    split_ratios: tuple = (0.7, 0.15, 0.15),
    include_telemetry: bool = True,
    network_label_correlation: float = 0.0,
) -> Manifest:
    """
    Create synthetic manifest for testing.
    
    Args:
        n_samples: Number of samples
        audio_length: Audio length in samples
        n_speakers: Number of speakers
        n_generators: Number of generators
        n_conditions: Number of communication conditions
        seed: Random seed
        split_ratios: Train/val/test split ratios
        include_telemetry: Whether to include extended telemetry
        network_label_correlation: Correlation between network features and labels.
            0.0 = no correlation (network features independent of label) - DEFAULT for valid evaluation
            > 0.0 = introduces label leakage (use only for debugging/ablation)
    """
    np.random.seed(seed)
    
    samples = []
    
    # Generate condition-level network characteristics (independent of label)
    condition_network_profiles = {}
    for c in range(n_conditions):
        condition_id = f"cond_{c:03d}"
        # Base network profile per condition (same for real and synthetic)
        condition_network_profiles[condition_id] = {
            "packet_loss_base": np.random.beta(2, 8),
            "jitter_base": np.random.exponential(0.02),
            "rtt_base": np.random.gamma(2, 0.05),
            "bitrate_base": np.random.uniform(32000, 256000),
        }
    
    for i in range(n_samples):
        is_synthetic = np.random.rand() > 0.5
        label = 1 if is_synthetic else 0
        
        condition_id = f"cond_{np.random.randint(n_conditions):03d}"
        
        if is_synthetic:
            generator_id = f"gen_{np.random.randint(n_generators):03d}"
        else:
            generator_id = "natural"
        
        speaker_id = f"spk_{np.random.randint(n_speakers):03d}"
        
        # Get condition-level base network characteristics
        cond_profile = condition_network_profiles[condition_id]
        
        # Add small label-dependent variation ONLY if network_label_correlation > 0
        # This is for ablation studies only - default is 0.0 (no leakage)
        if network_label_correlation > 0:
            label_factor = network_label_correlation * (1 if is_synthetic else -1)
            packet_loss = np.clip(cond_profile["packet_loss_base"] + label_factor * 0.05, 0, 1)
            jitter = max(0, cond_profile["jitter_base"] + label_factor * 0.005)
            rtt = max(0, cond_profile["rtt_base"] + label_factor * 0.01)
            bitrate = cond_profile["bitrate_base"] * (1 - label_factor * 0.1)
        else:
            # No label correlation - add only small random noise
            packet_loss = np.clip(cond_profile["packet_loss_base"] + np.random.normal(0, 0.01), 0, 1)
            jitter = max(0, cond_profile["jitter_base"] + np.random.normal(0, 0.002))
            rtt = max(0, cond_profile["rtt_base"] + np.random.normal(0, 0.005))
            bitrate = max(1000, cond_profile["bitrate_base"] + np.random.normal(0, 5000))
        
        codec = np.random.choice(["opus", "g722", "pcmu", "pcma"])
        
        # Generate extended telemetry
        if include_telemetry:
            n_packets = np.random.randint(50, 500)
            packet_timing = np.cumsum(np.random.exponential(0.02, n_packets)).tolist()
            packet_size = np.random.normal(1200, 200, n_packets).clip(64, 1500).tolist()
            
            # Burst loss statistics based on actual packet_loss (not label)
            if packet_loss > 0.1:
                burst_loss_count = np.random.randint(1, 10)
                burst_loss_duration = np.random.exponential(0.1)
                loss_burstiness = np.random.beta(2, 5)
            else:
                burst_loss_count = 0
                burst_loss_duration = 0.0
                loss_burstiness = 0.0
        else:
            packet_timing = None
            packet_size = None
            burst_loss_count = None
            burst_loss_duration = None
            loss_burstiness = None
        
        audio_path = f"audio/{speaker_id}/{generator_id}/{condition_id}/sample_{i:06d}.wav"
        
        sample = SampleManifest(
            sample_id=f"sample_{i:06d}",
            speaker_id=speaker_id,
            generator_id=generator_id,
            class_label=label,
            audio_path=audio_path,
            communication_condition_id=condition_id,
            codec=codec,
            packet_loss=float(packet_loss),
            jitter=float(jitter),
            rtt=float(rtt),
            bitrate=float(bitrate),
            packet_timing=packet_timing,
            packet_size=packet_size,
            burst_loss_count=burst_loss_count,
            burst_loss_duration=burst_loss_duration,
            loss_burstiness=loss_burstiness,
            split="unassigned"
        )
        samples.append(sample)
    
    manifest = Manifest(samples)
    manifest = assign_splits(manifest, split_ratios, seed=seed)
    return manifest


def assign_splits(
    manifest: Manifest,
    ratios: tuple = (0.7, 0.15, 0.15),
    strategy: Literal["speaker", "generator", "condition", "random"] = "speaker",
    seed: int = 42
) -> Manifest:
    np.random.seed(seed)
    df = manifest.to_dataframe()
    
    if strategy == "speaker":
        speakers = list(df["speaker_id"].unique())
        np.random.shuffle(speakers)
        n_train = int(len(speakers) * ratios[0])
        n_val = int(len(speakers) * ratios[1])
        
        train_speakers = set(speakers[:n_train])
        val_speakers = set(speakers[n_train:n_train + n_val])
        test_speakers = set(speakers[n_train + n_val:])
        
        df.loc[df["speaker_id"].isin(train_speakers), "split"] = "train"
        df.loc[df["speaker_id"].isin(val_speakers), "split"] = "val"
        df.loc[df["speaker_id"].isin(test_speakers), "split"] = "test"
        
    elif strategy == "generator":
        generators = list(df["generator_id"].unique())
        np.random.shuffle(generators)
        n_train = int(len(generators) * ratios[0])
        n_val = int(len(generators) * ratios[1])
        
        train_generators = set(generators[:n_train])
        val_generators = set(generators[n_train:n_train + n_val])
        test_generators = set(generators[n_train + n_val:])
        
        df.loc[df["generator_id"].isin(train_generators), "split"] = "train"
        df.loc[df["generator_id"].isin(val_generators), "split"] = "val"
        df.loc[df["generator_id"].isin(test_generators), "split"] = "test"
        
    elif strategy == "condition":
        conditions = list(df["communication_condition_id"].unique())
        np.random.shuffle(conditions)
        n_train = int(len(conditions) * ratios[0])
        n_val = int(len(conditions) * ratios[1])
        
        train_conditions = set(conditions[:n_train])
        val_conditions = set(conditions[n_train:n_train + n_val])
        test_conditions = set(conditions[n_train + n_val:])
        
        df.loc[df["communication_condition_id"].isin(train_conditions), "split"] = "train"
        df.loc[df["communication_condition_id"].isin(val_conditions), "split"] = "val"
        df.loc[df["communication_condition_id"].isin(test_conditions), "split"] = "test"
        
    elif strategy == "random":
        indices = np.random.permutation(len(df))
        n_train = int(len(df) * ratios[0])
        n_val = int(len(df) * ratios[1])
        
        df.loc[indices[:n_train], "split"] = "train"
        df.loc[indices[n_train:n_train + n_val], "split"] = "val"
        df.loc[indices[n_train + n_val:], "split"] = "test"
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    return Manifest.from_dataframe(df)


def check_leakage(manifest: Manifest) -> Dict[str, any]:
    df = manifest.to_dataframe()
    results = {}
    
    results["speaker_overlap"] = check_entity_overlap(df, "speaker_id")
    results["generator_overlap"] = check_entity_overlap(df, "generator_id")
    results["condition_overlap"] = check_entity_overlap(df, "communication_condition_id")
    
    results["condition_class_balance"] = check_condition_class_balance(df)
    results["network_feature_correlation"] = check_network_feature_correlation(df)
    
    return results


def check_entity_overlap(df: pd.DataFrame, entity_col: str) -> Dict[str, any]:
    splits = ["train", "val", "test"]
    entities_by_split = {}
    for split in splits:
        entities_by_split[split] = set(df[df["split"] == split][entity_col].unique())
    
    overlap = {}
    for i, s1 in enumerate(splits):
        for s2 in splits[i+1:]:
            intersection = entities_by_split[s1] & entities_by_split[s2]
            if len(intersection) > 0:
                overlap[f"{s1}_vs_{s2}"] = list(intersection)
    
    return {
        "entities_per_split": {s: len(entities_by_split[s]) for s in splits},
        "overlaps": overlap,
        "has_leakage": len(overlap) > 0
    }


def check_condition_class_balance(df: pd.DataFrame) -> Dict[str, any]:
    results = {}
    for cond in df["communication_condition_id"].unique():
        cond_df = df[df["communication_condition_id"] == cond]
        class_dist = cond_df["class_label"].value_counts().to_dict()
        total = len(cond_df)
        balance = {k: v/total for k, v in class_dist.items()}
        
        results[cond] = {
            "total": total,
            "class_distribution": class_dist,
            "balance": balance,
            "is_imbalanced": min(balance.values()) < 0.3 if len(balance) > 1 else True
        }
    
    return results


def check_network_feature_correlation(df: pd.DataFrame) -> Dict[str, float]:
    network_features = ["packet_loss", "jitter", "rtt", "bitrate"]
    correlations = {}
    
    for feat in network_features:
        if feat in df.columns:
            corr = df[feat].corr(df["class_label"])
            correlations[feat] = float(corr) if not np.isnan(corr) else 0.0
    
    return correlations


def enforce_condition_class_balance(manifest: Manifest, target_balance: float = 0.5) -> Manifest:
    df = manifest.to_dataframe()
    
    balanced_samples = []
    for cond in df["communication_condition_id"].unique():
        cond_df = df[df["communication_condition_id"] == cond]
        
        real = cond_df[cond_df["class_label"] == 0]
        synthetic = cond_df[cond_df["class_label"] == 1]
        
        min_count = min(len(real), len(synthetic))
        max_count = int(min_count / target_balance) if target_balance > 0 else max(len(real), len(synthetic))
        
        if len(real) > min_count:
            real = real.sample(n=min_count, random_state=42)
        if len(synthetic) > min_count:
            synthetic = synthetic.sample(n=min_count, random_state=42)
        
        balanced_samples.extend(real.to_dict("records"))
        balanced_samples.extend(synthetic.to_dict("records"))
    
    balanced_df = pd.DataFrame(balanced_samples)
    return Manifest.from_dataframe(balanced_df)