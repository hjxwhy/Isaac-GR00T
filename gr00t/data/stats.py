#!/usr/bin/env python
"""
Calculate dataset statistics for LeRobot datasets.
Note: Please update the `gr00t/configs/data/embodiment_configs.py` file with the correct modality configurations for the dataset you are using before running this script.

Usage:
    python gr00t/data/stats.py <dataset_path> <embodiment_tag>

Args:
    dataset_path: Path to the dataset.
    embodiment_tag: Embodiment tag to use to load modality configurations from `gr00t/configs/data/embodiment_configs.py`.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.state_action.action_chunking import EndEffectorActionChunk, JointActionChunk
from gr00t.data.state_action.pose import EndEffectorPose, JointPose
from gr00t.data.types import (
    ActionFormat,
    ActionRepresentation,
    ActionType,
    EmbodimentTag,
    ModalityConfig,
)
from gr00t.data.utils import to_json_serializable


LE_ROBOT_DATA_FILENAME = "data/*/*.parquet"
LE_ROBOT_INFO_FILENAME = "meta/info.json"
LE_ROBOT_STATS_FILENAME = "meta/stats.json"
LE_ROBOT_REL_STATS_FILENAME = "meta/relative_stats.json"


def _detect_dataset_version(dataset_path: Path) -> str:
    """Detect LeRobot dataset version from meta/info.json."""
    info_path = dataset_path / LE_ROBOT_INFO_FILENAME
    with open(info_path, "r") as f:
        info = json.load(f)
    version = info.get("codebase_version", "v2.0")
    return "v3" if version.startswith("v3") else "v2"


def _load_v3_episode_records(dataset_path: Path) -> list[dict]:
    """Load episode metadata from v3 parquet files in meta/episodes/."""
    episodes_dir = dataset_path / "meta" / "episodes"
    pq_paths = sorted(episodes_dir.glob("chunk-*/file-*.parquet"))
    assert len(pq_paths) > 0, f"No episode parquet found in {episodes_dir}"
    records: list[dict] = []
    for pq_path in pq_paths:
        table = pd.read_parquet(
            pq_path,
            columns=["episode_index", "data/chunk_index", "data/file_index",
                      "dataset_from_index", "dataset_to_index"],
        )
        records.extend(table.to_dict("records"))
    records.sort(key=lambda r: int(r["episode_index"]))
    return records


def calculate_dataset_statistics(
    parquet_paths: list[Path], features: list[str] | None = None
) -> dict[str, dict[str, float]]:
    """Calculate the dataset statistics of all columns for a list of parquet files.

    Args:
        parquet_paths (list[Path]): List of paths to parquet files to process.
        features (list[str] | None): List of feature names to compute statistics for.
            If None, computes statistics for all columns in the data.

    Returns:
        dict[str, DatasetStatisticalValues]: Dictionary mapping feature names to their
            statistical values (mean, std, min, max, q01, q99).
    """
    # Dataset statistics
    all_low_dim_data_list = []
    # Collect all the data
    for parquet_path in tqdm(
        sorted(list(parquet_paths)),
        desc="Collecting all parquet files...",
    ):
        # Load the parquet file
        parquet_data = pd.read_parquet(parquet_path)
        parquet_data = parquet_data
        all_low_dim_data_list.append(parquet_data)
    all_low_dim_data = pd.concat(all_low_dim_data_list, axis=0)
    # Compute dataset statistics
    dataset_statistics = {}
    if features is None:
        features = list(all_low_dim_data.columns)
    for le_modality in features:
        print(f"Computing statistics for {le_modality}...")
        np_data = np.vstack(
            [np.asarray(x, dtype=np.float32) for x in all_low_dim_data[le_modality]]
        )
        dataset_statistics[le_modality] = dict(
            mean=np.mean(np_data, axis=0).tolist(),
            std=np.std(np_data, axis=0).tolist(),
            min=np.min(np_data, axis=0).tolist(),
            max=np.max(np_data, axis=0).tolist(),
            q01=np.quantile(np_data, 0.01, axis=0).tolist(),
            q99=np.quantile(np_data, 0.99, axis=0).tolist(),
        )
    return dataset_statistics


def check_stats_validity(dataset_path: Path | str, features: list[str]):
    stats_path = Path(dataset_path) / LE_ROBOT_STATS_FILENAME
    if not stats_path.exists():
        return False
    with open(stats_path, "r") as f:
        stats = json.load(f)
    for feature in features:
        if feature not in stats:
            return False
        if not isinstance(stats[feature], dict):
            return False
        for stat in ["mean", "std", "min", "max", "q01", "q99"]:
            if stat not in stats[feature]:
                return False
    return True


def generate_stats(dataset_path: Path | str):
    dataset_path = Path(dataset_path)
    print(f"Generating stats for {str(dataset_path)}")
    lowdim_features = []
    with open(dataset_path / LE_ROBOT_INFO_FILENAME, "r") as f:
        le_features = json.load(f)["features"]
    for feature in le_features:
        if "float" in le_features[feature]["dtype"]:
            lowdim_features.append(feature)
    if check_stats_validity(dataset_path, lowdim_features):
        return

    parquet_files = list(dataset_path.glob(LE_ROBOT_DATA_FILENAME))
    stats = calculate_dataset_statistics(parquet_files, lowdim_features)
    stats_path = dataset_path / LE_ROBOT_STATS_FILENAME
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)


class RelativeActionLoader:
    def __init__(self, dataset_path: Path | str, embodiment_tag: EmbodimentTag, action_key: str):
        self.dataset_path = Path(dataset_path)
        self.modality_configs: dict[str, ModalityConfig] = {}
        self.action_key = action_key
        # Check action config
        assert action_key in MODALITY_CONFIGS[embodiment_tag.value]["action"].modality_keys
        idx = MODALITY_CONFIGS[embodiment_tag.value]["action"].modality_keys.index(action_key)
        action_configs = MODALITY_CONFIGS[embodiment_tag.value]["action"].action_configs
        assert action_configs is not None, MODALITY_CONFIGS[embodiment_tag.value]["action"]
        self.action_config = action_configs[idx]
        self.modality_configs["action"] = ModalityConfig(
            delta_indices=MODALITY_CONFIGS[embodiment_tag.value]["action"].delta_indices,
            modality_keys=[action_key],
        )
        # Check state config
        state_key = self.action_config.state_key or action_key
        print(f"State key: {state_key}")
        assert state_key in MODALITY_CONFIGS[embodiment_tag.value]["state"].modality_keys
        self.modality_configs["state"] = ModalityConfig(
            delta_indices=MODALITY_CONFIGS[embodiment_tag.value]["state"].delta_indices,
            modality_keys=[state_key],
        )
        # Check state-action consistency
        assert (
            self.modality_configs["state"].delta_indices[-1]
            == self.modality_configs["action"].delta_indices[0]
        )

        # Branch based on dataset version
        self._dataset_version = _detect_dataset_version(self.dataset_path)
        self._state_key = state_key
        if self._dataset_version == "v3":
            self._v3_records = _load_v3_episode_records(self.dataset_path)
            self._parquet_cache: dict[tuple[int, int], pd.DataFrame] = {}
            # Precompute per-file base offsets for efficient row slicing
            self._file_base_offsets: dict[tuple[int, int], int] = {}
            for r in self._v3_records:
                key = (int(r["data/chunk_index"]), int(r["data/file_index"]))
                from_idx = int(r["dataset_from_index"])
                if key not in self._file_base_offsets or from_idx < self._file_base_offsets[key]:
                    self._file_base_offsets[key] = from_idx
        else:
            self.loader = LeRobotEpisodeLoader(dataset_path, self.modality_configs)

    def _get_episode_df_v3(self, trajectory_id: int) -> pd.DataFrame:
        """Load state and action columns for a single episode from v3 parquet data."""
        record = self._v3_records[trajectory_id]
        chunk_idx = int(record["data/chunk_index"])
        file_idx = int(record["data/file_index"])
        cache_key = (chunk_idx, file_idx)

        state_col = f"observation.state.{self._state_key}"
        action_col = f"action.{self.action_key}"

        if cache_key not in self._parquet_cache:
            path = self.dataset_path / f"data/chunk-{chunk_idx:03d}/file-{file_idx:03d}.parquet"
            self._parquet_cache[cache_key] = pd.read_parquet(
                path, columns=[state_col, action_col]
            )

        full_df = self._parquet_cache[cache_key]
        from_idx = int(record["dataset_from_index"])
        to_idx = int(record["dataset_to_index"])

        base_offset = self._file_base_offsets[cache_key]
        start = from_idx - base_offset
        stop = to_idx - base_offset
        episode_slice = full_df.iloc[start:stop]

        result = pd.DataFrame()
        result[f"state.{self._state_key}"] = episode_slice[state_col].values
        result[f"action.{self.action_key}"] = episode_slice[action_col].values
        return result.reset_index(drop=True)

    def load_relative_actions(
        self, trajectory_id: int, output_format: ActionFormat | None = None
    ) -> list[np.ndarray]:
        if self._dataset_version == "v3":
            df = self._get_episode_df_v3(trajectory_id)
        else:
            df = self.loader[trajectory_id]

        # OPTIMIZATION: Extract columns once and convert to numpy arrays
        # This eliminates repeated DataFrame.__getitem__ and Series.__getitem__ calls
        if self.action_config.state_key is not None:
            state_key = f"state.{self.action_config.state_key}"
        else:
            state_key = f"state.{self.action_key}"
        action_key = f"action.{self.action_key}"

        # Convert to numpy arrays once - this is much faster than repeated pandas access
        state_data = df[state_key].values  # Shape: (episode_length, joint_dim)
        action_data = df[action_key].values  # Shape: (episode_length, joint_dim)
        trajectories = []
        usable_length = len(df) - self.modality_configs["action"].delta_indices[-1]
        action_delta_indices = np.array(self.modality_configs["action"].delta_indices)
        for i in range(usable_length):
            state_ind = self.modality_configs["state"].delta_indices[-1] + i
            action_inds = action_delta_indices + i
            last_state = state_data[state_ind]
            actions = action_data[action_inds]
            if self.action_config.type == ActionType.EEF:
                input_format = self.action_config.format
                out_fmt = output_format or input_format
                reference_frame = EndEffectorPose.from_action_format(last_state, input_format)
                traj = EndEffectorActionChunk.from_array(actions, input_format).relative_chunking(
                    reference_frame=reference_frame
                )
                trajectories.append(traj.to(out_fmt).astype(np.float32))
            elif self.action_config.type == ActionType.NON_EEF:
                action_dim = len(actions[0])
                state_ref = last_state[:action_dim] if len(last_state) > action_dim else last_state
                reference_frame = JointPose(state_ref)
                traj = JointActionChunk([JointPose(m) for m in actions]).relative_chunking(
                    reference_frame=reference_frame
                )
                trajectories.append(np.stack([p.joints for p in traj.poses], dtype=np.float32))
            else:
                raise ValueError(f"Unknown ActionType: {self.action_config.type}")
        return trajectories

    def __len__(self) -> int:
        if self._dataset_version == "v3":
            return len(self._v3_records)
        return len(self.loader)


def calculate_stats_for_key(
    dataset_path: Path | str,
    embodiment_tag: EmbodimentTag,
    group_key: str,
    max_episodes: int = -1,
    output_format: ActionFormat | None = None,
) -> dict:
    loader = RelativeActionLoader(dataset_path, embodiment_tag, group_key)
    trajectories = []
    for episode_id in tqdm(range(len(loader)), desc=f"Loading trajectories for key {group_key}"):
        if max_episodes != -1 and episode_id >= max_episodes:
            break
        trajectories.extend(loader.load_relative_actions(episode_id, output_format=output_format))
    # Per-step stats: shape (chunk_size, action_dim)
    per_step_stats = {
        "max": np.max(trajectories, axis=0),
        "min": np.min(trajectories, axis=0),
        "q01": np.quantile(trajectories, 0.01, axis=0),
        "q99": np.quantile(trajectories, 0.99, axis=0),
        "mean": np.mean(trajectories, axis=0),
        "std": np.std(trajectories, axis=0),
    }
    # Global stats: flatten across samples and chunk_size, shape (action_dim,)
    all_steps = np.concatenate(trajectories, axis=0)  # (N * chunk_size, action_dim)
    global_stats = {
        "global_max": np.max(all_steps, axis=0),
        "global_min": np.min(all_steps, axis=0),
        "global_q01": np.quantile(all_steps, 0.01, axis=0),
        "global_q99": np.quantile(all_steps, 0.99, axis=0),
        "global_mean": np.mean(all_steps, axis=0),
        "global_std": np.std(all_steps, axis=0),
    }
    return {**per_step_stats, **global_stats}


def generate_rel_stats(
    dataset_path: Path | str,
    embodiment_tag: EmbodimentTag,
    output_format: ActionFormat | None = None,
) -> None:
    dataset_path = Path(dataset_path)
    action_config = MODALITY_CONFIGS[embodiment_tag.value]["action"]
    if action_config.action_configs is None:
        return
    action_keys = [
        key
        for key, action_config in zip(action_config.modality_keys, action_config.action_configs)
        if action_config.rep == ActionRepresentation.RELATIVE
    ]
    stats_path = Path(dataset_path) / LE_ROBOT_REL_STATS_FILENAME
    if stats_path.exists():
        with open(stats_path, "r") as f:
            stats = json.load(f)
    else:
        stats = {}
    stats = {}
    for action_key in sorted(action_keys):
        if action_key in stats:
            continue
        print(f"Generating relative stats for {dataset_path} {embodiment_tag} {action_key}")
        stats[action_key] = calculate_stats_for_key(
            dataset_path, embodiment_tag, action_key, output_format=output_format
        )
    with open(stats_path, "w") as f:
        json.dump(to_json_serializable(dict(stats)), f, indent=4)


def main(
    dataset_path: Path | str,
    embodiment_tag: EmbodimentTag,
    output_format: ActionFormat | None = None,
):
    # generate_stats(dataset_path)
    generate_rel_stats(dataset_path, embodiment_tag, output_format=output_format)


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
