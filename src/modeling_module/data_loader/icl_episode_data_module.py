"""Torch DataModule that consumes sealed ICL episodes without rebuilding them."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from modeling_module.icl.contracts import (
    ICLEpisode,
    ICLEpisodeBundle,
    ICLManifest,
    ICLSplit,
)


@dataclass(frozen=True)
class ICLBatch:
    episode_ids: tuple[str, ...]
    series_ids: tuple[str, ...]
    splits: tuple[str, ...]
    origin_weeks: torch.Tensor
    query_context: torch.Tensor
    query_target: torch.Tensor
    demonstration_contexts: torch.Tensor
    demonstration_targets: torch.Tensor
    prompt_mask: torch.Tensor
    query_target_observed: torch.Tensor
    query_context_exogenous: torch.Tensor | None = None
    query_target_exogenous: torch.Tensor | None = None
    demonstration_context_exogenous: torch.Tensor | None = None
    demonstration_target_exogenous: torch.Tensor | None = None

    def to(self, device: torch.device | str) -> "ICLBatch":
        return ICLBatch(
            episode_ids=self.episode_ids,
            series_ids=self.series_ids,
            splits=self.splits,
            origin_weeks=self.origin_weeks.to(device),
            query_context=self.query_context.to(device),
            query_target=self.query_target.to(device),
            demonstration_contexts=self.demonstration_contexts.to(device),
            demonstration_targets=self.demonstration_targets.to(device),
            prompt_mask=self.prompt_mask.to(device),
            query_target_observed=self.query_target_observed.to(device),
            query_context_exogenous=_move_optional(
                self.query_context_exogenous,
                device,
            ),
            query_target_exogenous=_move_optional(
                self.query_target_exogenous,
                device,
            ),
            demonstration_context_exogenous=_move_optional(
                self.demonstration_context_exogenous,
                device,
            ),
            demonstration_target_exogenous=_move_optional(
                self.demonstration_target_exogenous,
                device,
            ),
        )


def _move_optional(
    value: torch.Tensor | None,
    device: torch.device | str,
) -> torch.Tensor | None:
    return None if value is None else value.to(device)


class ICLEpisodeDataset(Dataset):
    """Read-only Dataset over episodes that were already built and sealed."""

    def __init__(
        self,
        episodes: Sequence[ICLEpisode],
        *,
        manifest: ICLManifest | None = None,
    ) -> None:
        self._episodes = tuple(episodes)
        self.manifest = manifest

    def __len__(self) -> int:
        return len(self._episodes)

    def __getitem__(self, index: int) -> ICLEpisode:
        return self._episodes[index]


def collate_icl_episodes(episodes: Sequence[ICLEpisode]) -> ICLBatch:
    if not episodes:
        raise ValueError("Cannot collate an empty ICL episode batch.")
    lookback = len(episodes[0].query_context.weeks)
    horizon = len(episodes[0].query_target.weeks)
    channels = episodes[0].query_context.target_dim
    prompt_count = max(len(item.demonstrations) for item in episodes)
    if prompt_count <= 0:
        raise ValueError("ICL episodes must contain at least one prompt.")

    query_context = torch.zeros(len(episodes), lookback, channels, dtype=torch.float32)
    query_target = torch.zeros(len(episodes), horizon, channels, dtype=torch.float32)
    prompt_contexts = torch.zeros(
        len(episodes), prompt_count, lookback, channels, dtype=torch.float32
    )
    prompt_targets = torch.zeros(
        len(episodes), prompt_count, horizon, channels, dtype=torch.float32
    )
    prompt_mask = torch.zeros(len(episodes), prompt_count, dtype=torch.bool)
    past_exogenous_dim = episodes[0].query_context.exogenous_dim
    future_exogenous_dim = episodes[0].query_target.exogenous_dim
    if past_exogenous_dim:
        query_context_exogenous = torch.zeros(
            len(episodes), lookback, past_exogenous_dim, dtype=torch.float32
        )
        prompt_context_exogenous = torch.zeros(
            len(episodes), prompt_count, lookback, past_exogenous_dim, dtype=torch.float32
        )
    else:
        query_context_exogenous = None
        prompt_context_exogenous = None
    if future_exogenous_dim:
        query_target_exogenous = torch.zeros(
            len(episodes), horizon, future_exogenous_dim, dtype=torch.float32
        )
        prompt_target_exogenous = torch.zeros(
            len(episodes), prompt_count, horizon, future_exogenous_dim, dtype=torch.float32
        )
    else:
        query_target_exogenous = None
        prompt_target_exogenous = None

    for batch_index, episode in enumerate(episodes):
        if len(episode.query_context.weeks) != lookback:
            raise ValueError("All query contexts in a batch must use the same lookback.")
        if len(episode.query_target.weeks) != horizon:
            raise ValueError("All query targets in a batch must use the same horizon.")
        if episode.query_context.target_dim != channels:
            raise ValueError("All query targets in a batch must use the same channel count.")
        if episode.query_context.exogenous_dim != past_exogenous_dim:
            raise ValueError("All ICL episodes must use one past exogenous schema.")
        if episode.query_target.exogenous_dim != future_exogenous_dim:
            raise ValueError("All ICL episodes must use one future exogenous schema.")
        query_context[batch_index] = torch.tensor(
            episode.query_context.target,
            dtype=torch.float32,
        )
        query_target[batch_index] = torch.tensor(
            episode.query_target.target,
            dtype=torch.float32,
        )
        if past_exogenous_dim:
            assert episode.query_context.exogenous is not None
            assert query_context_exogenous is not None
            query_context_exogenous[batch_index] = torch.tensor(
                episode.query_context.exogenous,
                dtype=torch.float32,
            )
        if future_exogenous_dim:
            assert episode.query_target.exogenous is not None
            assert query_target_exogenous is not None
            query_target_exogenous[batch_index] = torch.tensor(
                episode.query_target.exogenous,
                dtype=torch.float32,
            )
        for prompt_index, prompt in enumerate(episode.demonstrations):
            if len(prompt.context.weeks) != lookback:
                raise ValueError("Prompt context length must match query lookback in ICL v1.")
            if len(prompt.target.weeks) != horizon:
                raise ValueError("Prompt target length must match query horizon in ICL v1.")
            prompt_contexts[batch_index, prompt_index] = torch.tensor(
                prompt.context.target,
                dtype=torch.float32,
            )
            prompt_targets[batch_index, prompt_index] = torch.tensor(
                prompt.target.target,
                dtype=torch.float32,
            )
            if past_exogenous_dim:
                assert prompt.context.exogenous is not None
                assert prompt_context_exogenous is not None
                prompt_context_exogenous[batch_index, prompt_index] = torch.tensor(
                    prompt.context.exogenous,
                    dtype=torch.float32,
                )
            if future_exogenous_dim:
                assert prompt.target.exogenous is not None
                assert prompt_target_exogenous is not None
                prompt_target_exogenous[batch_index, prompt_index] = torch.tensor(
                    prompt.target.exogenous,
                    dtype=torch.float32,
                )
            prompt_mask[batch_index, prompt_index] = True

    return ICLBatch(
        episode_ids=tuple(item.episode_id for item in episodes),
        series_ids=tuple(item.series_id for item in episodes),
        splits=tuple(item.split.value for item in episodes),
        origin_weeks=torch.tensor(
            [item.origin_week for item in episodes],
            dtype=torch.int64,
        ),
        query_context=query_context,
        query_target=query_target,
        demonstration_contexts=prompt_contexts,
        demonstration_targets=prompt_targets,
        prompt_mask=prompt_mask,
        query_target_observed=torch.tensor(
            [item.query_target_observed for item in episodes],
            dtype=torch.bool,
        ),
        query_context_exogenous=query_context_exogenous,
        query_target_exogenous=query_target_exogenous,
        demonstration_context_exogenous=prompt_context_exogenous,
        demonstration_target_exogenous=prompt_target_exogenous,
    )


def _seed_worker(_: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class ICLEpisodeDataModule:
    """Expose split loaders for an immutable Episode bundle."""

    def __init__(
        self,
        bundle: ICLEpisodeBundle,
        *,
        batch_size: int = 32,
        num_workers: int = 0,
        seed: int = 42,
        pin_memory: bool = False,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if num_workers < 0:
            raise ValueError("num_workers must be non-negative.")
        self.bundle = bundle
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.seed = int(seed)
        self.pin_memory = bool(pin_memory)

    def dataset(self, split: ICLSplit | str) -> ICLEpisodeDataset:
        return ICLEpisodeDataset(
            self.bundle.for_split(split),
            manifest=self.bundle.manifest,
        )

    def loader(
        self,
        split: ICLSplit | str,
        *,
        shuffle: bool | None = None,
    ) -> DataLoader:
        normalized = ICLSplit(split)
        dataset = self.dataset(normalized)
        if len(dataset) == 0:
            raise ValueError(f"ICL split {normalized.value!r} contains no episodes.")
        should_shuffle = normalized is ICLSplit.TRAIN if shuffle is None else bool(shuffle)
        generator = torch.Generator().manual_seed(self.seed)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=should_shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_icl_episodes,
            worker_init_fn=_seed_worker,
            generator=generator,
        )
