from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta
import json
from pathlib import Path

import polars as pl
import pytest

from modeling_module.data_loader import ICLEpisodeDataModule, collate_icl_episodes
from modeling_module.icl import (
    AutoTimesICLAdapter,
    EndogenousICLBuilderConfig,
    EndogenousICLDatasetBuilder,
    ICLContractError,
    ICLArtifactError,
    ICLPromptKind,
    ICLSplit,
    SELLMICLAdapter,
    read_icl_episode_artifact,
    write_icl_episode_artifact,
)


def _week(start: date, offset: int) -> int:
    iso = (start + timedelta(weeks=offset)).isocalendar()
    return int(iso.year) * 100 + int(iso.week)


def _history(*, series_count: int = 2, weeks: int = 320) -> pl.DataFrame:
    start = date.fromisocalendar(2020, 1, 1)
    rows = []
    for series_index in range(series_count):
        for offset in range(weeks):
            rows.append(
                {
                    "oper_part_no": f"part-{series_index + 1}",
                    "demand_dt": _week(start, offset),
                    "demand_qty": float(10 * (series_index + 1) + offset % 13),
                }
            )
    return pl.DataFrame(rows)


def _builder() -> EndogenousICLDatasetBuilder:
    return EndogenousICLDatasetBuilder(
        EndogenousICLBuilderConfig(
            lookback=52,
            horizon=26,
            seasonal_period=52,
            window_stride=4,
            validation_episodes_per_series=1,
            test_episodes_per_series=1,
        )
    )


def test_endogenous_icl_builder_is_deterministic_and_temporally_split():
    frame = _history()
    shuffled = frame.sample(fraction=1.0, shuffle=True, seed=91)

    first = _builder().build(frame, source_revision="demand-history-r1")
    second = _builder().build(shuffled, source_revision="demand-history-r1")

    assert first.manifest.manifest_hash == second.manifest.manifest_hash
    assert first.manifest.source_hash == second.manifest.source_hash
    assert first.manifest.series_count == 2
    assert first.manifest.split_counts["validation"] == 2
    assert first.manifest.split_counts["test"] == 2
    assert first.episodes == second.episodes

    for series_id in {item.series_id for item in first.episodes}:
        episodes = [item for item in first.episodes if item.series_id == series_id]
        splits = [item.split for item in episodes]
        assert splits[-2:] == [ICLSplit.VALIDATION, ICLSplit.TEST]
        assert all(split is ICLSplit.TRAIN for split in splits[:-2])
        train = [item for item in episodes if item.split is ICLSplit.TRAIN]
        validation = [item for item in episodes if item.split is ICLSplit.VALIDATION]
        test = [item for item in episodes if item.split is ICLSplit.TEST]
        assert max(item.query_target.end_week for item in train) < min(
            item.query_target.start_week for item in validation
        )
        assert max(item.query_target.end_week for item in validation) < min(
            item.query_target.start_week for item in test
        )


def test_h27_stride26_split_drops_overlapping_boundary_episodes():
    builder = EndogenousICLDatasetBuilder(
        EndogenousICLBuilderConfig(
            lookback=52,
            horizon=27,
            seasonal_period=52,
            window_stride=26,
            validation_episodes_per_series=1,
            test_episodes_per_series=1,
        )
    )
    bundle = builder.build(
        _history(series_count=1, weeks=420),
        source_revision="h27-boundary-r1",
    )
    by_split = {
        split: [item for item in bundle.episodes if item.split is split]
        for split in ICLSplit
    }

    assert by_split[ICLSplit.TRAIN]
    assert len(by_split[ICLSplit.VALIDATION]) == 1
    assert len(by_split[ICLSplit.TEST]) == 1
    assert max(
        item.query_target.end_week for item in by_split[ICLSplit.TRAIN]
    ) < by_split[ICLSplit.VALIDATION][0].query_target.start_week
    assert (
        by_split[ICLSplit.VALIDATION][0].query_target.end_week
        < by_split[ICLSplit.TEST][0].query_target.start_week
    )


def test_endogenous_icl_prompts_are_same_series_seasonal_and_non_overlapping():
    bundle = _builder().build(_history(series_count=1), source_revision="history-r2")
    episode = bundle.episodes[0]

    assert [item.kind for item in episode.demonstrations] == [
        ICLPromptKind.SEASONAL,
        ICLPromptKind.HISTORICAL,
    ]
    seasonal, historical = episode.demonstrations
    assert seasonal.series_id == historical.series_id == episode.series_id
    assert seasonal.end_week < historical.start_week
    assert historical.end_week < episode.query_context.start_week
    assert len(seasonal.context.weeks) == len(historical.context.weeks) == 52
    assert len(seasonal.target.weeks) == len(historical.target.weeks) == 26

    start = date.fromisocalendar(
        seasonal.target.start_week // 100,
        seasonal.target.start_week % 100,
        1,
    )
    query = date.fromisocalendar(
        episode.query_target.start_week // 100,
        episode.query_target.start_week % 100,
        1,
    )
    assert (query - start).days % (52 * 7) == 0
    assert any(week % 100 == 1 for week in episode.query_context.weeks)


def test_endogenous_icl_aggregates_duplicate_item_weeks_by_sum():
    frame = _history(series_count=1)
    duplicate_week = int(frame["demand_dt"][0])
    duplicate = pl.DataFrame(
        {
            "oper_part_no": ["part-1"],
            "demand_dt": [duplicate_week],
            "demand_qty": [5.0],
        }
    )
    bundle = _builder().build(
        pl.concat([frame, duplicate]),
        source_revision="history-r3",
    )

    seasonal = bundle.episodes[0].demonstrations[0]
    assert seasonal.context.target[0][0] == 15.0


def test_endogenous_icl_rejects_implicit_zero_fill_for_missing_weeks():
    frame = _history(series_count=1).filter(pl.col("demand_dt") != 202010)

    with pytest.raises(ICLContractError, match="missing weekly period"):
        _builder().build(frame, source_revision="history-gap")


def test_episode_data_module_and_model_adapters_preserve_prompt_contract():
    bundle = _builder().build(_history(), source_revision="history-r4")
    module = ICLEpisodeDataModule(bundle, batch_size=2, seed=17)
    batch = next(iter(module.loader(ICLSplit.TEST, shuffle=False)))

    assert batch.query_context.shape == (2, 52, 1)
    assert batch.query_target.shape == (2, 26, 1)
    assert batch.demonstration_contexts.shape == (2, 2, 52, 1)
    assert batch.demonstration_targets.shape == (2, 2, 26, 1)
    assert batch.prompt_mask.all()

    autotimes = AutoTimesICLAdapter().adapt(batch)
    sellm = SELLMICLAdapter().adapt(batch)

    assert autotimes.packed_context.shape == (2, 208, 1)
    assert autotimes.query_target.shape == (2, 26, 1)
    assert sellm.demonstration_contexts.shape == (2, 2, 52, 1)
    assert sellm.demonstration_targets.shape == (2, 2, 26, 1)
    assert sellm.query_context.shape == (2, 52, 1)
    assert sellm.query_target.shape == (2, 26, 1)
    assert autotimes.series_ids == sellm.series_ids


def test_shared_episode_contract_preserves_exogenous_features_for_both_adapters():
    episode = _builder().build(
        _history(series_count=1),
        source_revision="history-with-exogenous-r1",
    ).episodes[-1]

    def with_exogenous(window):
        return replace(
            window,
            exogenous=tuple((float(index), float(index % 4)) for index in range(len(window.weeks))),
        )

    exogenous_episode = replace(
        episode,
        query_context=with_exogenous(episode.query_context),
        query_target=with_exogenous(episode.query_target),
        demonstrations=tuple(
            replace(
                item,
                context=with_exogenous(item.context),
                target=with_exogenous(item.target),
            )
            for item in episode.demonstrations
        ),
    )
    batch = collate_icl_episodes([exogenous_episode])

    autotimes = AutoTimesICLAdapter().adapt(batch)
    sellm = SELLMICLAdapter().adapt(batch)

    assert autotimes.packed_exogenous is not None
    assert autotimes.packed_exogenous.shape == (1, 208, 4)
    assert autotimes.query_target_exogenous is not None
    assert autotimes.query_target_exogenous.shape == (1, 26, 2)
    assert sellm.demonstration_context_exogenous is not None
    assert sellm.demonstration_context_exogenous.shape == (1, 2, 52, 2)
    assert sellm.query_target_exogenous is not None
    assert sellm.query_target_exogenous.shape == (1, 26, 2)


def test_episode_artifact_round_trip_exact_replay_and_hash_verification(tmp_path: Path):
    bundle = _builder().build(_history(), source_revision="artifact-r1")
    artifact_dir = tmp_path / "weekly-icl-r1"

    receipt = write_icl_episode_artifact(bundle, artifact_dir)
    loaded, loaded_receipt = read_icl_episode_artifact(artifact_dir)
    replay = write_icl_episode_artifact(bundle, artifact_dir)

    assert loaded == bundle
    assert receipt.manifest_hash == bundle.manifest.manifest_hash
    assert loaded_receipt.episode_file_sha256 == receipt.episode_file_sha256
    assert replay.exact_replay is True

    changed = _builder().build(_history(), source_revision="artifact-r2")
    with pytest.raises(ICLArtifactError, match="different ICL manifest"):
        write_icl_episode_artifact(changed, artifact_dir)

    manifest_path = artifact_dir / "manifest.json"
    envelope = json.loads(manifest_path.read_text(encoding="utf-8"))
    envelope["episode_file"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises(ICLArtifactError, match="Parquet SHA256 mismatch"):
        read_icl_episode_artifact(artifact_dir)
