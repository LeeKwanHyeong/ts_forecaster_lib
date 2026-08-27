import types

import pytest
import torch

from modeling_module.data_loader.exogenous_contracts import (
    ExogenousBatch,
    ExogenousFeatureSchema,
)
from modeling_module.training.adapters import DefaultAdapter
from modeling_module.training.config import TrainingConfig
from modeling_module.training.engine import CommonTrainer


def _make_exogenous_trainer() -> CommonTrainer:
    cfg = TrainingConfig(
        device="cpu",
        amp_device="cpu",
        use_amp=False,
        lookback=4,
        horizon=2,
        use_exogenous_mode=True,
    )
    return CommonTrainer(
        cfg=cfg,
        adapter=DefaultAdapter(),
        device="cpu",
        logger=lambda _: None,
    )


def test_common_trainer_resolves_future_exo_without_nan_stat_attribute_error():
    cfg = TrainingConfig(
        device="cpu",
        amp_device="cpu",
        use_amp=False,
        lookback=4,
        horizon=2,
    )
    trainer = CommonTrainer(
        cfg=cfg,
        adapter=DefaultAdapter(),
        device="cpu",
    )

    x = torch.zeros(3, 4, 1)
    y = torch.zeros(3, 2)
    future_exo = torch.randn(3, 2, 5)

    resolved = trainer._resolve_future_exo(future_exo, x, y, device=torch.device("cpu"))

    assert resolved is not None
    assert resolved.shape == (3, 2, 5)
    assert resolved.device.type == "cpu"


def test_common_trainer_unpacks_future_cat_and_preserves_legacy_tuples():
    trainer = _make_exogenous_trainer()
    x = torch.zeros(2, 4, 1)
    y = torch.zeros(2, 2)
    part_ids = ["A", "B"]
    future_cont = torch.zeros(2, 2, 1)
    past_cont = torch.zeros(2, 4, 1)
    past_cat = torch.zeros(2, 4, 1, dtype=torch.long)
    future_cat = torch.ones(2, 2, 1, dtype=torch.long)

    unpacked = trainer._unpack_batch(
        (
            x,
            y,
            part_ids,
            future_cont,
            past_cont,
            past_cat,
            future_cat,
        )
    )

    assert unpacked == (
        x,
        y,
        part_ids,
        future_cont,
        past_cont,
        past_cat,
        future_cat,
    )

    legacy_batches = (
        (x, y),
        (x, y, part_ids),
        (x, y, part_ids, future_cont, past_cont),
        (x, y, part_ids, future_cont, past_cont, past_cat),
    )
    for legacy_batch in legacy_batches:
        assert trainer._unpack_batch(legacy_batch)[-1] is None


def test_common_trainer_prepares_and_validates_future_categorical_batch():
    trainer = _make_exogenous_trainer()
    schema = ExogenousFeatureSchema.from_columns(
        past_cont=["price"],
        past_cat=["segment"],
        future_cont=["promo"],
        future_cat=["event"],
        past_cat_cardinalities=[4],
        future_cat_cardinalities=[3],
    )
    x = torch.zeros(2, 4, 1)
    y = torch.zeros(2, 2)
    future_cont = torch.randn(2, 2, 1)
    past_cont = torch.randn(2, 4, 1)
    past_cat = torch.tensor(
        [
            [[0], [1], [2], [3]],
            [[3], [2], [1], [0]],
        ],
        dtype=torch.long,
    )
    future_cat = torch.tensor(
        [
            [[0], [1]],
            [[2], [0]],
        ],
        dtype=torch.long,
    )

    exogenous = trainer._prepare_exogenous_batch(
        x=x,
        y=y,
        future_exo_cont=future_cont,
        past_exo_cont=past_cont,
        past_exo_cat=past_cat,
        future_exo_cat=future_cat,
        device=torch.device("cpu"),
        schema=schema,
    )

    assert isinstance(exogenous, ExogenousBatch)
    assert exogenous.future_cat is not None
    assert exogenous.future_cat.shape == (2, 2, 1)
    assert exogenous.future_cat.dtype == torch.long
    assert exogenous.future_cat.device.type == "cpu"
    assert all(
        tensor is None or tensor.device.type == "cpu"
        for tensor in (
            exogenous.past_cont,
            exogenous.past_cat,
            exogenous.future_cont,
            exogenous.future_cat,
        )
    )


@pytest.mark.parametrize(
    ("future_cat", "match"),
    [
        (
            torch.zeros(2, 2, 1),
            "future_cat must use an integer dtype",
        ),
        (
            torch.zeros(2, 3, 1, dtype=torch.long),
            "future_cat time-axis mismatch",
        ),
        (
            torch.tensor(
                [
                    [[0], [3]],
                    [[1], [2]],
                ],
                dtype=torch.long,
            ),
            "future_cat category IDs exceed schema cardinality",
        ),
        (
            torch.tensor(
                [
                    [[0], [-1]],
                    [[1], [2]],
                ],
                dtype=torch.long,
            ),
            "future_cat category IDs must be non-negative",
        ),
    ],
)
def test_common_trainer_rejects_invalid_future_categorical_batch(
    future_cat: torch.Tensor,
    match: str,
):
    trainer = _make_exogenous_trainer()
    schema = ExogenousFeatureSchema.from_columns(
        future_cat=["event"],
        future_cat_cardinalities=[3],
    )

    with pytest.raises((TypeError, ValueError), match=match):
        trainer._prepare_exogenous_batch(
            x=torch.zeros(2, 4, 1),
            y=torch.zeros(2, 2),
            future_exo_cont=None,
            past_exo_cont=None,
            past_exo_cat=None,
            future_exo_cat=future_cat,
            device=torch.device("cpu"),
            schema=schema,
        )


@pytest.mark.parametrize(
    ("schema", "match"),
    [
        (
            None,
            "future_cat requires an ExogenousFeatureSchema",
        ),
        (
            ExogenousFeatureSchema.from_columns(future_cat=["event"]),
            "one resolved cardinality for every future categorical feature",
        ),
    ],
)
def test_common_trainer_requires_resolved_future_cat_cardinalities(
    schema: ExogenousFeatureSchema | None,
    match: str,
):
    trainer = _make_exogenous_trainer()

    with pytest.raises(ValueError, match=match):
        trainer._prepare_exogenous_batch(
            x=torch.zeros(2, 4, 1),
            y=torch.zeros(2, 2),
            future_exo_cont=None,
            past_exo_cont=None,
            past_exo_cat=None,
            future_exo_cat=torch.zeros(2, 2, 1, dtype=torch.long),
            device=torch.device("cpu"),
            schema=schema,
        )


def test_common_trainer_forwards_future_cat_to_compatible_adapter():
    class CapturingAdapter:
        def __init__(self):
            self.received_future_cat = None

        def forward(
            self,
            model,
            x,
            *,
            future_exo=None,
            future_exo_cat=None,
            past_exo_cont=None,
            past_exo_cat=None,
            part_ids=None,
            mode=None,
        ):
            self.received_future_cat = future_exo_cat
            return x[:, :2, 0]

    trainer = _make_exogenous_trainer()
    adapter = CapturingAdapter()
    trainer.adapter = adapter
    future_cat = torch.zeros(2, 2, 1, dtype=torch.long)

    output = trainer._forward_with_adapter(
        object(),
        torch.ones(2, 4, 1),
        future_exo=None,
        future_exo_cat=future_cat,
        past_exo_cont=None,
        past_exo_cat=None,
        part_ids=["A", "B"],
        mode="train",
    )

    assert adapter.received_future_cat is future_cat
    assert output.shape == (2, 2)


def test_common_trainer_preserves_legacy_adapter_signature_without_future_cat():
    class LegacyAdapter:
        def __init__(self):
            self.mode = None

        def forward(self, model, x, *, future_exo=None, mode=None):
            self.mode = mode
            return x[:, :2, 0]

    trainer = _make_exogenous_trainer()
    adapter = LegacyAdapter()
    trainer.adapter = adapter

    output = trainer._forward_with_adapter(
        object(),
        torch.ones(2, 4, 1),
        future_exo=torch.zeros(2, 2, 1),
        future_exo_cat=None,
        past_exo_cont=torch.zeros(2, 4, 1),
        past_exo_cat=None,
        part_ids=["A", "B"],
        mode="eval",
    )

    assert output.shape == (2, 2)
    assert adapter.mode == "eval"


def test_common_trainer_rejects_future_cat_for_legacy_adapter():
    class LegacyAdapter:
        def forward(self, model, x, *, future_exo=None, mode=None):
            return x[:, :2, 0]

    trainer = _make_exogenous_trainer()
    trainer.adapter = LegacyAdapter()

    with pytest.raises(
        NotImplementedError,
        match="LegacyAdapter.forward does not declare `future_exo_cat`",
    ):
        trainer._forward_with_adapter(
            object(),
            torch.ones(2, 4, 1),
            future_exo=None,
            future_exo_cat=torch.zeros(2, 2, 1, dtype=torch.long),
            past_exo_cont=None,
            past_exo_cat=None,
            part_ids=None,
            mode="train",
        )


def test_production_refit_runs_exact_epochs_without_best_state_restore():
    cfg = TrainingConfig(
        device="cpu",
        amp_device="cpu",
        use_amp=False,
        lookback=4,
        horizon=2,
        epochs=3,
        training_mode="production_refit",
    )
    trainer = CommonTrainer(
        cfg=cfg,
        adapter=DefaultAdapter(),
        device="cpu",
        logger=lambda _: None,
    )
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()

    epoch_values: list[float] = []

    def fake_run_epoch(self, current_model, _loader, *, train):
        assert train is True
        self.opt.zero_grad()
        self.opt.step()
        value = float(len(epoch_values) + 1)
        with torch.no_grad():
            current_model.weight.fill_(value)
        epoch_values.append(value)
        return value

    trainer._run_epoch = types.MethodType(fake_run_epoch, trainer)
    trained = trainer.fit(model, [object()], None)

    assert epoch_values == [1.0, 2.0, 3.0]
    assert trained.weight.item() == pytest.approx(3.0)
    assert trainer.epochs_completed_ == 3
    assert trainer.final_train_loss_ == pytest.approx(3.0)
    assert trainer.best_loss_ is None
    assert trainer.validation_enabled_ is False


def test_qualification_still_requires_validation_loader():
    cfg = TrainingConfig(
        device="cpu",
        amp_device="cpu",
        use_amp=False,
        epochs=1,
    )
    trainer = CommonTrainer(
        cfg=cfg,
        adapter=DefaultAdapter(),
        device="cpu",
        logger=lambda _: None,
    )

    with pytest.raises(ValueError, match="requires a validation loader"):
        trainer.fit(torch.nn.Linear(1, 1), [object()], None)
