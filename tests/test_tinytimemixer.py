"""자체 TTM의 입력 계약, 정적 경로, 복원 누수와 전이를 CPU에서 검증한다."""

from __future__ import annotations

import io
import json
import unittest
from dataclasses import asdict, replace
from pathlib import Path

import torch

from modeling_module.models.TinyTimeMixer import (
    TinyTimeMixerConfig,
    TinyTimeMixerModel,
    TinyTimeMixerPretrainModel,
)


class TinyTimeMixerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(19)
        self.cfg = TinyTimeMixerConfig(
            lookback=12, horizon=5, patch_length=3, patch_stride=3,
            d_model=16, decoder_d_model=12, num_layers=1, decoder_num_layers=1,
            dropout=0, categorical_vocab_size_list=(4, 8),
        )
        self.x = torch.randn(2, 12, 1)
        self.ids = torch.tensor([[1, 2], [2, 3]])

    def fixed_mask(self, channels=1):
        mask = torch.zeros(2, channels, 4, dtype=torch.bool)
        mask[:, :, 1] = True
        return mask

    def test_weekly_defaults(self):
        model = TinyTimeMixerModel(TinyTimeMixerConfig())
        self.assertEqual(model(torch.randn(2, 52, 1)).shape, (2, 26, 1))

    def test_forecast_architecture_options(self):
        for channels in (1, 3):
            for norm in (False, True):
                for levels in (0, 1, 2):
                    with self.subTest(channels=channels, norm=norm, levels=levels):
                        cfg = replace(self.cfg, num_input_channels=channels,
                                      use_norm=norm, adaptive_patching_levels=levels)
                        model = TinyTimeMixerModel(cfg)
                        value = model(torch.randn(2, 12, channels), static_categorical_values=self.ids)
                        self.assertEqual(value.shape, (2, 5, channels))
                        self.assertTrue(torch.isfinite(value).all())

    def test_decoder_can_be_disabled_without_static_inputs(self):
        cfg = replace(self.cfg, categorical_vocab_size_list=(), use_decoder=False)
        model = TinyTimeMixerModel(cfg)
        self.assertEqual(model(self.x).shape, (2, 5, 1))
        with self.assertRaises(ValueError):
            model(self.x, static_categorical_values=self.ids)

    def test_common_channel_decoder_without_static_inputs(self):
        cfg = replace(self.cfg, categorical_vocab_size_list=(), decoder_mode="common_channel")
        self.assertEqual(TinyTimeMixerModel(cfg)(self.x).shape, (2, 5, 1))

    def test_static_codes_change_forecast_not_backbone(self):
        model = TinyTimeMixerModel(self.cfg).eval()
        x = self.x[:1].expand(2, -1, -1).clone()
        hidden = model.backbone(x).hidden_state
        torch.testing.assert_close(hidden[0], hidden[1])
        forecast = model(x, static_categorical_values=self.ids)
        self.assertFalse(torch.allclose(forecast[0], forecast[1]))

    def test_backbone_channels_share_weights_without_channel_mixing(self):
        cfg = replace(self.cfg, num_input_channels=3)
        model = TinyTimeMixerModel(cfg).eval()
        x = torch.randn(2, 12, 3)
        permutation = [2, 0, 1]
        actual = model.backbone(x[:, :, permutation]).hidden_state
        expected = model.backbone(x).hidden_state[:, permutation]
        torch.testing.assert_close(actual, expected)

    def test_joint_gradient_reaches_static_embedding_and_backbone(self):
        model = TinyTimeMixerModel(self.cfg)
        model(self.x, static_categorical_values=self.ids).square().mean().backward()
        self.assertGreater(model.backbone.patch_projection.weight.grad.abs().sum().item(), 0)
        for embedding in model.decoder.static_embedding.embeddings:
            self.assertGreater(embedding.weight.grad.abs().sum().item(), 0)

    def test_freezing_stops_backbone_but_not_static_gradients(self):
        model = TinyTimeMixerModel(self.cfg)
        model.freeze_backbone()
        model(self.x, static_categorical_values=self.ids).square().mean().backward()
        self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))
        self.assertIsNotNone(model.decoder.static_embedding.embeddings[0].weight.grad)
        model.freeze_backbone(False)
        self.assertTrue(all(p.requires_grad for p in model.backbone.parameters()))

    def test_parameter_objects_are_not_created_during_forward(self):
        model = TinyTimeMixerModel(self.cfg)
        before = {name: id(parameter) for name, parameter in model.named_parameters()}
        model(self.x, static_categorical_values=self.ids)
        self.assertEqual(before, {name: id(p) for name, p in model.named_parameters()})

    def test_invalid_static_inputs_are_rejected(self):
        model = TinyTimeMixerModel(self.cfg)
        invalid = [None, self.ids.float(), self.ids[:, :1], self.ids.unsqueeze(0),
                   torch.tensor([[-1, 2], [2, 3]]), torch.tensor([[4, 2], [2, 3]])]
        for ids in invalid:
            with self.subTest(ids=ids):
                with self.assertRaises(ValueError):
                    model(self.x, static_categorical_values=ids)

    def test_unconfigured_static_input_is_not_silently_ignored(self):
        cfg = replace(self.cfg, categorical_vocab_size_list=())
        with self.assertRaises(ValueError):
            TinyTimeMixerModel(cfg)(self.x, static_categorical_values=self.ids)

    def test_invalid_time_series_shapes_and_dtypes(self):
        model = TinyTimeMixerModel(self.cfg)
        for x in (self.x.squeeze(-1), self.x[:, :-1], self.x.expand(-1, -1, 2),
                  self.x.long(), self.x[:0]):
            with self.subTest(shape=x.shape):
                with self.assertRaises(ValueError):
                    model(x, static_categorical_values=self.ids)

    def test_masked_missing_values_are_finite(self):
        model = TinyTimeMixerModel(self.cfg)
        x = self.x.clone()
        x[:, 3, :] = float("nan")
        prediction = model(x, static_categorical_values=self.ids)
        self.assertTrue(torch.isfinite(prediction).all())
        with self.assertRaises(ValueError):
            model(x, static_categorical_values=self.ids,
                  past_observed_mask=torch.ones_like(x, dtype=torch.bool))
        with self.assertRaises(ValueError):
            model(torch.full_like(x, float("nan")), static_categorical_values=self.ids)

    def test_invalid_observed_mask(self):
        model = TinyTimeMixerModel(self.cfg)
        with self.assertRaises(ValueError):
            model(self.x, static_categorical_values=self.ids,
                  past_observed_mask=torch.ones_like(self.x))

    def test_constant_series_and_disabled_normalization(self):
        for norm in (False, True):
            model = TinyTimeMixerModel(replace(self.cfg, use_norm=norm))
            result = model(torch.full_like(self.x, 7), static_categorical_values=self.ids)
            self.assertTrue(torch.isfinite(result).all())
            result.sum().backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_patch_padding_keeps_most_recent_value(self):
        cfg = replace(self.cfg, lookback=11, patch_length=4, patch_stride=3, use_norm=False)
        model = TinyTimeMixerModel(cfg)
        x = torch.arange(11, dtype=torch.float32).reshape(1, 11, 1)
        encoded = model.backbone(x)
        self.assertEqual(encoded.patch_values.shape, (1, 1, cfg.num_patches, 4))
        self.assertEqual(encoded.patch_values[0, 0, -1, -1].item(), 10)

    def test_resolution_prefix_and_its_gradient(self):
        cfg = replace(self.cfg, resolution_vocab_size=4)
        model = TinyTimeMixerModel(cfg)
        token = torch.tensor([1, 2])
        hidden = model.backbone(self.x, freq_token=token).hidden_state
        self.assertEqual(hidden.shape[2], cfg.num_patches + 1)
        prediction = model(self.x, static_categorical_values=self.ids, freq_token=token)
        prediction.square().mean().backward()
        self.assertGreater(model.backbone.resolution_embedding.weight.grad.abs().sum().item(), 0)
        for bad in (None, token.float(), torch.tensor([1, 4]), token[:, None]):
            with self.subTest(token=bad):
                with self.assertRaises(ValueError):
                    model(self.x, static_categorical_values=self.ids, freq_token=bad)

    def test_unconfigured_resolution_token_is_rejected(self):
        with self.assertRaises(ValueError):
            TinyTimeMixerModel(self.cfg)(self.x, static_categorical_values=self.ids,
                                        freq_token=torch.tensor([1, 2]))

    def test_invalid_configs(self):
        invalid = [
            {"lookback": 0}, {"horizon": True}, {"patch_length": 13},
            {"patch_stride": 4}, {"dropout": float("nan")}, {"dropout": 1},
            {"norm_eps": 0}, {"adaptive_patching_levels": 4},
            {"decoder_mode": "common_channel"}, {"use_decoder": False},
            {"categorical_vocab_size_list": (0,)}, {"decoder_mode": "unknown"},
        ]
        for override in invalid:
            with self.subTest(override=override):
                with self.assertRaises(ValueError):
                    replace(self.cfg, **override)

    def test_masked_pretraining_joint_gradient(self):
        model = TinyTimeMixerPretrainModel(self.cfg)
        out = model(self.x, static_categorical_values=self.ids, patch_mask=self.fixed_mask())
        self.assertEqual(out["reconstruction"].shape, self.x.shape)
        self.assertEqual(out["mask"].dtype, torch.bool)
        self.assertEqual(out["mask"].sum().item(), 6)
        out["loss"].backward()
        for parameter in (model.backbone.patch_projection.weight,
                          model.backbone.mask_token,
                          model.decoder.static_embedding.embeddings[0].weight,
                          model.reconstruction_head.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_masked_values_do_not_leak_through_normalization_or_raw_residual(self):
        model = TinyTimeMixerPretrainModel(self.cfg).eval()
        mask = self.fixed_mask()
        modified = self.x.clone()
        modified[:, 3:6, :] += 1000
        first = model(self.x, static_categorical_values=self.ids, patch_mask=mask)
        second = model(modified, static_categorical_values=self.ids, patch_mask=mask)
        torch.testing.assert_close(first["reconstruction"], second["reconstruction"])
        self.assertFalse(torch.allclose(first["loss"], second["loss"]))

    def test_pretraining_excludes_missing_targets(self):
        model = TinyTimeMixerPretrainModel(self.cfg)
        x = self.x.clone()
        x[:, 3, :] = float("nan")
        out = model(x, static_categorical_values=self.ids, patch_mask=self.fixed_mask(), loss_type="mae")
        self.assertEqual(out["mask"].sum().item(), 4)
        out["loss"].backward()
        self.assertTrue(torch.isfinite(out["loss"]))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_pretraining_without_decoder_and_with_resolution_prefix(self):
        cfg = replace(self.cfg, categorical_vocab_size_list=(), use_decoder=False,
                      resolution_vocab_size=3, num_input_channels=2)
        out = TinyTimeMixerPretrainModel(cfg)(torch.randn(2, 12, 2), freq_token=torch.tensor([1, 2]))
        self.assertEqual(out["reconstruction"].shape, (2, 12, 2))

    def test_pretraining_rejects_overlap_and_unaligned_length(self):
        for override in ({"patch_stride": 1}, {"lookback": 13},
                         {"patch_length": 12, "patch_stride": 12}):
            with self.subTest(override=override):
                with self.assertRaises(ValueError):
                    TinyTimeMixerPretrainModel(replace(self.cfg, **override))

    def test_pretraining_rejects_invalid_masks(self):
        model = TinyTimeMixerPretrainModel(self.cfg)
        masks = [torch.zeros(2, 1, 4, dtype=torch.bool),
                 torch.ones(2, 1, 4, dtype=torch.bool),
                 torch.zeros(2, 1, 4), torch.zeros(2, 4, dtype=torch.bool)]
        for mask in masks:
            with self.subTest(shape=mask.shape):
                with self.assertRaises(ValueError):
                    model(self.x, static_categorical_values=self.ids, patch_mask=mask)
        for ratio in (0, 1, float("nan")):
            with self.assertRaises(ValueError):
                model(self.x, static_categorical_values=self.ids, mask_ratio=ratio)

    def test_mask_generator_is_reproducible_and_keeps_visible_patches(self):
        model = TinyTimeMixerPretrainModel(self.cfg)
        first = model.make_patch_mask(self.x, 0.99, torch.Generator().manual_seed(3))
        second = model.make_patch_mask(self.x, 0.99, torch.Generator().manual_seed(3))
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.all(first.sum(dim=-1) == 3))

    def test_transfer_preserves_forecast_head_and_accepts_new_horizon(self):
        source = TinyTimeMixerPretrainModel(self.cfg)
        target = TinyTimeMixerModel(replace(self.cfg, horizon=7))
        head = {k: v.clone() for k, v in target.head.state_dict().items()}
        source.transfer_to(target, freeze_backbone=True)
        for name, value in source.backbone.state_dict().items():
            torch.testing.assert_close(value, target.backbone.state_dict()[name])
        for name, value in source.decoder.state_dict().items():
            torch.testing.assert_close(value, target.decoder.state_dict()[name])
        for name, value in head.items():
            torch.testing.assert_close(value, target.head.state_dict()[name])
        self.assertTrue(all(not p.requires_grad for p in target.backbone.parameters()))
        self.assertEqual(target(self.x, static_categorical_values=self.ids).shape, (2, 7, 1))

    def test_transfer_mismatch_does_not_modify_target(self):
        source = TinyTimeMixerPretrainModel(self.cfg)
        target = TinyTimeMixerModel(replace(self.cfg, decoder_d_model=16))
        original = {k: v.clone() for k, v in target.state_dict().items()}
        with self.assertRaises(ValueError):
            source.transfer_to(target)
        for name, value in original.items():
            torch.testing.assert_close(value, target.state_dict()[name])

    def test_weights_and_json_config_roundtrip(self):
        source = TinyTimeMixerModel(self.cfg).eval()
        cfg = json.loads(json.dumps(asdict(self.cfg)))
        target = TinyTimeMixerModel.from_config(cfg).eval()
        stream = io.BytesIO()
        torch.save(source.state_dict(), stream)
        stream.seek(0)
        target.load_state_dict(torch.load(stream, weights_only=True), strict=True)
        torch.testing.assert_close(
            source(self.x, static_categorical_values=self.ids),
            target(self.x, static_categorical_values=self.ids),
        )

    def test_module_does_not_import_public_model_dependencies(self):
        # 불필요한 공개 모델 호출·다운로드 의존이 새 코드에 들어오는 것을 막는다.
        import ast
        import modeling_module.models.TinyTimeMixer as package
        for path in Path(package.__file__).parent.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    imports.append(node.module or "")
            self.assertFalse(any(name.split(".")[0] in {"transformers", "tsfm_public", "granite_tsfm"}
                                 for name in imports), str(path))


if __name__ == "__main__":
    unittest.main()
