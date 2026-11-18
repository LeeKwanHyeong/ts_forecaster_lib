"""지정된 커밋의 실제 학습 함수를 관측만 하며 실행하는 독립 프로세스."""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import importlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

if __package__:
    from .common import SETTINGS, save_json, scores, sha256
else:
    from common import SETTINGS, save_json, scores, sha256


class Windows(Dataset):
    def __init__(self, data, split):
        self.x = torch.from_numpy(data[f"{split}_x"])
        self.y = torch.from_numpy(data[f"{split}_y"])
        self.future = torch.from_numpy(data[f"{split}_f"])
        self.ids = data["parts"][data[f"{split}_part"]].tolist()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        # 삼성 브랜치의 실제 6개 필드 배치 계약. 과거 외생변수는 이번 비교에서 미사용.
        return (self.x[index], self.y[index], self.ids[index], self.future[index],
                torch.empty(self.x.shape[1], 0), torch.empty(self.x.shape[1], 0, dtype=torch.long))


def collate(rows):
    columns = tuple(zip(*rows))
    return (torch.stack(columns[0]), torch.stack(columns[1]), list(columns[2]),
            torch.stack(columns[3]), torch.stack(columns[4]), torch.stack(columns[5]))


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def state_hash(state):
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode())
        digest.update(state[key].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def shared_state_hash(model):
    return state_hash({k: v for k, v in model.state_dict().items() if not k.startswith("revin_layer.affine_")})


@torch.no_grad()
def reconstruction(model, x):
    """학습 중 바뀌지 않는 동일 마스크와 affine 제거 좌표계로 재구성 품질 비교."""
    training = model.training
    model.eval()
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1729)
            count = (x.shape[1] - model.patch_len) // model.stride + 1
            mask = torch.rand(len(x), count) < SETTINGS["mask_ratio"]
            mask[:, 0] = True
            out = model(x, patch_mask=mask, return_loss=True)
            revin = model.revin_layer
            gamma = revin.affine_weight.detach() if revin.affine else torch.ones(1)
            beta = revin.affine_bias.detach() if revin.affine else torch.zeros(1)
            result = {"fixed_mask_native_mse": float(out["loss"]), "affine_gamma": float(gamma.mean()),
                      "affine_beta": float(beta.mean()), "gamma_min_abs": float(gamma.abs().min())}
            if result["gamma_min_abs"] < 1e-8:
                result.update(fixed_scale_reconstruction_mse=None, raw_reconstruction_mae=None)
                return result
            mean = x.mean(dim=1, keepdim=True)
            std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + revin.eps)
            normalized = ((x - mean) / std).permute(0, 2, 1)
            target, _ = model.backbone.patchify(normalized)
            pred = out["patch_pred"].permute(0, 2, 1, 3).flatten(2)
            pred = (pred - beta) / gamma
            selected = mask.unsqueeze(-1).expand_as(target)
            difference = pred - target
            result["fixed_scale_reconstruction_mse"] = float(difference.square()[selected].mean())
            result["raw_reconstruction_mae"] = float((difference.abs() * std[:, :, :1])[selected].mean())
            return result
    finally:
        model.train(training)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", required=True, choices=["before", "after"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--pretrain-lr", type=float, default=SETTINGS["lr"])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.code_root = args.code_root.expanduser().resolve()
    args.data = args.data.expanduser().resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    os.chdir(args.output)
    sys.path.insert(0, str(args.code_root.resolve() / "src"))
    from modeling_module.models.PatchTST.common.configs import AttentionConfig, PatchTSTConfig
    from modeling_module.models.PatchTST.self_supervised.PatchTST import PatchTSTPretrainModel
    from modeling_module.models.PatchTST.supervised.PatchTST import PatchTSTModel
    from modeling_module.training.config import TrainingConfig, StageConfig
    from modeling_module.training.model_losses.loss_module import SpikeWeightedHuberLoss
    from modeling_module.training.model_trainers.patchtst_finetune import train_patchtst_finetune

    pre_module = importlib.import_module("modeling_module.training.model_trainers.patchtst_pretrain")
    ft_module = importlib.import_module("modeling_module.training.model_trainers.patchtst_train")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    data = np.load(args.data, allow_pickle=False)
    s = SETTINGS
    cfg = TrainingConfig(device="cpu", use_amp=False, lookback=s["lookback"], horizon=s["horizon"],
                         lr=s["lr"], weight_decay=s["weight_decay"], patience=100, t_max=s["warmup_epochs"],
                         loss=SpikeWeightedHuberLoss(delta=5.0), use_exogenous_mode=True, exo_is_normalized=True)
    cfg.grad_clip = 1.0
    cfg.log_every = 10000
    cfg.spike_loss.enabled = False
    pre_cfg = PatchTSTConfig(lookback=s["lookback"], horizon=s["horizon"], patch_len=s["patch_len"],
                            stride=s["patch_stride"], d_model=s["d_model"], d_ff=s["d_ff"],
                            n_layers=s["n_layers"], dropout=s["dropout"], norm=s["norm"],
                            future_exo_dim=0, head_dropout=s["dropout"], loss=copy.deepcopy(cfg.loss))
    pre_cfg.attn = AttentionConfig(n_heads=s["n_heads"], d_model=s["d_model"], attn_dropout=s["dropout"], proj_dropout=s["dropout"])
    seed_all(args.seed)
    pre_model = PatchTSTPretrainModel(pre_cfg)
    initial_hash = shared_state_hash(pre_model)
    torch.save(pre_model.state_dict(), args.output / "pretrain_initial.pt")

    def loaders(seed):
        return [DataLoader(Windows(data, split), batch_size=s["batch_size"], shuffle=(split == "train"),
                           num_workers=0, collate_fn=collate, drop_last=False,
                           generator=torch.Generator().manual_seed(seed + index))
                for index, split in enumerate(("train", "val", "test"))]

    train, val, _ = loaders(args.seed + 1_000)
    val_x = torch.from_numpy(data["val_x"])
    traces, train_losses = [], []
    observing = False

    def hook(module, inputs, output):
        if module.training and not observing:
            train_losses.append(float(output["loss"].detach()))

    handle = pre_model.register_forward_hook(hook)
    original_eval = pre_module._eval_pretrain

    def observed_eval(model, loader, device, **kwargs):
        nonlocal observing
        loss = original_eval(model, loader, device, **kwargs)
        observing = True
        try:
            row = {"epoch": len(traces) + 1, "train_loss": float(np.mean(train_losses)), "validation_loss": loss,
                   **reconstruction(model, val_x)}
        finally:
            observing = False
        traces.append(row)
        train_losses.clear()
        save_json(args.output / "pretrain_history.json", traces)
        print("[AB-PRETRAIN]", json.dumps(row), flush=True)
        return loss

    pre_module._eval_pretrain = observed_eval
    start = time.perf_counter()
    print("[AB] PRETRAIN START", args.variant, args.seed, args.pretrain_lr, flush=True)
    result = pre_module.train_patchtst_pretrain(
        pre_model, train, val, device="cpu", train_cfg=cfg,
        stages=[StageConfig(epochs=2 if args.smoke else s["pretrain_epochs"], lr=args.pretrain_lr, spike_enabled=False)],
        mask_ratio=s["mask_ratio"], loss_type="mse", save_dir=str(args.output / "pretrain"))
    pretrain_seconds = time.perf_counter() - start
    handle.remove()
    best_pretrain = reconstruction(pre_model, val_x)
    checkpoint = args.output / "pretrain" / "patchtst_pretrain_best.pt"
    best_pre_epoch = int(np.argmin([r["validation_loss"] for r in traces]) + 1)

    # FT 초기화와 배치 순서의 난수를 재설정하여 SSL의 난수 소비 차이를 분리.
    seed_all(args.seed + 10_000)
    fine_cfg = copy.deepcopy(pre_cfg)
    fine_cfg.future_exo_dim = 4
    fine_model = PatchTSTModel(fine_cfg)
    initial_ft_hash = shared_state_hash(fine_model)
    train, val, test = loaders(args.seed + 20_000)
    ft_history = []
    original_trainer = ft_module.CommonTrainer

    class ObservedTrainer(original_trainer):
        def __init__(self, *a, **kw):
            log = kw.get("logger", print)
            self.validation_losses = []
            self.current_train_loss = None
            def observed_log(message):
                log(message)
                if isinstance(message, str) and message.startswith("Epoch "):
                    row = {"global_epoch": len(ft_history) + 1, "train_loss": self.current_train_loss,
                           "validation_loss": float(np.mean(self.validation_losses)),
                           "spike_enabled": self.cfg.spike_loss.enabled,
                           "lr_next_epoch": float(self.sched.get_last_lr()[0])}
                    ft_history.append(row)
                    self.validation_losses.clear()
                    save_json(args.output / "finetune_history.json", ft_history)
                    print("[AB-FINETUNE]", json.dumps(row), flush=True)
            kw["logger"] = observed_log
            super().__init__(*a, **kw)
            compute = self.loss_comp.compute
            def observed_compute(*ca, **ck):
                value = compute(*ca, **ck)
                if ck.get("is_val", False):
                    self.validation_losses.append(float(value.detach()) if torch.is_tensor(value) else float(value))
                return value
            self.loss_comp.compute = observed_compute

        def _run_epoch(self, model, loader, *, train=True):
            value = super()._run_epoch(model, loader, train=train)
            if train:
                self.current_train_loss = value
            return value

    ft_module.CommonTrainer = ObservedTrainer
    stages = [StageConfig(epochs=2 if args.smoke else s["warmup_epochs"], lr=s["lr"], spike_enabled=False)]
    if not args.smoke:
        stages.append(StageConfig(epochs=s["spike_epochs"], lr=s["spike_lr"], spike_enabled=True))
    start = time.perf_counter()
    print("[AB] FINETUNE START", flush=True)
    fine_result = train_patchtst_finetune(
        fine_model, train, val, device="cpu", train_cfg=cfg, stages=stages,
        pretrain_ckpt_path=str(checkpoint), freeze_encoder_before_ft=False, load_strict=False)
    finetune_seconds = time.perf_counter() - start
    fine_model = fine_result["model"].eval()
    torch.save(fine_model.state_dict(), args.output / "finetune_best.pt")
    predictions = []
    # 실행 점검에는 검증 정답만 사용하고 최종 테스트로 학습 설정을 조정하지 않는 정책.
    evaluation_loader, split = (val, "val") if args.smoke else (test, "test")
    with torch.no_grad():
        for batch in evaluation_loader:
            prediction = fine_model(batch[0], future_exo=batch[3])
            predictions.append(prediction.detach().cpu().numpy().reshape(len(batch[0]), s["horizon"]))
    prediction = np.concatenate(predictions)
    truth = data[f"{split}_y"].squeeze(-1)
    metrics = scores(truth, prediction)
    np.savez_compressed(args.output / "predictions.npz", actual=truth, prediction=prediction,
                        part_indices=data[f"{split}_part"], x=data[f"{split}_x"].squeeze(-1))
    native_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.code_root, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=args.code_root, text=True).strip())
    summary = {"variant": args.variant, "seed": args.seed, "pretrain_lr": args.pretrain_lr,
               "code_commit": native_commit, "code_tree_dirty": dirty, "data_sha256": sha256(args.data),
               "initial_shared_pretrain_hash": initial_hash, "initial_finetune_hash": initial_ft_hash,
               "pretrain_best_val": float(result["best_val"]), "pretrain_best_epoch": best_pre_epoch,
               "pretrain_best_diagnostics": best_pretrain, "pretrain_last_diagnostics": traces[-1],
               "finetune_best_val": float(fine_result["best_val_loss"]), "test_metrics": metrics,
               "clipped_test_metrics": scores(truth, np.maximum(prediction, 0)),
               "pretrain_seconds": pretrain_seconds, "finetune_seconds": finetune_seconds,
               "smoke": args.smoke, "torch": torch.__version__, "python": sys.version,
               "settings": {**s, "pretrain_lr": args.pretrain_lr}}
    save_json(args.output / "summary.json", summary)
    print("[AB] COMPLETE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
