from dataclasses import dataclass, field
from typing import Sequence, Dict, List, Optional, Callable, Tuple

import torch
import polars as pl
import numpy as np

@dataclass
class FutureScenarioStore:
    """
    Scenario Table을 (uid, dt_idx) -> feature_vector 로 인덱싱해둔 조회기
    """
    id_col: str
    date_col: str
    idx_col: str           # 예: "dt_idx"
    feat_cols: Sequence[str]
    table: pl.DataFrame

    def __post_init__(self):
        need = {self.id_col, self.idx_col, *self.feat_cols}
        missing = [c for c in need if c not in self.table.columns]
        if missing:
            raise KeyError(f"Scenario table missing columns: {missing}")

        # uid -> {dt_idx: feat_vector} dict
        self.map: Dict[str, Dict[int, np.ndarray]] = {}

        # 빠른 조회를 위해 파티셔닝
        for g in self.table.partition_by(self.id_col):
            uid = str(g[self.id_col][0])
            idxs = g[self.idx_col].to_list()
            feats = g.select(self.feat_cols).to_numpy()
            self.map[uid] = {int(i): feats[k].astype(np.float32) for k, i in enumerate(idxs)}

        self.feat_dim = len(self.feat_cols)

    def get_batch(self, uids: List[str], start_idxs: List[int], H: int, *, missing_policy: str = "error") -> np.ndarray:
        """
        반환: (B,H,F_manual)
        missing_policy:
          - "error": 한 칸이라도 비면 에러
          - "zero" : 비는 칸은 0으로 채움
        """
        B = len(uids)
        out = np.zeros((B, H, self.feat_dim), dtype=np.float32)

        for b, (uid, s) in enumerate(zip(uids, start_idxs)):
            m = self.map.get(str(uid), None)
            if m is None:
                if missing_policy == "error":
                    raise KeyError(f"Scenario missing uid={uid}")
                continue

            for k in range(H):
                key = int(s) + k
                v = m.get(key, None)
                if v is None:
                    if missing_policy == "error":
                        raise KeyError(f"Scenario missing (uid={uid}, dt_idx={key})")
                    # zero 정책이면 이미 0
                else:
                    out[b, k, :] = v

        return out


@dataclass
class TrainCollateWithFutureExo:
    """
    학습 배치 생성 시 Future Exogenous(미래 외생 변수) 데이터를 동적으로 생성 및 병합하는 Collate 클래스.

    특징:
      - 캐싱(Caching): 빈번히 조회되는 시점의 외생 변수 데이터를 메모리에 저장하여 연산 부하 감소.
      - 배치 처리 지원: 콜백 함수가 배치 입력을 지원할 경우 한 번에 생성, 실패 시 개별 루프로 Fallback.
    """
    horizon: int
    future_exo_cb: Optional[Callable] = None
    part_future_exo_fn: Optional[Callable] = None

    scenario_store: Optional[FutureScenarioStore] = None
    scenario_mode: str = "append"  # "append" | "replace"
    scenario_missing_policy: str = "error"  # "error" | "zero"

    cache_size: int = 15000
    cache: Dict[Tuple[str, int], np.ndarray] = field(default_factory=dict)
    cache_keys: List[Tuple[str, int]] = field(default_factory=list)

    def _cache_get(self, k: Tuple[str, int]) -> Optional[np.ndarray]:
        return self.cache.get(k, None)

    def _cache_put(self, k: Tuple[str, int], v: np.ndarray):
        if self.cache_size <= 0:
            return
        if k in self.cache:
            return
        self.cache[k] = v
        self.cache_keys.append(k)
        if len(self.cache_keys) > self.cache_size:
            old = self.cache_keys.pop(0)
            self.cache.pop(old, None)

    @staticmethod
    def _stack_float(items: Sequence[object]) -> torch.Tensor:
        """Stack tensor or NumPy samples as a float32 tensor."""
        arrays = [item.detach().cpu().numpy() if torch.is_tensor(item) else np.asarray(item) for item in items]
        return torch.from_numpy(np.stack(arrays).astype(np.float32, copy=False))

    @staticmethod
    def _stack_int(items: Sequence[object]) -> torch.Tensor:
        """Stack tensor or NumPy samples as an int64 tensor."""
        arrays = [item.detach().cpu().numpy() if torch.is_tensor(item) else np.asarray(item) for item in items]
        return torch.from_numpy(np.stack(arrays).astype(np.int64, copy=False))

    @staticmethod
    def _future_array(value: object, batch_size: int, horizon: int, source: str) -> np.ndarray:
        """Validate and normalize a future-exogenous batch to ``(B, H, E)``."""
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        array = np.asarray(value, dtype=np.float32)
        if array.ndim == 2 and batch_size == 1:
            array = array[None, ...]
        if array.ndim != 3 or array.shape[:2] != (batch_size, horizon):
            raise ValueError(f"{source} must return ({batch_size}, {horizon}, E); got {array.shape}")
        return array.astype(np.float32, copy=False)

    def __call__(self, batch):
        """Collate target, series IDs, and all supported exogenous payloads."""
        if not batch:
            raise ValueError("Cannot collate an empty batch.")

        sample_lengths = {len(sample) for sample in batch}
        if len(sample_lengths) != 1:
            raise ValueError(
                "All exogenous Dataset samples in one batch must use the same "
                f"tuple length; got {sorted(sample_lengths)}."
            )
        sample_length = sample_lengths.pop()
        if sample_length == 6:
            xs, ys, uids, start_idxs, pe_conts, pe_cats = zip(*batch)
            future_cats = None
        elif sample_length == 7:
            (
                xs,
                ys,
                uids,
                start_idxs,
                pe_conts,
                pe_cats,
                future_cats,
            ) = zip(*batch)
        else:
            raise ValueError(
                "Exogenous Dataset samples must be 6-tuples or 7-tuples; "
                f"got length {sample_length}."
            )

        x = self._stack_float(xs)
        y = self._stack_float(ys)
        pe_cont = self._stack_float(pe_conts)
        pe_cat = self._stack_int(pe_cats)
        uid_list = [str(u) for u in uids]

        B = len(start_idxs)
        H = int(self.horizon)
        future_cat = None
        if future_cats is not None:
            future_cat = self._stack_int(future_cats)
            if future_cat.ndim != 3 or tuple(future_cat.shape[:2]) != (B, H):
                raise ValueError(
                    "future_cat payload must be shaped "
                    f"(B,H,E); got {tuple(future_cat.shape)}."
                )

        def finalize(fe: torch.Tensor):
            output = (x, y, uid_list, fe, pe_cont, pe_cat)
            if future_cat is not None:
                return (*output, future_cat)
            return output

        first_payload = start_idxs[0]
        if torch.is_tensor(first_payload) or isinstance(first_payload, np.ndarray):
            fe = self._stack_float(start_idxs)
            if fe.ndim != 3 or tuple(fe.shape[:2]) != (B, H):
                raise ValueError(f"future_exo payload must be shaped (B,H,E); got {tuple(fe.shape)}")
            return finalize(fe)

        # -----------------------------------------
        # 1) auto fe (callback)
        # -----------------------------------------
        fe_auto = None
        if self.future_exo_cb is not None:
            # 기존 코드와 동일하게 miss 모아서 batch 호출 (생략 가능)
            # 여기서는 간단 버전만:
            miss = []
            miss_pos = []
            fe_list = []

            for bi, (uid, s) in enumerate(zip(uid_list, start_idxs)):
                key = (uid, int(s))
                cached = self._cache_get(key)
                if cached is None:
                    miss.append(int(s))     # auto fe는 uid가 필요 없다고 가정(캘린더 등)
                    miss_pos.append(bi)
                    fe_list.append(None)
                else:
                    fe_list.append(cached)

            if miss:
                try:
                    res = self._future_array(
                        self.future_exo_cb(miss, H, device="cpu"),
                        len(miss),
                        H,
                        "future_exo_cb",
                    )
                except (TypeError, ValueError):
                    res = np.concatenate(
                        [
                            self._future_array(
                                self.future_exo_cb(start_idx, H, device="cpu"),
                                1,
                                H,
                                "future_exo_cb",
                            )
                            for start_idx in miss
                        ],
                        axis=0,
                    )

                for k, bi in enumerate(miss_pos):
                    uid = uid_list[bi]
                    s = int(start_idxs[bi])
                    fe_arr = res[k]
                    fe_list[bi] = fe_arr
                    self._cache_put((uid, s), fe_arr)

            fe_auto = torch.from_numpy(np.stack(fe_list, axis=0)).to(torch.float32)  # (B,H,E_auto)

        # -----------------------------------------
        # 2) manual fe (scenario table)
        # -----------------------------------------
        fe_manual = None
        if self.scenario_store is not None:
            fe_man_np = self.scenario_store.get_batch(
                uid_list, [int(s) for s in start_idxs], H,
                missing_policy=self.scenario_missing_policy
            )
            fe_manual = torch.from_numpy(fe_man_np).to(torch.float32)  # (B,H,E_man)

        fe_part = None
        if self.part_future_exo_fn is not None:
            part_value = self.part_future_exo_fn(uid_list, list(start_idxs), H)
            fe_part = torch.from_numpy(self._future_array(part_value, B, H, "part_future_exo_fn"))

        # -----------------------------------------
        # 3) merge
        # -----------------------------------------
        if fe_auto is None and fe_manual is None and fe_part is None:
            fe = torch.zeros((B, H, 0), dtype=torch.float32)
        elif self.scenario_mode == "replace":
            fe = fe_manual if fe_manual is not None else fe_auto
            if fe is None:
                fe = fe_part
        else:  # append
            components = [value for value in (fe_auto, fe_manual, fe_part) if value is not None]
            fe = components[0] if len(components) == 1 else torch.cat(components, dim=-1)

        return finalize(fe)
