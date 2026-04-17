# 🧩 PatchTST Self-Supervised(Pretrain) 코드 레벨 분석

아래 내용은 현재 관형님 코드(`PatchTSTPretrainModel`, `PatchTSTSelfSupBackbone`, `train_patchtst_pretrain`)에서 **self-supervised가 실제로 어떻게 구성되고 학습되는지**를 *코드 흐름 그대로* 정리한 문서입니다.  
핵심 과제는 **Masked Patch Reconstruction(MAE/BERT 스타일)** 입니다.

---

## 0) 전체 파이프라인 한 줄 요약

> 입력 시계열 `x`를 패치 토큰 시퀀스로 변환(patchify)한 뒤 일부 패치를 마스킹(mask)하고, Transformer encoder가 문맥으로부터 마스킹된 패치를 복원(reconstruction)하도록 **masked-only reconstruction loss(MSE/MAE)** 로 학습한다.

---

## 1) Top-Down 호출 흐름(학습 루프에서 실제 호출)

학습 루프에서 self-supervised를 수행하는 핵심 호출은 다음입니다.

```python
out = pre_model(x, mask_ratio=0.3, return_loss=True, loss_type="mse")
loss = out["loss"]
loss.backward()
optimizer.step()
```

이 한 줄 호출 안에서 모델은 다음을 수행합니다.</br>

	1.	입력 x 레이아웃 정리 ([B,C,L] / [B,L,C])</br>
	2.	(옵션) RevIN 정규화</br>
	3.	patchify로 패치 토큰 생성 ([B,N,D])</br>
	4.	패치 일부 마스킹(mask)</br>
	5.	Transformer encoder로 문맥화(contextualize)</br>
	6.	복원 head로 원래 패치 값 복원</br>
	7.	마스킹된 위치에서만 loss 계산(masked-only)</br>
---

## 2) PatchTSTPretrainModel 코드 레벨 동작 (PatchTST.py)
### 2.1 __init__(cfg): pretrain 구성요소 세팅
    PatchTSTPretrainModel은 self-supervised용 backbone을 감싸는 래퍼 역할입니다.</br>
        •	self.n_vars: 입력 채널 수(= C)</br>
	    •	self.use_revin: RevIN 사용 여부
	    •	self.revin_layer: RevIN 모듈(옵션)
	    •	self.backbone: PatchTSTSelfSupBackbone (masking/encoder/decoder 수행)
	    •	mask_ratio, loss_type 기본값 유지

---

### 2.2 _make_patch_mask(B, N, mask_ratio): 배치마다 랜덤 mask 생성
	    •	출력: patch_mask: [B, N] (bool)
	    •	True인 위치 = 가려진 패치(복원해야 하는 타깃)

        예시 동작(개념):
        •	torch.rand(B,N) < mask_ratio 형태로 생성
        •	mask_ratio=0.3이면 평균 30% 패치가 True

---

2.3 forward(x, ...): self-supervised objective의 실제 구성

(1) 입력 레이아웃 정리: [B,C,L] / [B,L,C] 호환
	•	x가 [B,C,L]이면 그대로 사용
	•	x가 [B,L,C]이면 permute(0,2,1)로 [B,C,L] 변환

이 단계에서 cfg.n_vars와 채널 축이 맞지 않으면 ValueError가 발생합니다.

---

(2) RevIN 정규화(옵션)
use_revin=True일 경우:
	•	RevIN이 [B,L,C] 형태를 기대하는 구현이 많으므로
	•	[B,C,L] -> [B,L,C]로 permute
	•	revin_layer(..., "norm")
	•	다시 [B,C,L]로 permute

즉, 정규화된 시계열 공간에서 복원 loss를 학습합니다.

---

(3) patchify로 “정답(target)” 생성
```python
patches_target, N = self.backbone.patchify(x_n)  # [B, N, C*patch_len]
```
	•	외부 라벨 y가 아니라, 입력 x에서 잘라낸 패치 자체가 정답입니다.
	•	self-supervised의 핵심: “정답이 입력으로부터 자동 생성됨”

---

(4) mask 생성 또는 입력된 mask 사용
	•	patch_mask가 없으면 _make_patch_mask()로 생성
	•	있으면 shape/dtype 보정 후 사용

---

(5) backbone 호출: 마스킹 복원 실행
```python
z, patches_pred = self.backbone.forward_from_patches(
    patches=patches_target,
    patch_mask=patch_mask
)
```
	•	z: encoder 출력 (문맥화된 표현) [B, N, d_model]
	•	patches_pred: 복원 패치 값 [B, N, C*patch_len]

⸻

(6) loss 계산: masked-only reconstruction loss
```python
loss = _masked_recon_loss(patches_pred, patches_target, patch_mask, loss_type="mse")
```

	•	mask=True인 패치 위치에서만 오차를 계산합니다.
	•	이 설계가 없으면 모델이 trivial하게 입력을 통과시키는 방향으로 학습될 수 있어 신호가 약해집니다.

⸻

(7) 디버깅/검증용 reshape(선택)
복원 결과를 보기 좋게 다음 형태로 바꾸는 경우가 많습니다.
	•	patch_pred: [B, C, N, patch_len]
	•	patch_target: [B, C, N, patch_len]

패치별 복원 품질 시각화에 유리합니다.

⸻

2.4 _masked_recon_loss(pred, target, mask): masked-only loss 구현 포인트

입력:
	•	pred, target: [B, N, D]
	•	mask: [B, N]

핵심 구현:
	•	mask.unsqueeze(-1) -> [B,N,1]로 확장
	•	per_elem = (pred-target)^2 또는 abs(pred-target)
	•	masked = per_elem * mask
	•	분모는 “mask된 원소 수” 기준으로 평균 (0 방지 위해 clamp/eps 적용)

⸻

2.5 export_encoder_state(): supervised로 이식할 weight만 추출

보통 다음 prefix만 추립니다.
	•	backbone.patch_embed.*
	•	backbone.encoder.*
	•	backbone.norm_out.*

이유:
	•	forecasting head는 task마다 달라서 새로 학습하는 경우가 많음
	•	핵심 표현학습 부품(embedding/encoder/norm)을 이식하려는 목적

⸻

3) PatchTSTSelfSupBackbone 코드 레벨 동작 (backbone.py)

3.1 patchify(x): 시계열을 패치 토큰 시퀀스로 변환

입력:
	•	x: [B, C, L]

출력:
	•	patches: [B, N, D] (D = C*patch_len)

동작(개념):
	1.	unfold로 슬라이딩 윈도우 생성: [B, C, N, patch_len]
	2.	reshape: [B, N, C*patch_len]

즉, 시계열이 N개의 토큰(패치) 로 토크나이즈됩니다.

⸻

3.2 mask_token: 마스킹된 패치를 대체하는 학습 파라미터
```python
self.mask_token = nn.Parameter(torch.zeros(1, 1, C*patch_len))
```
마스킹 위치는 원래 패치 대신 mask_token이 들어갑니다(learnable placeholder).

⸻

3.3 forward_from_patches(patches, patch_mask): masking → encoding → decoding

(1) 마스킹 적용
```python
patches_masked = torch.where(
    patch_mask.unsqueeze(-1),
    self.mask_token.expand(B, N, D),
    patches
)
```
	•	mask=True 위치는 실제 값이 제거되고 mask_token만 남습니다.
	•	모델은 문맥(주변 패치)으로 해당 값을 복원해야 합니다.

---

(2) patch embedding + positional encoding
```python
z = self.patch_embed(patches_masked)    # Linear(D -> d_model)
z = z + pos_emb                         # 위치정보 추가
z = dropout(z)
```

⸻

(3) Transformer encoder
```python
z = self.encoder(z)                     # [B, N, d_model]
z = self.norm_out(z)
```
	•	각 패치 토큰이 주변 패치 정보를 attention으로 모아 문맥화됩니다.

⸻

(4) pretrain head로 패치 공간 복원
```python
patch_pred = self.pretrain_head(z)      # Linear(d_model -> D)
```
	•	patch_pred: [B, N, D]가 복원 결과입니다.

⸻

4) train_patchtst_pretrain()에서 self-supervised가 학습 루프와 연결되는 방식

학습 루프는 다음 단계로 self-supervised를 수행합니다.
	1.	배치에서 x만 추출 (_extract_x(batch))
	2.	model(x, mask_ratio=..., return_loss=True) 호출
	3.	loss.backward() → optimizer step
	4.	val_loader가 있으면 동일 objective로 val loss 측정

중요:
	•	supervised에서 쓰는 y는 전혀 사용하지 않습니다.
	•	입력 x로부터 target patch를 생성하고(masked reconstruction) 그것으로 학습합니다.

⸻

5) 관형님 로그가 의미하는 것(코드 관점)

train/val loss가 안정적으로 감소한다는 것은 아래가 모두 정상이라는 뜻입니다.
	•	mask 생성/적용이 정상 동작
	•	mask_token 치환 로직 정상
	•	patch_embed/encoder/head forward/backward 정상
	•	masked-only loss 계산이 올바름
	•	optimizer 업데이트 정상

⸻

6) 코드 분석 관점에서 추가로 체크하면 좋은 2가지
	1.	마스킹 개수(분모)가 0이 되는 배치가 없는지
	•	mask_ratio가 너무 작으면 학습 신호 약화 가능
	•	일반적으로 0.3이면 안정적
	2.	RevIN 적용 위치가 의도와 일치하는지
	•	현재는 RevIN으로 정규화된 x_n을 patchify하여 loss 계산
	•	스케일보다 패턴에 집중시키려는 의도라면 합리적

⸻

✅ 결론(코드 레벨 정의)

현재 코드는 입력 x를 patchify하여 토큰 시퀀스로 만든 뒤, 일부 토큰을 mask_token으로 대체하고 Transformer encoder로 문맥 표현을 만든 후, head로 원래 patch 값을 복원하도록 학습하며, loss는 마스킹된 패치 위치에서만 계산하는 Masked Patch Reconstruction 기반 self-supervised pretraining이다.