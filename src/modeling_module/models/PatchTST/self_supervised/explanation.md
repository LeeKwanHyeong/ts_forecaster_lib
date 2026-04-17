# PatchTST Self-Supervised

## 1) 이 코드에서 self-supervised란 무엇인가

이 파이프라인의 self-supervised는 요약하면 다음입니다.
	•	외부 라벨 y(미래 정답)를 쓰지 않습니다.
	•	대신 입력 시계열 x 자체를 잘라서(patchify) 일부를 가리고(mask),
	•	모델이 그 가려진 부분을 복원(reconstruction) 하게 합니다.
	•	즉, “정답”은 x 안에 이미 존재합니다.
→ 입력이 곧 정답을 만드는 원천이므로 self-supervised 입니다.

이때 학습 목적함수(loss)는:

“마스킹된 패치 위치에서의 복원 오차(MSE/MAE)를 최소화”

입니다.

## 2) 데이터가 모델로 들어가는 형태와 전처리(RevIN 포함)

### 2.1 입력 텐서 형상

데이터로더에서 x는 일반적으로 3차원 텐서입니다.
	•	[B, C, L] 또는 [B, L, C]
	•	B: batch size
	•	C: 변수(채널) 수 (관형님 케이스는 C=1)
	•	L: lookback 길이(예: 52)

관형님은 처음에 cfg.n_vars=1인데 x.shape[1]=52라서 에러가 났고, 이는 로더가 [B, L, C] 형태였기 때문입니다.
지금은 모델 forward에서 [B, C, L]로 통일해서 처리하도록 수정해 두셨기 때문에 문제 없이 학습이 진행된 상태입니다.

### 2.2 RevIN(선택) 적용

코드에서는 use_revin=True일 경우 RevIN을 적용합니다.
	•	RevIN의 역할:
	•	part별/샘플별로 스케일·오프셋이 크게 다른 시계열에서
	•	모델이 “스케일”이 아니라 “패턴”을 학습하도록 돕는 정규화 기법입니다.
	•	self-supervised에서도 유의미합니다.
	•	복원 과제는 입력을 그대로 맞추는 문제라서, 스케일 차이가 크면 학습이 스케일에 끌려갈 수 있습니다.
	•	RevIN을 쓰면 패턴(상승/하락/계절성/변동성)을 더 안정적으로 학습하는 경향이 있습니다.


## 3) Patchify: 시계열을 “토큰”으로 만드는 과정

self-supervised 핵심은 “시계열을 patch 단위 토큰 시퀀스로 바꾸고, 일부 토큰을 숨긴다”입니다.

### 3.1 Patchify 동작

Patchify는 x: [B, C, L]를 다음으로 바꿉니다.
	•	patches: [B, N, C * patch_len]
	•	여기서
	•	patch_len: 한 패치가 포함하는 시간 길이
	•	stride: 패치가 이동하는 간격
	•	N: 패치 개수
N = \left\lfloor \frac{L - patch\_len}{stride} \right\rfloor + 1

즉, 시계열이 N개의 토큰(패치) 으로 변환됩니다.
PatchTST 계열은 이 패치 시퀀스를 Transformer 입력 시퀀스로 보고 학습합니다.

## 4) Masked Patch Reconstruction: “자기지도 과제”를 만드는 핵심

### 4.1 마스킹은 어떻게 만드나

매 배치마다 [B, N] boolean mask를 만듭니다.
	•	patch_mask[b, n] = True
→ b번째 샘플의 n번째 패치는 “가려짐(예측해야 함)”
	•	mask_ratio=0.3이면 평균적으로 30% 패치를 가립니다.

중요 포인트:
	•	마스킹 패턴은 매 step/epoch마다 랜덤하게 바뀝니다.
	•	따라서 모델은 “특정 위치만 외우는 것”이 아니라,
	•	앞뒤 문맥 패치,
	•	장기/단기 패턴,
	•	반복 구조(계절성),
	•	급변 패턴의 전후 관계
를 이용해 복원하는 능력을 키우게 됩니다.

### 4.2 마스킹된 입력은 어떻게 모델에 넣나

가려진 패치는 실제 값 대신 학습 가능한 mask token으로 치환됩니다.
	•	mask_token: [1, 1, C*patch_len] (학습 파라미터)
	•	마스킹 위치는 mask_token으로 대체
	•	비마스킹 위치는 원래 패치 값을 그대로 유지

이 구조는 NLP의 BERT류, 비전의 MAE류와 동일한 패턴입니다.


## 5) Encoder: 패치 시퀀스를 문맥 표현으로 인코딩

### 5.1 Patch embedding

패치는 길이가 C*patch_len인 벡터입니다. 이를 d_model로 올립니다.
	•	patch_embed: Linear(C*patch_len → d_model)
	•	결과: z0: [B, N, d_model]

### 5.2 Positional encoding

Transformer는 순서를 모르므로 위치정보를 더합니다.
	•	z0 = z0 + pos_emb

### 5.3 Transformer encoder

그 다음 z0를 Transformer encoder에 통과시켜 문맥화된 표현을 얻습니다.
	•	z: [B, N, d_model]

여기서 중요한 의미는:
	•	각 패치 토큰의 표현 z[:, n, :]는
	•	주변 패치(이전/이후)의 정보를 attention으로 끌어와서
	•	“문맥적 의미”를 담은 표현이 됩니다.
	•	마스킹된 패치도 결국 주변 문맥으로부터 복원될 수 있도록
	•	해당 위치 토큰의 표현이 주변 패치의 신호를 모으도록 학습됩니다.

## 6) Decoder(Head): 문맥 표현을 다시 “패치 값”으로 복원

encoder 출력 z를 다시 원래 패치 공간으로 내립니다.
	•	pretrain_head: Linear(d_model → C*patch_len)
	•	patch_pred: [B, N, C*patch_len]

즉, 각 패치 토큰마다 “복원된 패치 벡터”가 출력됩니다.

## 7) Loss: “가려진 패치에서만” 복원 오차를 계산

여기가 self-supervised 목적의 핵심입니다.

### 7.1 타깃(target)은 무엇인가

타깃은 외부 라벨이 아니라 원래 입력에서 잘라낸 패치 자체입니다.
	•	patch_target = patchify(x)
	•	patch_pred = model(patch_masked_input)

즉, 입력 x가 스스로 정답을 제공합니다.

### 7.2 손실은 어디에서만 계산하나

손실은 masked 위치에서만 계산합니다.
	•	mask=True인 패치들에 대해서만
	•	MSE(pred, target) 또는 MAE(pred, target)
	•	mask=False인 패치(안 가린 패치)는 loss에 거의 기여하지 않습니다.

왜냐하면, 만약 안 가린 패치까지 포함해버리면 모델이 trivial하게 “그냥 그대로 통과”하는 방향으로 최적화될 수 있고, 마스킹 복원이라는 학습 신호가 약해집니다.
그래서 MAE류는 보통 masked-only loss가 기본입니다.

## 8) 학습 루프 관점에서 “self-supervised가 어떻게 수행되는가”

train_patchtst_pretrain()에서 실제로 일어나는 것은 다음입니다.</br>
	1.	train_loader에서 batch를 받음</br>
    	•	여기서는 기존 supervised용 loader를 그대로 써도 되고,</br>
    	•	pretrain 쪽은 batch에서 x만 꺼내 씁니다.</br>
	2.	out = pre_model(x, mask_ratio=0.3, return_loss=True)</br>
	•	모델 내부에서:</br>
	•	patchify</br>
	•	mask 생성/적용(또는 외부 mask 입력)</br>
	•	encoder</br>
	•	head로 patch 복원</br>
	•	masked-only reconstruction loss 계산</br>
	3.	loss.backward()</br>
	4.	optimizer step</br>

즉, y가 전혀 없어도 학습이 진행됩니다.</br>
이게 self-supervised의 구조적 정의입니다.


## 9) 왜 loss가 이렇게 깔끔하게 떨어졌나(로그 해석)

관형님 로그는 다음 특성을 보입니다.
	•	train loss: 0.88 → 0.16
	•	val loss: 0.77 → 0.15
	•	둘 다 안정적으로 감소

이는 보통 다음을 의미합니다.
	•	마스킹/복원 task가 제대로 구성되었고
	•	모델이 패치 간 문맥관계를 학습하면서
	•	“가려진 부분을 주변 패치로부터 추정”하는 능력이 개선되고 있음

또한 val이 train보다 약간 낮게 나오는 현상은 흔히 발생합니다.
	•	dropout/augmentation 차이
	•	train은 매 step mask가 바뀌어 더 어려운 평균 난이도를 갖는 경우
	•	val은 상대적으로 안정적인 난이도 분포


