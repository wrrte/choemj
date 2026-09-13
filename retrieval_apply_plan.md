> 구현 검토: 큰 방향은 타당하지만 replay 구조, TD 시간 정렬, cache/경계 마스크,
> 실제 terminal discount 및 가중 평균의 정규화에 보완이 필요합니다.
> 아래 원안은 보존했으며, 문서 끝의 "구현 검토 및 보완"과
> [TWISTER/RETRIEVAL.md](TWISTER/RETRIEVAL.md)에 실제 적용 내용을 기록했습니다.

TWISTER에 retrieval을 적용하려면 핵심적으로 **세 부분**을 추가하면 됩니다.

```text
1. World model 학습 중 계산된 latent로 anchor trigger
2. Replay buffer에 hash/context 검색 기능 추가
3. Retrieved context를 RSSM warmup 후 actor/critic imagination에 추가
```

월드 모델 loss 자체는 변경하지 않습니다.

---

## 1. Retrieval 설정과 manager 초기화

`TWISTER` 모델에 `RetrievalContextManager`를 생성합니다.

적절한 위치는 `set_replay_buffer()` 이후 또는 모델 초기화 이후입니다.

```python
self.retrieval_manager = RetrievalContextManager(
    num_envs=self.config.num_envs,
    config=retrieval_config,
    latent_dim=self.config.model_stoch_size * self.config.model_discrete,
)
```

TWISTER의 stochastic latent 차원은 기본적으로:

```text
32 categorical variables × 32 classes = 1024
```

이므로 `retrieval.py`의 hash 입력 차원 `1024`와 맞습니다.

추가 설정 예시는 다음과 같습니다.

```python
self.config.retrieval_enabled = False
self.config.retrieval_context_length = 8
self.config.retrieval_warmup_steps = 5000
self.config.retrieval_max_anchors = 10
self.config.retrieval_max_contexts = 256
```

---

## 2. Replay buffer에 retrieval용 정보 추가

현재 TWISTER replay buffer는 trajectory dictionary 기반입니다.

[ `replay_buffer.py` ](TWISTER/nnet/datasets/replay_buffer.py)

현재 `sample()`은 trajectory tensor만 반환합니다.

```python
return traj
```

하지만 retrieval은 각 transition이 replay buffer의 어디에 있는지 알아야 합니다. 따라서 sample 결과에 다음 metadata를 추가해야 합니다.

```text
trajectory_id
start_index
end_index
environment_id
```

예를 들면:

```python
sample = {
    "data": traj,
    "trajectory_id": traj_id,
    "start_index": start_index,
    "env_index": env_index,
}
```

또는 STORM/Drama와 비슷하게 다음을 반환하는 별도 함수를 만들 수 있습니다.

```python
sample_with_indices()
```

retrieval manager가 필요로 하는 정보는 다음과 같습니다.

```text
obs
action
reward
termination
base_indexes
base_envs
```

TWISTER는 trajectory 단위로 저장하므로, ring buffer를 새로 만들기보다는 **기존 buffer 위에 retrieval adapter를 추가하는 방식**이 적절합니다.

---

## 3. 환경 상호작용 중 hash bucket 구축

`env_step()`에서 현재 관측을 처리할 때 hash key를 등록합니다.

현재 TWISTER의 환경 action 계산 과정에도 이미 encoder와 RSSM이 있습니다.

```text
current observation
→ encoder
→ RSSM
→ policy action
```

여기서 single-frame stochastic latent를 얻어 hash key를 만들 수 있습니다.

개념적으로:

```python
with torch.no_grad():
    encoded = self.encode_obs(current_state, sample_mode="probs")
    retrieval_manager.add_transition(
        pointer=buffer_pointer,
        env_idx=env_idx,
        latent_b=encoded,
    )
```

TWISTER에 다음 wrapper를 추가하면 편합니다.

```python
def encode_obs(self, obs, sample_mode="probs"):
    latent = self.encoder_network(obs)
    sample = self.rssm.sample_stoch(
        latent["stoch"],
        sample_mode=sample_mode,
    )
    return sample.flatten(-2, -1)
```

정확한 sampling 구현은 TWISTER의 RSSM API에 맞춰야 하지만, 목적은 **hash용 stochastic latent만 반환하는 것**입니다.

Hash bucket은 다음 때도 재구축할 수 있습니다.

```text
retrieval warmup 종료 시
representation drift가 커졌을 때
checkpoint resume 직후
```

---

## 4. World model 학습 중 anchor trigger 계산

이 부분은 world model loss를 변경하지 않습니다.

현재 `TWISTER.WorldModel.forward()`에서는 이미 다음을 계산합니다.

```python
latent = self.encoder_network(states)
posts, priors = self.rssm.observe(...)
feats = self.rssm.get_feat(posts)
```

이 `feats`를 그대로 anchor 계산에 사용합니다.

다만 현재 `forward()`는 `feats`를 외부에 반환하지 않으므로, 다음처럼 보관하거나 반환해야 합니다.

```python
self.outer.anchor_feats = feats.detach()
self.outer.anchor_rewards = rewards.detach()
self.outer.anchor_dones = dones.detach()
self.outer.anchor_is_firsts = is_firsts.detach()
```

그 후 최상위 `TWISTER.train_step()`에서 world model update가 끝난 뒤:

```python
with torch.no_grad():
    anchor_values = self.value_network(
        self.anchor_feats
    ).mode().squeeze(-1)

    self.retrieval_manager.add_batch_transitions(
        v_t=anchor_values,
        reward=self.anchor_rewards,
        termination=self.anchor_dones,
        gamma=self.config.gamma,
        base_indexes=sample_indexes,
        base_envs=sample_envs,
        max_buf_len=...,
        skip_len=self.config.retrieval_context_length,
        is_warmup=is_retrieval_warmup,
    )
```

흐름은 다음입니다.

```text
WorldModel.forward()
    ↓
posterior feature 계산
    ↓
world model loss 계산 및 업데이트
    ↓
같은 feature로 value 계산
    ↓
TD-error 계산
    ↓
anchor 등록
```

중요한 점은 `feats`를 다시 계산하지 않는다는 것입니다.

---

## 5. Retrieved context 검색

Actor 학습 직전에 retrieval context를 가져옵니다.

```python
ret_obs, ret_action, ..., ret_weights = (
    self.retrieval_manager.retrieve_contexts(
        self.replay_buffer,
        self,
        max_anchors=...,
        max_contexts=...,
    )
)
```

검색 결과는 대략 다음 형태입니다.

```text
ret_obs:    [B_retrieved, context_length, C, H, W]
ret_action: [B_retrieved, context_length, action_dim]
ret_weights
```

일반 random context도 함께 유지합니다.

```text
random context + retrieved context
```

즉, replay batch 전체를 retrieved sample로 대체하는 것이 아니라, STORM/Drama처럼 일부를 추가하는 구조가 적절합니다.

---

## 6. Retrieved context를 RSSM으로 warmup

이 부분이 hidden state를 만드는 단계입니다.

```python
context_encoded = self.encoder_network(ret_obs)

context_posts, context_priors = self.rssm.observe(
    states=context_encoded,
    prev_actions=ret_action,
    is_firsts=context_is_firsts,
    prev_state=None,
    is_firsts_hidden=None,
)
```

그러면 retrieved context의 마지막 시점에 대한 다음 정보가 생깁니다.

```text
마지막 stochastic latent
RSSM/Transformer hidden state
episode boundary 정보
```

이를 기존 `detached_posts`와 같은 구조로 만듭니다.

```text
현재 random batch의 posterior states
+
retrieved context의 posterior states
```

이후 actor가 사용하는 `self.detached_posts`를 이 combined batch로 교체합니다.

```text
combined detached_posts
→ rssm.imagine(...)
→ actor action
→ imagined reward/value
```

World model은 `eval()` 및 `torch.no_grad()` 상태로 warmup하므로, 이 과정이 world model loss에 gradient를 전달하지 않도록 합니다.

---

## 7. Actor와 Critic loss에 retrieval weight 적용

현재 TWISTER의 actor/critic은 모든 imagination sample을 동일하게 취급합니다.

retrieval을 사용할 경우 weight를 추가로 전달합니다.

```text
random context weight = 1.0
retrieved context weight = retrieval weight
```

Actor에서는:

```python
actor_loss = actor_loss * detached_weights
```

Critic에서는:

```python
value_loss = value_loss * detached_weights
```

를 적용합니다.

World model loss에는 이 weight를 적용하지 않습니다.

```text
World model: 기존 loss 그대로
Actor/Critic: retrieved context weight 적용
```

---

## 8. TWISTER에서 변경될 주요 위치

### `twister.py`

변경할 부분:

- retrieval manager 초기화
- `WorldModel.forward()`에서 anchor feature 노출
- 최상위 `train_step()`에서 anchor trigger 실행
- retrieval context를 RSSM warmup
- `detached_posts`에 retrieved state 추가
- actor/critic loss에 sample weight 적용
- 환경 interaction 중 hash 등록

### `replay_buffer.py`

변경할 부분:

- transition 위치 metadata 관리
- retrieval context 샘플링
- context validity 검사
- trajectory 경계 및 episode termination 검사
- 필요하다면 retrieval adapter 제공

### `retrieval.py`

대부분 재사용 가능하지만, TWISTER buffer API에 맞추어 다음 부분은 조정해야 할 수 있습니다.

- `obs_buffer` 접근
- `action_buffer` 접근
- `max_length`
- `num_envs`
- `last_pointer`
- `_valid_context()`

---

## 권장 구현 순서

### 1단계: Anchor trigger만 추가

```text
world model forward의 feats 재사용
→ value 계산
→ TD-error
→ anchor 등록
```

이 단계에서는 actor/critic batch를 아직 바꾸지 않습니다.

### 2단계: Hash bucket과 retrieval 검증

```text
single-frame latent
→ hash key
→ bucket 등록
→ context 검색
```

검색된 context가 올바른 episode 구간인지 확인합니다.

### 3단계: Retrieved context warmup

```text
retrieved obs/action
→ encoder
→ rssm.observe
→ last posterior + hidden state
```

이 단계에서 tensor shape와 hidden state 구조를 검증합니다.

### 4단계: Actor/Critic batch에 추가

```text
random initial states + retrieved initial states
→ imagination
→ actor/critic update
```

### 5단계: Weight와 warmup 추가

마지막으로:

- `batch_weights`
- retrieval warmup
- global hash rebuild
- checkpoint 저장/복원

을 추가합니다.

## 최종 구조

```text
Replay buffer sample
        ↓
World model 학습
        ↓
이미 계산된 posterior feature 재사용
        ↓
Value 및 TD-error 계산
        ↓
Anchor 등록
        ↓
Random context + retrieved context 검색
        ↓
각 context를 RSSM으로 warmup
        ↓
posterior latent + hidden state 생성
        ↓
Actor imagination
        ↓
Actor/Critic update
```

결론적으로 TWISTER 적용에서 가장 중요한 것은 **world model을 retrieval로 재가중하는 것이 아니라**, world model이 이미 계산한 latent를 anchor 판정에 재사용하고, retrieved context만 RSSM warmup을 거쳐 actor/critic imagination에 추가하는 것입니다.

## 구현 검토 및 보완

1. **World-model loss를 보존하고 actor/critic 시작 상태를 추가하는 방향은 타당합니다.**
   단, retrieval 활성화 후 정책과 데이터 분포가 달라지므로 전체 실행의 미래 world-model
   loss까지 같다는 의미는 아닙니다. 같은 입력과 난수 상태에서 해당 update가 같다는 뜻입니다.

2. **Replay를 에피소드 dictionary로 해석하면 안 됩니다.**
   TWISTER는 길이 `L`의 겹치는 sliding window를 저장하며 capacity도 window 개수입니다.
   구현은 환경별 증가하는 frame ID와 참조 계수로 겹침을 제거하고 기존 tensor를 공유합니다.
   샘플에는 `(base_frame_id, env_id)`를 추가합니다. 기존의 uniform sampling은 유지됩니다.

3. **TD-error 입력은 한 칸 이동해야 합니다.**
   저장 행은 `(o_t, a_{t-1}, r_t, done_t)`이므로
   `delta_t = r_{t+1} + gamma * (1-done_{t+1}) * V_{t+1} - V_t`입니다.
   `is_first`와 terminal 경계를 건너는 residual 및 사용할 수 없는 anchor context는
   최대 trigger 선택과 EMA 업데이트 전에 mask로 제외합니다.

4. **Hash용 encoder는 sample을 먼저 만들면 안 됩니다.**
   원안의 `rssm.sample_stoch`는 실제 TWISTER API에 없습니다. Encoder의 CNN과
   representation head로 logits를 얻고 기존 uniform mixture를 적용한 확률을 hash합니다.
   이렇게 해야 `probs` 모드가 결정적이며 sampling RNG를 소비하지 않습니다.
   공유 retrieval의 `[0,1]` 입력은 TWISTER의 `[-0.5,0.5]`로 변환합니다.

5. **추가할 상태는 context 전체의 posterior가 아니라 마지막 posterior 하나입니다.**
   각 Transformer block의 K/V cache와 padding, 현재 및 history `is_first` 마스크,
   마지막 상태의 실제 `done`도 함께 추가해야 합니다. 그렇지 않으면 batch shape가
   어긋나거나 terminal에서 시작한 imagination에 잘못된 할인 가중치가 붙습니다.
   `observe`는 action을 제자리 수정하므로 복제한 action을 전달합니다.

6. **가중치 곱셈만으로는 정규화가 맞지 않습니다.**
   할인 가중치를 적용한 시간별 loss를 `ell`, context 가중치를 `w`라 하면
   `sum_i(w_i * mean_h(ell_i,h)) / sum_i(w_i)`로 actor와 critic을 학습해야 합니다.
   entropy와 critic target regularizer도 포함합니다. 기존 `detached_weights`는
   continuation/discount 가중치이므로 retrieval 가중치로 덮어쓰지 않습니다.

7. **비활성화의 동일성은 manager 초기화와 난수 소비까지 포함합니다.**
   `retrieval_enabled=false`에서는 manager 자체를 만들지 않습니다.
   Replay metadata, hash, 추가 forward, 가중 reduction도 실행하지 않습니다.
   원본 코드와 실제 CPU 학습 두 update의 손실·파라미터·세 optimizer 상태·RNG를
   정확히 비교하는 회귀 테스트를 추가했습니다.

8. **추가 방식의 retrieval 강도는 STORM/Drama와 다릅니다.**
   원안대로 기존 `B*L`개 시작점을 모두 유지합니다. 기본값에서는 일반 시작점이 1024개,
   retrieval group은 최대 10개이므로 retrieval의 명목상 가중 비율은 최대 약 0.97%입니다.
   이는 성능 보장의 근거가 아니며, 실험에서 `max_anchors`와 실제 검색량을 함께 확인해야 합니다.
   Return percentile 통계는 기존 방식대로 결합 배치에서 계산하며 가중 quantile로 바꾸지 않습니다.

9. **재구축과 checkpoint에는 별도 처리가 필요합니다.**
   Warmup 종료와 resume 후 첫 활성화 시 hash를 재구축하고, 이후 낮은 lazy hit rate와
   cooldown으로 drift를 처리합니다. 새 projection에 맞지 않는 대기 anchor는 제거합니다.
   Frame metadata가 없는 이전 checkpoint는 window별 독립 구간으로 이관합니다.
   기존 TWISTER는 simulator 상태를 저장하지 않으므로 resume의 전체 궤적 동일성은 보장하지 않습니다.

설정과 실행 예시, 수식, legacy checkpoint의 제약, 검증 범위는
[TWISTER/RETRIEVAL.md](TWISTER/RETRIEVAL.md)를 참고하십시오.
