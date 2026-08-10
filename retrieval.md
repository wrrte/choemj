# Retrieval-Augmented Imagination in STORM (Implementation Details)

본 문서는 STORM 아키텍처에서 강화학습 에이전트의 효율적인 상상(Imagination)과 학습을 돕기 위해 도입된 **가치 기반 검색(Value-based Retrieval)** 기법의 실제 구현 기작을 코드를 바탕으로 분석하여 정리한 문서입니다. 시스템의 오버헤드를 최소화하고 검색의 질을 높이기 위해 도입된 핵심 설계들을 설명합니다.

---

## 1. Core Concept: 해시 기반 Lazy Rebuild 및 Fast Hash Bucket
과거 경험을 검색하기 위해 매번 전체 버퍼와 거리를 계산하는 것은 불가능에 가깝습니다. 이를 해결하기 위해 **LSH (Locality Sensitive Hashing)**와 **O(1) 연산을 지원하는 FastHashBucket**을 채택하였습니다.
- 환경과 상호작용하며 수집되는 최신 `latent` 상태들은 12-bit 해시 함수를 거쳐 `FastHashBucket`에 O(1) 시간 복잡도로 저장됩니다. 
- 과거의 비슷한 상황이 필요할 때, 해당 상태의 해시 버킷에서 유사한 프레임들을 즉각적으로 끌어올려 상상 배치(Imagine Batch)에 추가 컨텍스트로 활용합니다.
- `FastHashBucket`은 리스트와 딕셔너리를 함께 사용하여 무작위 샘플링, 삽입, 삭제를 모두 O(1)에 처리할 수 있도록 최적화되어 있습니다. 최대 용량을 초과하면 랜덤하게 기존 항목을 교체합니다.

## 2. 앵커 발굴 (Anchor Triggering) 철학
단순히 무작위로 과거 프레임을 검색하는 것이 아니라, 에이전트에게 **결정적인(Critical) 배움의 순간**을 앵커(Anchor)로 삼습니다.
- **가치 변화량 기반 평가:** 에이전트의 가치 네트워크를 활용하여 $V_{target} - V_{curr}$ 즉, TD-Error의 절댓값(`abs_delta_v`)을 관찰합니다. 예상치 못한 보상이나 상태 변화가 일어난 순간이 배움의 가치가 높다고 판단합니다.
- **Z-Score 스케일링:** 각 환경별로 가치 변화량 절댓값의 평균(`ema_mean`)과 분산(`ema_var`)을 지수 이동 평균(EMA)으로 추적합니다. 변화량이 Z-score 임계값(Threshold)을 넘는 순간, 해당 프레임이 앵커로 대기열(`active_anchors` 큐)에 추가됩니다.
- **가장 극적인 순간 포착:** 배치(Batch) 내의 시간축에서 임계값을 넘은 여러 프레임이 있더라도 가장 큰 Z-score를 기록한 단 하나의 인덱스(`max_idx`)만 앵커로 발굴합니다.

---

## 3. 핵심 설계 및 최적화 기작 (코드 레벨 분석)

### 3.1. 환경 상호작용과 병행되는 실시간 해싱 (Online Hashing)
- 환경 스텝이 진행될 때(`train.py`의 샘플링 파트), 현재 관측된 최신 프레임의 잠재 표현(Latent)을 해싱하여 `retrieval_manager.add_transition()`을 통해 즉시 `FastHashBucket`에 맵핑합니다.
- 이는 매 스텝 무거운 연산을 하지 않고, 오직 단일 프레임 해싱 및 딕셔너리 삽입(O(1))만 수행하므로 매우 가볍게 동작합니다.

### 3.2. 배치(Batch) 기반의 지연된 앵커 발굴 (Batch-based Triggering)
- 앵커를 찾는 작업은 환경 스텝과 동시에 하지 않고, **World Model 학습 루프(`train_world_model_step`)** 안에서 일괄 처리(`add_batch_transitions`)합니다.
- 리플레이 버퍼에서 추출된 궤적(Trajectory)들에 대해 **현재 최신 상태의 가치 네트워크**를 일괄 적용하여 평가합니다. 이는 과거의 낡은 가치 평가를 피하고 가장 정확하고 최신의 관점에서 놀라운(Surprising) 순간을 발굴하기 위함입니다.

### 3.3. Warm-up 구간 제외를 통한 노이즈 억제
- World Model은 초기 시퀀스(`imagine_context_length`, 기본 8프레임)를 받아야만 내부 Hidden State가 수렴하여 정상적인 예측을 할 수 있습니다.
- 코드에서는 `v_t_eval = v_t[:, skip_len:]` 와 같이 앞의 8프레임을 명시적으로 잘라내어(Slice), 웜업이 안 된 쓰레기 가치 예측값들이 Z-score나 앵커 발굴에 혼입되는 것을 원천 차단합니다.

### 3.4. 메모리/연산 병목 제어 및 타겟팅
- 발굴된 앵커 큐(`active_anchors`)에서 매 상상(Imagine) 스텝마다 설정된 `max_anchors` 개수만큼만 꺼내어 검색을 수행합니다.
- 선택된 앵커와 같은 해시 키를 공유하는 버킷 안에서 무작위로 `multiplier * (target - 1)` 만큼 샘플링을 진행한 후, 최신 모델로 다시 단일 프레임을 인코딩하여 원래 앵커 키와 동일한지 빠르게 재검증합니다. 
- 이후 유효한 시퀀스들만 모아 `max_contexts` 개수 제한 내에서 최종 배치로 구성합니다.

### 3.5. Hit-Rate 모니터링 및 PCA 기반 Global Rebuild
- 시간이 지나 World Model이 고도화되면 과거의 Latent로 만들어둔 해시 버킷은 구식이 됩니다(Representation Drift).
- `retrieve_contexts` 시점에 과거 프레임을 최신 모델로 다시 인코딩해 보며 현재 해시 키와 일치하는 비율(`lazy_hit_rate`)을 측정합니다.
- `train.py`에서 이 히트율이 `global_rebuild_threshold` 미만으로 떨어지거나 웜업 기간이 종료되는 시점에 `rebuild_all_hash_buckets`를 발동시킵니다.
- **PCA 최적화:** Global Rebuild 시 전체 유효 Latent를 모아 주성분 분석(PCA, `torch.pca_lowrank`)을 수행하여 해싱 투영축(`hash_proj`)을 데이터 분포에 맞게 최적화합니다. 이는 기존의 무작위 투영(Random Projection)보다 더 유의미한 해싱 퀄리티를 보장합니다.
- **안정적 청크 단위 처리:** 사용 가능한 GPU 메모리가 넉넉하더라도 전체 버퍼를 한 번에 슬라이싱(Vectorized Slicing)하여 읽어오면, Shared Memory나 Memmap 환경에서 값이 모두 0으로 읽히는 OS 레벨의 읽기 실패 현상(Zero-read)이 발생할 수 있습니다. 이를 방지하기 위해 데이터를 안전하게 파이썬 루프로 순차 접근하되, 풍부한 VRAM에 맞춰 청크 사이즈를 크게(`chunk_size=8192`) 잡아 GPU 병렬 연산 효율을 잃지 않도록 설계했습니다. 연산 속도 확보를 위해 PCA에 사용되는 샘플 수는 `max_pca_samples`로 제한됩니다.

---

## 4. 모니터링 체계 (WandB 로깅)
학습 코드(`train.py`)에서는 검색 시스템의 건전성을 위해 다음과 같은 핵심 지표들을 추적하고 있습니다.
* `Retrieval/triggered_anchors_step`: 모델 업데이트 스텝에서 새롭게 발굴된 앵커 수
* `Retrieval/active_anchors_queue`: 대기열(Queue)에 쌓여있는 미처리 앵커의 수
* `Retrieval/candidates_before_max`: 앵커들이 버킷에서 1차적으로 건져 올린 유효한 후보 프레임들의 개수
* `Retrieval/retrieved_contexts`: 최종적으로 필터링되어 상상 배치에 삽입된 프레임 수
* `Retrieval/lazy_rebuild_hit_rate`: Lazy Rebuild 단계에서 최신 모델로 재검증 시 원래 키와 일치하는 비율
* `Retrieval/global_rebuild_triggered`: 해시 버킷 전면 리셋(Global Rebuild)이 발동된 플래그 (1.0 = 발동)
* `Retrieval/avg_bucket_size` / `num_active_buckets` / `bucket_size_hist`: 현재 운용 중인 해시 버킷들의 크기 및 분포 현황

---

## 5. 핵심 설정 파라미터 (STORM.yaml 기준)
검색 아키텍처의 동작은 `JointTrainAgent.Retrieval` 하위 설정들로 세밀하게 조정됩니다.

* **`trigger_mode`** (`"z_score"` or `"absolute"`): 가치 변화를 평가하는 방식입니다.
* **`z_score_threshold`**: 절댓값 가치 변화(`abs_delta_v`)가 EMA 평균으로부터 몇 표준편차 벗어났을 때 앵커로 지정할지 결정합니다.
* **`ema_alpha`**: Z-score 계산을 위한 평균 및 분산 업데이트의 Momentum 계수입니다.
* **`anchor_offset`**: 발굴된 앵커 시점으로부터 실제 환경 인덱스를 시프트할 양입니다.
* **`hash_bits`**: LSH에서 Latent를 몇 비트로 해싱할지 결정합니다.
* **`use_pca`** / **`max_pca_samples`**: Global Rebuild 시 PCA 기반의 해시 투영축 최적화를 수행할지 여부와, PCA 연산에 사용할 최대 샘플 수를 지정합니다.
* **`max_anchors`**: 매 스텝 꺼내어 검색을 시도할 최대 앵커의 개수입니다.
* **`target`**: 앵커 하나당 버킷에서 구성하고자 단 최종 목표 타겟 프레임 수입니다.
* **`multiplier`**: 타겟을 찾기 위해 버킷에서 1차로 무작위 샘플링할 배수입니다.
* **`max_contexts`**: 상상 배치로 들어갈 최종 궤적들의 최대 허용 상한선입니다.
* **`global_rebuild_enable`** / **`global_rebuild_threshold`** / **`global_rebuild_cooldown`**: Hit-rate 기반의 Global Rebuild 발동 여부, 임계값, 쿨다운 스텝 설정입니다.

---

## 6. 결론
이러한 구조적 분리(Online Hashing & Batch-based Triggering)와 지연 평가(Lazy Hit-rate), 텐서 및 해시 버킷 레벨의 O(1) 최적화를 통해, **World Model 및 Actor-Critic 학습 프로세스의 기존 샘플링 비율과 동작을 전혀 방해하지 않고** 매우 적은 오버헤드로 극적인 경험들을 상상 배치에 투입할 수 있습니다.