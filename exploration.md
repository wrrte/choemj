아래는 그대로 Markdown(.md) 파일로 저장 가능한 형태로 정리한 버전이다.

### Return-Bottleneck Exploration (RBE)

Progress-Conditioned Coherent Exploration for Model-Based RL

### 1. 문제 인식

Atari 100k 환경의 일부 게임(Frostbite, Seaquest 등)에서는 성능 분포가 매우 큰 분산을 보인다.

예를 들어 Frostbite에서 다음과 같은 결과가 관찰된다.

`267, 402, 1649, 2031, 2231`

이 분포는 단순한 연속적 성능 향상보다는 이봉형(bimodal) 구조에 가깝다.

* 실패 모드: 약 250~400점

* 성공 모드: 약 1600~2200점

* 중간 구간은 상대적으로 희소

이는 정책이 전체 에피소드에서 탐험이 부족한 것이 아니라, 특정 진행도(progress) 구간에서만 새로운 행동 패턴을 발견하지 못한다는 가능성을 시사한다.

### 2. 핵심 가설

정책은 이미 잘 수행하는 prefix(초반 구간)에서는 안정적으로 exploit해야 하고, 학습이 병목에 걸리는 return 구간에서만 일관된(coherent) exploration을 수행해야 한다.

즉, 문제는 다음과 같이 해석된다.

* 0~1800점: 이미 안정적으로 도달 가능

* 1800~2600점: 새로운 행동 패턴 필요

* 2600점 이후: 다시 안정적으로 고점으로 이어짐

### 3. 기본 아이디어

Replay buffer 또는 world model의 imagined trajectories에서 return progression의 분포를 수집한다.

각 transition에 대해 현재 에피소드의 누적 점수 R_t를 기록한다.

예시:

`episode A: 0 → 50 → 120 → 250 episode B: 0 → 100 → 400 → 900 → 1800 episode C: 0 → 80 → 350 → 700 → 2100`

이 정보로 return 히스토그램을 구축한다.

### 4. Return 히스토그램

최근 N개의 에피소드(또는 replay 전체)에 대해 다음을 계산한다.

H(r)=count of transitions with return rH(r) = \text{count of transitions with return } rH(r)=count of transitions with return r

예시:

`0-500 ████ 500-1000 ███ 1000-1500 ███ 1500-2000 █ 2000-2500 ▏ 2500-3000 ██ 3000+ ███`

밀도가 급격히 낮아지는 구간이 return bottleneck이다.

### 5. 병목 구간 탐지

이동 평균 밀도 대비 상대적으로 희소한 구간을 선택한다.

예시 규칙:

density(r)<α⋅moving_average_density(r)\text{density}(r) < \alpha \cdot \text{moving\_average\_density}(r)density(r)<α⋅moving_average_density(r)

탐지 결과:

`B = [1800, 2600]`

### 6. 진행도 기반 모드 전환

에피소드 중 현재 누적 점수 R_t에 따라 정책 모드를 결정한다.

`if R_t < 1800: mode = EXPLOIT elif 1800 <= R_t < 2600: mode = EXPLORE else: mode = EXPLOIT`

### 7. 왜 exploit → explore → exploit 인가?

### 0~1800: Exploit

* 이미 학습된 안정적인 행동

* 불필요한 노이즈는 성능을 파괴할 가능성이 큼

### 1800~2600: Explore

* 새로운 trajectory가 필요한 구간

* 정책 다양성을 증가시켜야 함

### 2600+: Exploit

* 병목을 넘은 이후

* 고점 trajectory를 안정적으로 재현하는 것이 중요

### 8. Exploration 방식

핵심은 step-wise 랜덤 행동이 아니라 trajectory-level coherent exploration이다.

### 8.1 에피소드 노이즈

에피소드 시작 시 정책 파라미터를 perturb한다.

θ′=θ+ϵ,ϵ∼N(0,σ2)\theta' = \theta + \epsilon, \qquad \epsilon \sim \mathcal{N}(0,\sigma^2)θ′=θ+ϵ,ϵ∼N(0,σ2)

해당 에피소드 동안 θ'를 고정 사용한다.

### 8.2 병목 진입 시 파라미터 노이즈

탐험 구간에 들어갈 때만 noisy actor를 생성한다.

`if enters_bottleneck(R_t): actor_explore = perturb(actor)`

이후 병목 구간에서는 actor_explore를 사용한다.

### 8.3 여러 actor head (선택 사항)

* exploit head

* explore head

병목 구간에서만 explore head를 사용한다.

### 9. 사용하지 않는 방식

다음과 같은 매 스텝 랜덤 행동은 사용하지 않는다.

`a_t = π(s_t) + random_noise`

이유:

* 장기 trajectory를 깨뜨림

* 병목 돌파에 필요한 일관성을 파괴함

### 10. Replay와 Imagination

### Replay 기반

실제 경험 분포를 사용.

장점:

* 안정적

* 실제 병목을 직접 반영

### Imagination 기반

World model rollout의 return 분포를 사용.

장점:

* 더 빠른 병목 추정

* 미래 병목 예측 가능

### 11. 온라인 알고리즘

`for each training update: update_histogram(replay, imagined) B = detect_low_density_intervals(H) for each environment step: R_t = current_episode_return if R_t in B: action = explore_policy(s_t) else: action = exploit_policy(s_t)`

### 12. 기존 연구와의 관계

### Go-Explore

* 공통점: 잘 아는 prefix는 보호하고 이후만 탐험

* 차이점: 상태 기반 vs return 기반

### InfoBot

* 공통점: 분기점(decision states) 탐지

* 차이점: 정보 병목 vs return 병목

### Parameter Space Noise

* 공통점: 일관된 trajectory-level exploration

* 차이점: 언제 탐험을 켤지 다름

### Bootstrapped DQN

* 공통점: 정책 다양성 증가

* 차이점: 병목 조건 없음

### Scheduled Intrinsic Drive (SID)

* 공통점: exploit/explore 모드 전환

* 차이점: 고정 스케줄 vs 데이터 기반

### 13. 이 방법이 적합한 게임

적합:

* Frostbite

* Seaquest

* Montezuma’s Revenge

* 기타 first-success bottleneck이 강한 게임

덜 적합:

* Pong

* Breakout

* Asterix

* 점수가 연속적으로 증가하는 게임

### 14. 예상 장점

* 저점(low tail) 개선

* 불필요한 탐험 감소

* 성공 trajectory 발견 확률 증가

* 고점 유지 가능성

* 게임별 임계 시점 자동 적응

### 15. 예상 실패 모드

### Return aliasing

같은 return이라도 상태가 다를 수 있음.

### Too-late exploration

병목 이전의 준비 행동이 필요한 경우 탐험이 늦을 수 있음.

### Noisy histogram

표본이 적으면 가짜 병목이 생길 수 있음.

### Progressive games

점진적 게임에서는 불필요한 모드 전환이 성능을 떨어뜨릴 수 있음.

### 16. 개선 방향

### 병목 앞당기기

`[R_low - Δ, R_high]`

### 상태 정보 보강

Return + latent novelty 사용.

bottleneck(s)=low_density(Rt)∧high_novelty(zt)\text{bottleneck}(s) = \text{low\_density}(R_t) \land \text{high\_novelty}(z_t)bottleneck(s)=low_density(Rt)∧high_novelty(zt)

### Imagination frontier

World model에서 자주 도달하는 return과 거의 도달하지 못하는 return 사이의 경계를 탐험 대상으로 사용.

### 17. 실험 프로토콜

Frostbite 기준.

### 비교군

`A. Baseline B. Always parameter noise C. Return-Bottleneck Exploration (제안 방법)`

### 평가 지표

* 15k 시점 성공률 (예: score ≥ 800)

* 최종 평균 점수

* 하위 25% 점수

* return 분포 변화

* 병목 구간 통과 확률

### 18. 핵심 차별점

기존 탐험 기법들은 주로 다음 중 하나에 의존한다.

* 시간(step) 기반

* 상태 novelty 기반

* uncertainty 기반

* 고정 entropy 스케줄

본 방법은 return-progress 분포의 저밀도 구간을 온라인으로 탐지하고, 그 구간에서만 trajectory-level coherent exploration을 수행한다는 점이 핵심 차별점이다.

### 19. 한 문장 요약

Return-Bottleneck Exploration은 replay/imagination의 return 분포에서 **가장 늦은?** 저밀도 병목 구간을 온라인으로 탐지하고, 그 구간에서만 에피소드/파라미터 노이즈를 사용해 일관된 trajectory-level exploration을 수행하며, 그 외 구간에서는 기존 정책을 안정적으로 exploit하는 방법이다.
