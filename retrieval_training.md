# STORM / Drama retrieval 학습

두 프로젝트 모두 설정 파일의 `JointTrainAgent.Retrieval`을 사용합니다.
STORM은 `STORM/config_files/STORM.yaml`, Drama는
`Drama/config_files/configure.yaml`에서 설정합니다. 기본값은 기존처럼 `False`입니다.

```yaml
JointTrainAgent:
  Retrieval:
    enable: Both
    warmup_steps: 50000
```

| enable | 동작 |
| --- | --- |
| `False` | retrieval 없이 한 번 학습 (`_X`) |
| `True` | warmup 동안 통계/해시를 수집하고, warmup 이후 retrieval context로 학습 (`_O`) |
| `Both` | warmup을 한 번 학습 (`_Both`)하고, `_X` (`False`) 완료 후 `_O` (`True`)를 순서대로 실행 |

문자열 `True`, `False`, `Both`도 대소문자 구분 없이 처리합니다. 다른 값은 오류로
처리합니다. `warmup_steps: 0`이면 첫 환경 step 전에 분기하며,
`warmup_steps: -1`이면 `min_warmup_steps`, `dynamic_warmup_target_steps`,
`max_warmup_steps`와 최근 에피소드 보상의 기존 STORM 동적 조건을 사용합니다.
고정 warmup은 `SampleMaxSteps`보다 작아야 합니다. step은 frame skip/action repeat
이후의 환경 transition 수이고, Drama는 `NumEnvs: 1`을 지원합니다.

## 실행

Drama 디렉터리에서 기존 실행 인자에 다음 설정을 추가합니다.

```bash
python train.py --JointTrainAgent.Retrieval.enable Both --JointTrainAgent.Retrieval.warmup_steps 50000
```

STORM 디렉터리에서는 기존 필수 인자 뒤에 YACS 설정 override를 추가합니다.

```bash
python train.py -n Pong -seed 3710 -config_path config_files/STORM.yaml -env_name ALE/Pong-v5 -trajectory_path Pong.pkl JointTrainAgent.Retrieval.enable Both JointTrainAgent.Retrieval.warmup_steps 50000
```

각 프로젝트에서 사용하던 Python/conda 환경과 GPU 선택 설정을 그대로 사용합니다.
`Both`의 두 자식 프로세스는 동일한 `CUDA_VISIBLE_DEVICES`와 Drama의
`BasicSettings.Device`를 상속합니다. 큐 실행기는 부모 프로세스의 종료를 기다리면
두 실험의 완료까지 기다리게 됩니다.

## 분기 상태와 결과

`warmup_steps: 50000`이면 step `0..49999`는 한 번 수행하고, step `50000`부터
먼저 `False` 실험을 끝까지 실행합니다. 최종 모델과 로그를 저장하고 프로세스가
정상 종료하면 `True` 실험을 시작합니다. `True`도 원래의 warmup 체크포인트에서
step `50000`부터 시작합니다. `False`가 학습한 뒤의 상태를 넘기지 않습니다.
각 실험은 모델·optimizer·AMP scaler·EMA·retrieval 해시/통계·
replay buffer·난수 상태를 복원합니다. Drama는 학습률/warmup scheduler,
관측 정규화 통계와 replay 샘플링 횟수도 복원합니다. 이후 두 모델과 replay는
각각 갱신됩니다.

분기 시 환경 자체의 내부 상태는 복제하지 않습니다. 두 실험은 같은 seed로
새 에피소드를 시작하며, 진행 중이던 에피소드/context는 초기화됩니다. 분기 전
replay의 마지막 transition에 에피소드 경계를 표시하여 초기화 전후 context가
연결되지 않게 합니다. STORM은 terminal 표시를 사용하고, Drama는 별도의
`episode_end_buffer`를 사용해 기존 critic terminal 값을 보존합니다.
따라서 환경까지 끊김 없이 실행한 단일 실험과 완전히 같은 궤적을
재현하는 방식은 아닙니다.

공통 warmup 로그는 `_Both`에 남고, 분기 로그는 `_O`와 `_X`에 따로 남습니다.
공통 구간을 각 분기 로그에 중복 복사하지 않습니다. 각 분기의
`shared_warmup.json`에서 공통 체크포인트/로그 위치와 분기 step을 확인할 수 있습니다.

- STORM 체크포인트: `ckpt/<이름>_Both/shared_warmup_<step>/`
- Drama 체크포인트: `saved_models/<이름>_Both/<환경>/<run-id>/shared_warmup/`
- 체크포인트의 `branches.json`: 실제 두 실행 명령과 실행 디렉터리
- 체크포인트의 `branch_results.json`: 실행 순서, 현재 실행 중인 실험과 완료한 실험의 종료 코드

부모 학습 프로세스는 GPU를 사용하지 않는 감독 프로세스로 교체됩니다. 이후에는
한 번에 하나의 학습 프로세스만 실행하므로 두 모델/replay가 GPU 메모리를 나눠
쓰지 않습니다. 공통 warmup 체크포인트는 자동 삭제하거나 덮어쓰지 않습니다.

`Both`의 각 실험은 주기적 저장 간격과 별개로 마지막 모델을 저장합니다.
Drama에서 `SaveModels: False`여도 `Both`로 실행한 두 실험의 최종 결과는 저장합니다.

- STORM 최종 모델: `ckpt/<이름>_X/world_model_final.pth`, `agent_final.pth` (True는 `_O`)
- Drama 최종 모델: 각 `_X`/`_O` run의 `ckpt/world_model.pth`, `agent.pth`
- 각 `ckpt/training_complete.json`: 최종 모델 파일명과 마지막 학습 다음 step

`False`의 성공 상태는 `True` 실행 전에 `branch_results.json`에 저장됩니다.
따라서 `True`가 오류나 종료 신호로 중단되어도 `_X`의 최종 모델·로그와 성공
기록은 보존됩니다. `False`가 실패하면 `True`는 시작하지 않고 실패 코드를 반환합니다.
중단된 `True`를 공통 warmup 상태에서 다시 실행하려면 `branches.json`의
`commands.retrieval_on` 명령을 기록된 `cwd`에서 실행하면 됩니다.

## Drama에 적용되는 retrieval 입력

세계 모델을 학습한 시퀀스의 value/TD 통계로 anchor를 만들고, `retrieval.py`에서
같은 해시 버킷의 context를 찾습니다. 검색된 관측/action과 동일한 replay 위치의
reward/termination을 상상 학습에 연결합니다. `batch_size_reduction`
(`retrieved`, `anchors`, `half`)과 `anchor_weight`를 적용하며 AC/PPO 손실에도
context 가중치를 반영합니다. 검색 배치가 변하면 imagination 버퍼도 함께 조정합니다.

## 검증

PyTorch와 NumPy/einops가 설치된 프로젝트 환경에서 저장소 루트에서 실행합니다.

```bash
python -m unittest discover -s tests -v
```

CPU 테스트는 retrieval 인덱스/가중치, replay 순환/에피소드 경계,
warmup 분기점, 체크포인트 복원, 실제 자식 프로세스의 순차 실행과
`True` 실패/중단 시 `False` 결과 보존을
검증합니다. GPU 처리량 및 실제 게임 학습 성능은 GPU 환경에서 별도 측정해야 합니다.
