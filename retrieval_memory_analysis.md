현재 코드와 설정을 기준으로 한 retrieval 추가 메모리 분석 (2026-09-19)

두 구현 모두 가장 큰 추가 할당은 전체 해시 재구축에서 발생한다. 버퍼에 100,000개 transition이 있을 때 재구축의 명시적 텐서 저장공간 피크는 약 **1.580 GiB**다. 50,000개에서는 약 **0.813 GiB**다. 이는 해당 단계에서 추가로 살아 있는 텐서 크기이며, 실행 전체의 최대 VRAM이 반드시 그만큼 증가한다는 의미는 아니다.

GPU 드라이버를 사용할 수 없어 CUDA 실행의 피크는 측정하지 못했다. 대신 소스에서 dtype·shape·참조 수명을 추적하고, 재구축의 텐서 연산을 CPU에서 재현해 PyTorch MemTracker로 저장공간을 검증했다. 해시 자료구조는 실제 RetrievalContextManager._insert_into_bucket()으로 구성해 측정했다. CUDA allocator, cuDNN/cuBLAS 작업공간, 컴파일 및 CUDA Graph 메모리는 아래 텐서 산식에 포함하지 않는다.

단위는 1 MiB = 2^20 bytes, 1 GiB = 2^30 bytes다. 값은 학습 프로세스 하나 기준이다.

**계산에 적용한 설정**

근거: [STORM 설정](STORM/config_files/STORM.yaml), [Drama 설정](Drama/config_files/configure.yaml). STORM은 실행 시 -config_path가 필수이므로 이 문서는 저장소의 STORM.yaml을 사용하는 경우다. Drama/job_queue_pro6k.txt는 환경과 seed만 바꾸므로 현재 configure.yaml의 아래 값이 적용된다.

| 항목 | STORM | Drama |
|---|---:|---:|
| replay 최대 transition 수 | 100,000 | 100,000 |
| NumEnvs | 1 | 1 |
| 이미지 / 저장 dtype | 64 × 64 × 3 / uint8 | 동일 |
| ReplayBufferOnGPU | True | True |
| 해싱 latent 차원 D | 32 × 32 = 1,024 | 동일 |
| hash_bits h | 10 | 10 |
| PCA 최대 표본 | 100,000 | 100,000 |
| 재구축 chunk_size | 1,024 | 1,024 |
| world model 학습 B × T | 16 × 64 | 16 × 128 |
| imagination 배치 B / context T | 1,024 / 8 | 동일 |
| imagination rollout 길이 | 16 | 16 |
| max_anchors / multiplier / target | 16 / 16 / 16 | 동일 |
| max_contexts | 256 | 256 |
| batch_size_reduction | retrieved | retrieved |
| warmup_steps | 50,000 | 50,000 |
| enable | Both | Both |

**상시 GPU 저장공간과 replay**

RetrievalContextManager는 신경망이나 optimizer를 추가하지 않는다. retrieval.py:103–115, 563–564의 상시 텐서는 다음과 같다.

| 텐서 | 크기 | bytes |
|---|---:|---:|
| hash_proj | FP32 [1024, 10] | 40,960 |
| hash_mean, PCA 이후 | FP32 [1, 1024] | 4,096 |
| hash_bit_values | int64 [10] | 80 |
| prev_v | FP32 [1] | 4 |
| 합계 | | **45,140 = 44.082 KiB** |

두 train.py 모두 enable=False에서도 manager를 생성한다. 따라서 처음부터 False로 실행한 경우와 비교하면 위 44 KiB 전체가 순증하는 것은 아니며, PCA 이후의 hash_mean 약 4 KiB가 주된 상시 GPU 순증이다. Both에서 checkpoint를 읽은 False branch에는 hash_mean도 복원될 수 있다.

hash_memory는 이미지·latent의 사본을 보관하지 않고 CPU에서 (pointer, env_idx)를 보관한다. Drama의 retrieval_view()도 value[:, None]인 view이므로 replay를 복사하지 않는다 (Drama/replay_buffer.py:122, 182–189).

기존 replay 전체 크기는 STORM 약 1,173.019 MiB, Drama 약 1,173.878 MiB다. Drama의 episode_end 및 두 counter도 True/False 모두 할당하므로 현재 코드의 retrieval on/off 차이에 포함하지 않는다. 이 약 1.146 GiB replay는 retrieval을 켜며 새로 생기는 메모리가 아니다.

**평상시 retrieval과 학습 단계**

검색에 성공한 context 수를 R, 유효 anchor 수를 A라 하면 현재 설정에서 R ≤ min(256, 16A), A ≤ 16이다. 실제 R은 trigger 수·유효 context·해시 일치율에 따라 달라진다.

batch_size_reduction='retrieved'이므로 무작위 배치 = 1024 - R, 최종 배치 = (1024 - R) + R = 1024다 (STORM/train.py:116–158, Drama/train.py:144–171). 따라서 imagination rollout과 actor/critic 학습 배치가 커지지 않으며, 모델/optimizer 메모리도 늘지 않는다.

반환한 ret_obs는 FP32 [R, 8, 3, 64, 64]이므로:

    ret_obs bytes = R × 8 × 3 × 64 × 64 × 4
                  = R × 393,216 = R × 0.375 MiB

| R | ret_obs 저장공간 |
|---:|---:|
| 16 | 6 MiB |
| 64 | 24 MiB |
| 128 | 48 MiB |
| 256 | 96 MiB |

최종 sample_obs는 기존과 같은 384 MiB다. 그러나 torch.cat 결과와 별도로 ret_obs의 지역변수 참조가 world_model.imagine_data() 호출 중 유지되므로 이 구간에는 최대 96 MiB가 추가로 살아 있다. 이 값은 모든 retrieval 임시 메모리를 합친 엄밀한 상한은 아니다.

R=256에서 이미지 텐서 수명은 다음과 같다.

| 구간 | 동시에 살아 있는 주요 이미지 텐서 |
|---|---:|
| retrieval context를 FP32로 정규화 | uint8 stack 24 MiB + 변환 FP32 96 MiB + 나눗셈 결과 96 MiB = 약 216 MiB |
| random / retrieval context를 최종 cat | random 288 MiB + ret_obs 96 MiB + 결과 384 MiB = 768 MiB |
| 이후 world model imagination | 최종 입력 384 MiB + ret_obs 96 MiB = 480 MiB |

768 MiB 전체 또는 cat 결과 384 MiB 전체를 baseline 대비 증가로 계산하면 안 된다. baseline의 random sample 역시 FP32 변환과 나눗셈에서 임시 사본을 만들기 때문이다. 실제 전체 피크는 이 구간과 모델 forward/backward의 최대값으로 결정된다.

Lazy re-encoding은 anchor별로 순차 실행한다. 한 번의 후보 수는 multiplier × (target - 1) = 240이며 16 × 240개를 한꺼번에 encoder로 넘기지 않는다 (retrieval.py:343–385). 240장의 FP32 입력은 11.25 MiB, FP32 latent는 0.9375 MiB다. 여기에 CNN 중간 activation과 라이브러리 작업공간이 붙는다. 이 forward는 no_grad이며 다음 anchor까지 전체 학습 그래프를 쌓지 않는다.

추가 trigger 평가에서 사용하는 결합 feature는 다음과 같다.

| 항목 | STORM | Drama |
|---|---:|---:|
| [B, T, 1024 + 512] FP32 feature | 16 × 64 × 1536 × 4 = **6 MiB** | 16 × 128 × 1536 × 4 = **12 MiB** |
| 평가 | 기존 critic no_grad forward | 기존 critic no_grad + AMP forward |

근거: STORM/train.py:72–81, Drama/train.py:73–88, Drama/sub_models/world_models.py:731–732. feature 외에 critic 중간 activation·TD/z-score 텐서가 추가된다. 이 단계와 context retrieval, 재구축의 개별 최대치를 모두 합쳐 상시 사용량으로 계산하면 안 된다. STORM의 단일 프레임 hash encoding 호출은 False에서도 실행되므로 해당 encoding 자체는 현재 STORM의 on/off 순증이 아니다 (STORM/train.py:310–313).

**전체 해시 재구축: 가장 큰 추가 할당**

두 모델은 AMP를 사용해도 현재 해싱 경로의 확률 latent를 FP32로 반환한다. DistHead의 softmax/log, OneHotCategorical 경로를 확인했으며, Drama는 unimix에서 FP32 tensor도 명시한다. 따라서 latent를 단순히 BF16 2 bytes로 계산하면 과소평가한다. CUDA autocast의 softmax/log FP32 정책은 [PyTorch AMP 문서](https://docs.pytorch.org/docs/2.8/amp.html#cuda-ops-that-can-autocast-to-float32)와 로컬 PyTorch autocast 헤더에서 확인했다.

버퍼의 실제 transition 수를 N이라 하자. 현재는 N ≤ max_pca_samples이므로 pca_input은 full_latents를 공유한다. 다음 네 저장공간은 동시에 존재한다.

| 저장공간 | 코드 | N=100,000일 때 |
|---|---|---:|
| all_encoded_latents에 쌓인 청크 전체 | retrieval.py:541 | 390.625 MiB |
| torch.cat으로 만든 full_latents | retrieval.py:547 | 390.625 MiB |
| PCA centered_input | retrieval.py:557 | 390.625 MiB |
| _hash_keys의 latent_f - hash_mean 결과 | retrieval.py:154 | 390.625 MiB |
| 네 텐서 합계 | 4 × N × D × 4 bytes | **1,562.500 MiB = 1.526 GiB** |

PCA 함수가 끝나도 centered_input과 U는 rebuild_all_hash_buckets의 지역변수로 남아 있다. all_encoded_latents도 cat 이후 비워지지 않는다. 재구축에 no_grad가 있어도 이러한 명시적 사본은 남는다.

해싱 중 int64 변환과 곱셈이 겹치는 순간에는 다음도 존재한다.

- U: 4Nh bytes.
- scores: 4Nh bytes.
- bits: Nh bytes.
- int64 bits와 곱셈 출력 두 개: 16Nh bytes.
- 마지막 chunk의 obs_tensor: 4r × 3 × 64 × 64 bytes. r = ((N - 1) mod 1024) + 1.
- projection, mean, S 등의 작은 텐서.

따라서 이 경로의 주요 저장공간 피크는:

    P(N) ≈ 16ND + 25Nh + 4rHWC + 4Dh + 4D + 4h bytes
    D=1024, h=10, H=W=64, C=3

| 채워진 transition 수 N | 네 대형 텐서 | 부가 텐서를 포함한 계산 |
|---:|---:|---:|
| 50,000 | 781.250 MiB | **832.964 MiB = 0.81344 GiB** |
| 75,000 | 1,171.875 MiB | **1,201.424 MiB = 1.17327 GiB** |
| 100,000 | 1,562.500 MiB | **1,617.885 MiB = 1.57997 GiB** |

50,000 및 100,000 행은 CPU에서 같은 shape/dtype의 청크 latent를 생성하고 실제 torch.pca_lowrank 및 RetrievalContextManager._hash_keys를 실행한 MemTracker 결과와 일치했다. 이 재현은 encoder를 동일 shape의 생성 텐서로 대체하므로 실제 encoder의 activation, CUDA 작업공간을 측정한 것이 아니다. 원래 모델·replay·imagination 결과 등 호출 전에 존재하던 텐서도 제외했다. 처음부터 존재하던 manager의 수십 KiB 포함 방식에 따른 작은 차이는 위 반올림 해석에 영향을 주지 않는다.

chunk_size=1024는 이미지 encoder 입력을 제한하지만, latent 청크 전체는 계속 모으므로 위 O(ND) 피크를 제한하지 못한다. max_pca_samples도 전체 latent 수집·최종 일괄 해싱 크기를 제한하지 않는다. PCA는 q=10의 low-rank 연산이며 N×N 행렬을 만드는 구조는 아니다.

**CPU 해시 메모리**

실제 FastHashBucket과 _insert_into_bucket()으로 N개 항목을 삽입한 뒤, 공유 객체를 id로 중복 제거하여 sys.getsizeof를 재귀 합산했다. key는 실제 Tensor.tolist()처럼 큰 정수 객체가 항목별로 만들어지도록 구성했다. 두 분포는 실제 실행을 대신하는 예시이며, 전체 가능한 분포의 엄밀한 최소·최대값은 아니다.

| N / 합성 해시 분포 | STORM 환경 Python 3.10 | Drama 환경 Python 3.12 |
|---|---:|---:|
| 50,000 / 1,024 buckets 균등 | 10.369 MiB | 10.353 MiB |
| 50,000 / 1 bucket 집중 | 12.094 MiB | 12.094 MiB |
| 100,000 / 1,024 buckets 균등 | **20.678 MiB** | **20.662 MiB** |
| 100,000 / 1 bucket 집중 | **24.110 MiB** | **24.110 MiB** |

이 값에는 hash_memory, 각 bucket의 items/data_map, index_to_bucket, tuple/int 및 빈 anchor deque를 포함한다. Python interpreter·PyTorch 자체·allocator arena·RSS fragmentation은 포함하지 않는다. retrieval은 latent 전체를 CPU에 상시 저장하지 않는다.

max_bucket_size=1,000,000,000은 선할당 크기가 아니다. 빈 Python list/dict에서 실제 항목 수만큼 커지므로 10억 항목의 메모리를 예약하지 않는다.

재구축 중 all_valid_indices의 임시 tuple/list, keys list 등으로 CPU 사용량이 더 늘어난다. 최종 hash의 pointer 정수와 공유되는 부분을 제외해도 100,000개에서 임시 tuple/list는 대략 7 MiB 규모다. 정확한 RSS에는 딕셔너리 확장과 allocator 재사용도 영향을 준다.

현재 설정에서는 world model update 한 번당 anchor가 최대 16개 생성되고, 매 스텝 retrieval이 최대 16개를 제거한다. 따라서 정상 학습 중 queue가 계속 누적되지는 않는다. train 주기·epoch·max_anchors를 바꾸면 active_anchors deque에는 별도 maxlen이 없어 증가할 수 있다.

state_dict()는 자료구조를 deepcopy한다. 합성 N=100,000 상태에서 원본+snapshot을 함께 센 Python 객체 크기는 약 31.4–34.9 MiB였다. immutable tuple/int 공유가 있어 단순 2배는 아니며, deepcopy의 작업용 memo 등 일시적인 메모리는 이 수치에서 빠진다.

**Both와 전체 실행 최대값 해석**

training_branches.py:141–171은 retrieval_on을 끝낸 뒤 retrieval_off를 실행한다. 부모도 exec로 supervisor가 되므로 두 학습 모델이 동시에 상주하는 방식이 아니다. STORM.yaml 주석의 순서보다 실행 코드를 기준으로 판단해야 한다.

두 branch 모두 warmup checkpoint의 retrieval state를 읽고 manager에 복원한다. False에서도 이미 생성된 CPU hash가 남을 수 있으며, 원래 resume state도 참조되어 frozen snapshot이 남는다 (STORM/train.py:212–214, Drama/train.py:238–251). 따라서 위 CPU 표는 active manager 자체의 크기다. Both에서 실제 _O와 _X 프로세스 RSS 차이를 그대로 그 표와 동일시하면 안 된다. branch checkpoint에는 replay/model/optimizer의 CPU 저장 사본도 생기지만 이는 retrieval 검색 알고리즘의 상시 메모리와 구별해야 한다.

실행 전체 최대 VRAM은 단계별 합이 아니라 최대값이다. 예를 들어 baseline의 학습 backward가 이미 더 큰 피크를 갖는다면, 재구축에서 1.58 GiB의 텐서를 추가하더라도 실행 전체 피크 차이는 1.58 GiB보다 작을 수 있다. 반대로 allocator 예약·컴파일·CUDA 작업공간 때문에 nvidia-smi 값은 명시적 텐서 계산보다 커질 수 있다. tensor allocation과 allocator reservation의 구별은 [PyTorch CUDA 메모리 문서](https://docs.pytorch.org/docs/2.8/notes/cuda.html#cuda-memory-management)를 따른다.

CUDA 실측에서는 동일 환경·설정·버퍼 크기를 맞추고, 재구축 직전 synchronize/reset_peak_memory_stats 후 max_memory_allocated를 기록해야 한다. 함수 종료 후 memory_allocated만 비교하면 해제된 1.58 GiB 임시 피크를 놓친다. 별도로 전체 학습 단계의 max_memory_allocated/max_memory_reserved를 기록해야 on/off 실행 피크 차이를 알 수 있다.

**설정을 바꿀 경우의 수치**

아래는 현재 코드에 실제 변경을 적용하지 않고, 동일한 텐서 연산을 CPU에서 재현한 저장공간이다.

| N=100,000의 재구축 설정 | 주요 명시적 텐서 피크 |
|---|---:|
| 현재: use_pca=True, max_pca_samples=100000 | 1,617.885 MiB = **1.580 GiB** |
| use_pca=True, max_pca_samples=10000 | 1,302.715 MiB = **1.272 GiB** |
| 처음부터 use_pca=False, hash_mean=None | 832.777 MiB = **0.813 GiB** |

PCA 표본을 1/10로 줄여도 전체 latent 수집·해싱은 유지되어 이 피크는 약 19.5%만 감소한다. N보다 작은 M개를 선택하면 pca_input의 복사가 추가되어 큰 텐서 항은 12ND + 8MD bytes가 된다. PCA가 계산된 checkpoint를 불러오며 use_pca만 False로 바꾸면 기존 hash_mean이 남으므로 마지막 행의 조건과 다르다. PCA 설정 변경은 해시 분포와 실험 결과에도 영향을 준다.

batch_size_reduction을 변경하면 R=256, A=16일 때:

| 설정 | 최종 imagination batch | baseline 대비 |
|---|---:|---:|
| retrieved, 현재 | 1,024 | 동일 |
| half | 1,144 | +11.72% |
| anchors | 1,264 | +23.44% |

half/anchors에서는 입력·rollout·actor/critic activation과 캐시의 batch 비례 부분도 증가한다. 모델 parameter와 optimizer가 같은 비율로 늘어나는 것은 아니다. 현재 분석의 '최종 배치 동일' 결론은 retrieved 설정에 한정된다.

계산 재현용 스크립트: [/tmp/retrieval_memory_audit.py](/tmp/retrieval_memory_audit.py). Drama 환경에서는 전체 audit를 실행했고 STORM 환경에서는 --hash-only로 Python 컨테이너 크기를 비교했다. 원본 학습 코드는 수정하지 않았다.
