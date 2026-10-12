# Target Allocation

두 버전을 제공합니다.

| 버전 | 입력과 행동 |
|---|---|
| **Target Allocation** | 표적 위치·타입·life를 알고 드론별로 표적을 선택합니다. |
| **Belief Target Allocation** | 보고 정보와 센서 관측으로 belief를 갱신하며, 탐색 후보와 발견된 표적 중에서 선택합니다. |

## 설치

```bash
conda activate pointmass-rl
python -m pip install -e .
python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
```

Target Allocation은 `--device auto`로 CUDA가 있으면 GPU를 사용합니다. 환경 시뮬레이션은 CPU에서 실행하며, Belief Target Allocation은 CPU를 사용합니다.

## Target Allocation

Actor는 드론–후보 표적 쌍마다 동료 정보를 조회하는 peer attention을 두 번 적용하고 후보별 점수를 출력합니다. Critic은 드론–표적 관계를 양방향으로 전달하는 graph network이며, 공동 행동의 Q값으로 counterfactual advantage를 계산합니다. 드론·표적 수가 달라도 같은 가중치를 사용합니다. 평가에서는 score resolver로 표적의 남은 수용량에 맞춰 선택을 조정합니다.

`configs/target_allocation_variable.json`은 드론과 표적을 각각 5–10개로 무작위 배치하고, 타입·편제 구성도 무작위로 정합니다. 편제 1의 표적 수가 편제 2보다 많습니다. `configs/five_agents.json`은 같은 MDP의 고정 5×5 설정입니다. 보상은 제거한 표적 가치이며, episode 길이 100, gamma 1.0, GAE lambda 0.97입니다.

```bash
bash scripts/train_target_allocation.sh basic 10000000 runs/target_allocation_basic
```

기본 학습 설정은 seed 7, 환경 32개, rollout 200, actor epochs 10, critic epochs 40, 각 minibatch 256입니다. 평가는 100,000 transition마다 같은 seed 10000의 30개 episode로 실행합니다.

모델 기반 advantage를 사용하는 학습:

```bash
bash scripts/train_target_allocation.sh model_based 10000000 runs/target_allocation_model_based
```

고정 5×5 학습은 앞 명령에 `CONFIG=configs/five_agents.json`을 붙입니다. 기존 Target Allocation 체크포인트를 그대로 불러올 수 있으며, `--initial-checkpoint`는 가중치부터 새 학습을 시작합니다.

```bash
python -m pointmass_rl evaluate --mode known \
  --config configs/target_allocation_variable.json --model runs/target_allocation_basic/best.pt \
  --episodes 30 --seed 10000 --out runs/eval_target_allocation
```

평가 결과는 `episodes.csv`, `summary.json`, `comparison.png`, `replay.html`에 저장합니다. 학습 중 평가는 `training_evaluations.csv`에 저장합니다.

## Belief Target Allocation

초기 보고 표적마다 위치 Gaussian과 알려진 타입을 둡니다. 전역 탐색 belief는 이 Gaussian들의 합과 총질량 1의 기본 uniform 배경으로 구성됩니다. 배경 타입 prior는 각 타입 1/3이며 실제 미발견 표적 수를 참조하지 않습니다.

매 스텝 센서 관측을 반영합니다. 미검출 위치에는 `1 − 탐지 확률`을 곱하고, 보고 Gaussian의 남은 질량·평균·공분산을 갱신합니다. 배경에서 줄어든 질량은 가까운 덜 관측된 곳으로 재분배하며 총질량과 타입 prior를 유지합니다. 배경은 연속 Gaussian 기저로 표현합니다.

새 발견은 위치·타입으로 보고 Gaussian과 배경의 가능성을 비교합니다. 보고 Gaussian에 가장 잘 맞으면 해당 성분 하나를 제거합니다. 발견된 표적은 정확한 위치·타입·life를 가진 개별 선택 후보가 됩니다. 재관측은 Gaussian을 추가로 제거하지 않으며, 제거한 보고 질량을 배경에 더하지 않습니다.

설정은 `configs/belief.json`입니다. 맵은 10×10 km, 표적 배치와 탐색 belief 영역은 안쪽 8×8 km입니다. 센서는 전방 80도·500 m이며, 실제 검출과 belief 갱신에 같은 위치별 탐지 확률을 사용합니다. 이동은 Dubins 경로를 따릅니다. 현재 학습 액션의 탐색 후보는 16개 영역이고, 영역별 질량·타입 비율·위치 평균·공분산을 요약합니다. 화면의 격자는 표시와 수치적분용입니다.

```bash
bash scripts/train_belief_target_allocation.sh 10000000 runs/belief_target_allocation
python -m pointmass_rl evaluate --mode belief \
  --model runs/belief_target_allocation/best.pt \
  --episodes 10 --seed 10000 --out runs/eval_belief_target_allocation
```

Belief 학습은 seed 7, 환경 8개, rollout 64, actor·critic epochs 4, minibatch 32입니다. Replay에서 belief 변화와 발견된 표적을 볼 수 있으며 실제 표적 위치는 토글로 확인합니다. `--no-lawnmower`는 평가의 lawnmower 탐색을 끄고 Dubins 이동을 유지합니다.

### Polygon 탐색 데모

데모는 같은 belief에서 사다리꼴 polygon 후보를 생성하고, 배정된 polygon을 100 m 간격 lawnmower와 Dubins 경로로 탐색합니다. 탐색 중 polygon은 유지하고 나머지 후보를 갱신합니다. 발견 후 표적 선택·처리·드론 consume까지 포함합니다.

```bash
python scripts/compare_belief_heuristics.py
```

동일한 10개 시나리오에서 H1은 belief 기반 polygon 탐색 후 즉시 표적을 처리합니다. H2는 두 대가 전반부 900초 동안 영역을 나눠 고르게 탐색하고 이후 타입 우선 처리에 합류합니다. 결과·영상은 `runs/belief_polygon_heuristic_comparison/index.html`, `results.csv`, `comparison.pdf`에 저장합니다. `--render-only`는 저장된 궤적으로 영상만 다시 생성합니다.
