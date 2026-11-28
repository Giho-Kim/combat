# Target Allocation

두 버전을 제공합니다.

| 버전 | 입력과 행동 |
|---|---|
| **Target Allocation** | 작업 지점의 위치·타입·남은 작업량을 알고 드론별로 작업 지점을 선택합니다. |
| **Belief Target Allocation** | 보고 정보와 센서 관측으로 belief를 갱신하며, 탐색 후보와 발견된 작업 지점 중에서 선택합니다. |

## 설치

```bash
conda activate pointmass-rl
python -m pip install -e .
python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
```

Target Allocation은 `--device auto`로 CUDA가 있으면 GPU를 사용합니다. 탐색 전용 PPO 실행 스크립트는 기본 `--device cuda`로 모델 학습·추론을 GPU에서 실행합니다. 환경 시뮬레이션과 belief 갱신은 CPU에서 실행하며, 기존 Belief COMA 학습도 CPU를 사용합니다.

## Target Allocation

Actor는 드론–후보 작업 지점 쌍마다 동료 정보를 조회하는 peer attention을 두 번 적용하고 후보별 점수를 출력합니다. Critic은 드론–작업 지점 관계를 양방향으로 전달하는 graph network이며, 공동 행동의 Q값으로 counterfactual advantage를 계산합니다. 드론·작업 지점 수가 달라도 같은 가중치를 사용합니다. 평가에서는 score resolver로 작업 지점의 남은 수용량에 맞춰 선택을 조정합니다.

`configs/target_allocation_variable.json`은 드론과 작업 지점을 각각 5–10개로 무작위 배치하고, 타입·그룹 구성도 무작위로 정합니다. 그룹 1의 작업 지점 수가 그룹 2보다 많습니다. `configs/five_agents.json`은 같은 MDP의 고정 5×5 설정입니다. 보상은 작업 지점 작업 진행에 따라 얻은 가치이며, episode 길이 100, gamma 1.0, GAE lambda 0.97입니다.

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

액터의 남은 시간은 별도 `Linear(1, 64)`로 임베딩하여 자기–후보 query 벡터에만 더합니다. 동료 K/V에는 시간을 넣지 않습니다. 나머지 query 특징 12개는 별도 MLP로 인코딩합니다. 기존 스칼라 시간 체크포인트는 평가 시 기존 구조로 자동 복원되며, 새 시간 구조는 새로 학습합니다.

## Belief Target Allocation

초기 보고는 작업 지점별로 서로 다른 위치와 알려진 타입을 제공합니다. 각 위치를 중심으로 개별 Gaussian을 두며, 실제 작업 지점은 자기 보고 위치 주변에 Gaussian 변위로 배치됩니다. 전역 탐색 belief는 이 Gaussian들의 합과 총질량 1의 기본 uniform 배경으로 구성됩니다. 배경 타입 prior는 각 타입 1/3이며 실제 미발견 작업 지점 수를 참조하지 않습니다.

매 스텝 센서 관측을 반영합니다. 미검출 위치에는 `1 − 탐지 확률`을 곱하고, 보고 Gaussian의 남은 질량·평균·공분산을 갱신합니다. 배경에서 줄어든 질량은 가까운 덜 관측된 곳으로 재분배하며 총질량과 타입 prior를 유지합니다. 배경은 연속 Gaussian 기저로 표현합니다.

새 발견은 위치·타입으로 보고 Gaussian과 배경의 가능성을 비교합니다. 보고 Gaussian에 가장 잘 맞으면 해당 성분 하나를 제거합니다. 발견된 작업 지점은 정확한 위치·타입·남은 작업량을 가진 개별 선택 후보가 됩니다. 재관측은 Gaussian을 추가로 제거하지 않으며, 제거한 보고 질량을 배경에 더하지 않습니다.

설정은 `configs/belief.json`입니다. 맵은 10×10 km, 작업 지점 배치와 탐색 belief 영역은 안쪽 8×8 km입니다. 센서는 전방 80도·500 m이며, 실제 검출과 belief 갱신에 같은 위치별 탐지 확률을 사용합니다. 이동은 Dubins 경로를 따릅니다. 현재 학습 액션의 탐색 후보는 16개 영역이고, 영역별 질량·타입 비율·위치 평균·공분산을 요약합니다. 화면의 격자는 표시와 수치적분용입니다.

```bash
bash scripts/train_belief_target_allocation.sh 10000000 runs/belief_target_allocation
python -m pointmass_rl evaluate --mode belief \
  --model runs/belief_target_allocation/best.pt \
  --episodes 10 --seed 10000 --out runs/eval_belief_target_allocation
```

Belief 학습은 seed 7, 환경 8개, rollout 64, actor·critic epochs 4, minibatch 32입니다. Replay에서 belief 변화와 발견된 작업 지점을 볼 수 있으며 실제 작업 지점 위치는 토글로 확인합니다. `--no-lawnmower`는 평가의 lawnmower 탐색을 끄고 Dubins 이동을 유지합니다.

### 탐색 전용 RL

`configs/belief_exploration.json`은 안쪽 8×8 km 전체를 겹침 없는 1×1 km 사각 polygon 64개으로 나눕니다. 후보별 belief는 매 스텝 갱신하며, 선택한 polygon은 100 m 간격 lawnmower와 Dubins 경로로 한 번 탐색한 뒤 다시 선택합니다. 최초 발견 시 작업 지점 가치(Type 1: 5, Type 2: 2, Type 3: 1)를 한 번 지급하며 드론은 발견 후에도 탐색을 계속합니다. 모든 작업 지점을 발견하거나 1,800초가 지나면 에피소드가 끝납니다.

정책은 Target Allocation의 peer-attention actor를 사용합니다. Critic은 드론·polygon 사이의 양방향 cross-attention 두 회와 시간 조건의 전체 표현 집계로 상태가치 V를 예측하며, 현재 선택한 행동은 입력받지 않습니다. 입력은 기존 polygon belief와 탐색 진행 상태를 유지하고, actor와 critic의 인코더는 별도입니다.

탐색 전용 실행 스크립트는 `--exploration-ppo`로 학습합니다. 드론마다 구역 선택부터 탐색 완료 또는 에피소드 종료까지를 한 샘플로 모읍니다. 그 구간의 팀 발견 보상을 누적하고 이후 구간까지 이어지는 실제 Monte Carlo return에서 수집 당시 V를 빼 PPO advantage를 계산합니다. 센서·물리는 계속 1초 단위이고 할인은 실제 경과 step에 적용합니다. COMA 대안 Q와 GAE bootstrap은 사용하지 않습니다. 정책을 고정한 채 에피소드를 끝까지 수집한 뒤 업데이트하므로 미완료 탐색도 시간 종료 시 샘플에 포함됩니다.

```bash
bash scripts/train_belief_exploration.sh
python -m pointmass_rl evaluate --mode belief --model runs/belief_exploration_decision_ppo_variable/best.pt --episodes 10 --seed 10000 --out runs/eval_belief_exploration
```

기본값은 환경 8개, 업데이트당 구역 선택 최소 64개, actor·critic epochs 4, minibatch 32입니다. `ROLLOUT_DECISIONS`, `N_ENVS`, `PYTHON_BIN` 등의 환경 변수로 조정합니다. `--steps`는 기존과 같은 드론별 물리 step 합계이며, 시작한 에피소드 묶음을 끝내므로 예산을 초과할 수 있습니다. `latest.pt`는 시작 시와 매 업데이트 후, `checkpoint_*.pt`는 100,000 transition 간격, `best.pt`는 평가 점수 개선 시, `exploration_ppo.pt`는 종료 시 저장합니다. 기존 COMA 체크포인트도 기존 구조로 계속 평가할 수 있습니다.

기본 설정은 에피소드마다 작업 지점 수를 5–10개에서 뽑고, 드론 수를 작업 지점 수 이상 10대 이하에서 뽑습니다. 학습과 평가는 같은 reset 규칙을 쓰고, 학습 에피소드 로그에 실제 수를 저장합니다.

스크립트의 기본 장치는 `DEVICE=cuda`입니다. CUDA가 없으면 오류로 종료하며, CPU 실행은 `DEVICE=cpu`, GPU가 있을 때 자동 선택은 `DEVICE=auto`로 지정합니다. 시작 로그에 사용 장치를 출력하고 체크포인트 metadata에도 저장합니다.

학습 중 평가는 기본 `EVAL_WORKERS=4`개의 프로세스로 실행합니다. 비교는 `none`과 `score_time_limit` 두 가지입니다. 후자는 현재 방향·속도·선회 반경과 실제 Dubins/lawnmower 경로를 사용해 구역 경계에 처음 진입할 step을 계산하고, 남은 시간 안에 진입 가능한 후보에 score resolver를 적용합니다. 전체 탐색 완료 시간은 요구하지 않으며, score 재배정도 같은 시간 필터를 따릅니다. 진행 중인 탐색은 유지하고, 진입 가능한 후보가 전혀 없으면 idle 행동이 없으므로 기존 선택지를 유지합니다. 학습 시 행동 선택에는 resolver를 적용하지 않습니다.

각 평가 방식의 에피소드를 워커에 분배하고 기존 seed·에피소드 순서로 집계합니다. 워커별 모델 추론은 학습과 같은 장치를 사용하고, 환경·belief는 각 워커의 CPU에서 계산합니다. `EVAL_WORKERS=1`은 순차 평가이며, CLI 옵션은 `--eval-workers`입니다. `best.pt`는 `score_time_limit` 평가 점수 기준으로 선택합니다. 탐색 모델의 별도 `evaluate` 명령도 `none`과 `score_time_limit`만 비교합니다.

런 스크립트는 매번 두 평가 방식에 전체 시드를 사용합니다. 필요하면 `EVAL_SCREEN_EPISODES`로 중간 체크포인트 선별을 켤 수 있습니다.

학습 없이 구역을 배정하는 `--belief-lookahead`도 실행할 수 있습니다. 현재 belief에서 후보를 추리고, 실제 이동 경로와 센서 모델로 두 구역을 연속 탐색했을 때의 기대 최초 발견 가치를 계산합니다. 첫 구역에서 이미 발견할 것으로 예상되는 가치는 둘째 구역에서 중복 집계하지 않습니다. 점수가 가장 높은 계획의 첫 구역을 배정하고, 다음 선택 때 실제 관측으로 갱신된 belief를 사용해 다시 계획합니다.

```bash
python -m pointmass_rl evaluate --mode belief --config configs/belief_exploration.json \
  --belief-lookahead --episodes 10 --seed 10000 --out runs/eval_belief_lookahead
```

같은 명령에 `--model <checkpoint>`를 추가하면 두 PPO 평가 방식도 같은 seed에서 실행합니다. Lookahead는 연속 센서 관측을 간격을 두고 근사하고, 미래 관측별 분기 대신 기대 미발견 질량을 다음 선택에 전달합니다.

### Polygon 탐색 데모

데모는 같은 belief에서 사다리꼴 polygon 후보를 생성하고, 배정된 polygon을 100 m 간격 lawnmower와 Dubins 경로로 탐색합니다. 탐색 중 polygon은 유지하고 나머지 후보를 갱신합니다. 발견한 작업 지점을 드론에 배정하고, 작업을 마친 드론의 활동을 종료합니다.

```bash
python scripts/compare_belief_heuristics.py
```

동일한 10개 시나리오에서 H1은 belief 기반 polygon 탐색 후 즉시 작업 지점을 처리합니다. H2는 두 대가 전반부 900초 동안 영역을 나눠 고르게 탐색하고 이후 타입 우선 처리에 합류합니다. 결과·영상은 `runs/belief_polygon_heuristic_comparison/index.html`, `results.csv`, `comparison.pdf`에 저장합니다. `--render-only`는 저장된 궤적으로 영상만 다시 생성합니다.
