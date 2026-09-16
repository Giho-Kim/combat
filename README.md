# Strike 우선순위 다중 드론 RL

여러 고정익 드론이 어떤 표적을 먼저 타격할지 학습하는 2D point-mass 환경입니다. 임무 선택은 제거했으며 모든 드론은 항상 Strike를 수행합니다. 정책은 매초 드론별 `target_id`만 선택하고, 환경이 최신 추정 위치를 향한 160 km/h 접근과 교전 판정을 처리합니다.

## 설치 및 실행

Python 3.10–3.12를 권장합니다.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[rl]"
python -m unittest discover -s tests -v
python -m pointmass_rl train --config configs/default.json --steps 200000 --seed 7 --eval-interval 10000 --eval-episodes 5 --out runs/train
python -m pointmass_rl evaluate --config configs/default.json --model runs/train/strike_mappo.pt --episodes 30 --out runs/eval
```

기본 장기 학습(1,000만 transition)과 평가는 각각 다음 스크립트로 실행할 수 있습니다.

```bash
./scripts/train.sh
./scripts/evaluate.sh runs/train_10m/best.pt
```

## Strike MDP

행동은 하나뿐입니다. 접근 중에는 매 step 표적을 다시 선택할 수 있습니다. 실제 타격 참여자로 확정된 뒤에는 10스텝 타격 행동이 끝날 때까지 고정되며, 타격을 완료한 드론은 즉시 환경에서 제거됩니다. 에피소드는 성공·전멸·전 표적 파괴와 무관하게 고정 horizon(기본 200 step)까지 진행합니다.

```python
{"target": int[n_agents]}
```

각 드론의 관측에는 정규화된 남은 시간·현재 선택 표적·자신이 참여 중인 타격 진행도, 모든 생존 표적의 정확한 현재 상대 위치·타입·life가 포함됩니다. 센서 범위·FoV·관측 오차·기억은 적용하지 않습니다. 제거된 드론 슬롯은 0으로 패딩되며 이후 행동, 관측, 보상과 PPO loss에서 제외됩니다.

`B`는 편제 1의 초기 타입·life 가치 합이고, `D`는 편제 1·2 전체에서 지금까지 감소시킨 타입별 life 가치의 누적합입니다. 타입 1 life는 2.5, 타입 2는 2, 타입 3은 1의 가치를 가집니다. 매 step 팀 보상은 `-penalty_time × (B-D)/B`이며 `max`로 자르지 않습니다. 따라서 `D>B`가 되면 이후 보상은 양수가 됩니다. 아군이 전멸해도 보상을 미리 당겨 지급하거나 조기 종료하지 않고 고정 horizon까지 진행합니다.

horizon 도달 시 `D<=B`이면 임무 실패로 판정해 팀 보상 `-mission_failure_penalty`를 추가합니다. 기본값은 `-100`입니다.

학습 알고리즘은 CTDE Strike MAPPO입니다. 공유 actor는 실행 시 각 드론의 local observation과 agent ID만 사용해 생존 표적의 categorical 분포를 출력합니다. centralized critic은 joint observation과 타격 진행도로 공통 팀 value를 추정합니다. 소진 전 선택에는 에피소드 종료까지의 팀 보상을 반영하며, 소진 이후 슬롯은 PPO loss에서 제외합니다.

학습 중에는 `--eval-interval` agent transition마다 `--eval-seed`부터 시작하는 동일한 고정 평가 시나리오에서 random, heuristic, deterministic MAPPO를 모두 실행합니다. 세 정책 각각의 평균 누적 환경 return, 성공률, 점수 비율, 섬멸률, score AUC, 평균 종료 step이 터미널에 출력되고 `training_evaluations.csv`에는 정책별 행으로 누적됩니다. `--eval-interval 0`으로 중간 평가를 끌 수 있습니다.

중간 평가는 각 지표의 중앙 50% interquartile mean(IQM)으로 로깅합니다. MAPPO의 IQM team return이 갱신될 때마다 `best.pt`를 저장하며, 학습 종료 시점 모델은 별도로 `strike_mappo.pt`에 저장됩니다.

`latest.pt`는 최고 성능 갱신 여부와 관계없이 매 평가 시점과 학습 종료 시 저장합니다. `--eval-interval 0`이면 종료 시에만 저장합니다. 모델 가중치 체크포인트이며 optimizer를 포함한 학습 재개 파일은 아닙니다.

Critic은 joint observation에 표적별 `strike_progress / strike_steps_per_life`를 추가로 받습니다. 기본 관측은 드론당 23차원, ID를 포함한 actor 입력은 29차원입니다. critic은 143차원 팀 상태와 6차원 0 padding을 받습니다. 체크포인트 구조는 `strike_mappo_v8`이며 이전 보상 규칙으로 학습한 모델은 재학습해야 합니다.

아군은 6대이며 모델 선택 기준은 team return입니다. 성공 여부는 전체 섬멸이 아니라 모든 편제의 누적 섬멸값 `D`가 편제 1 초기 구성으로 정한 `B`를 초과했는지로 판정합니다.

## 시나리오

- 기본 맵 `100 × 100`은 `10 km × 10 km`, 한 step은 1초이고 제한시간은 200 step입니다.
- 아군과 표적 편대 중심은 기본 35–40 좌표 단위(3.5–4 km) 거리에서 시작합니다.
- 기본 시나리오는 매 에피소드 아군 드론 6대와 표적 5대로 수를 고정합니다. 위치와 나머지 상태는 seed에 따라 바뀝니다.
- 표적은 두 편대로 균등 분할됩니다. 편제 2는 아군에서 편제 1로 향하는 경로의 55% 지점에서 측면으로 800 m 빗겨 배치됩니다. 각 편대 반경은 650 m이고 표적 간 최소 거리는 300 m입니다.
- 표적 state-space의 앞 3개 슬롯은 항상 편제 1, 뒤 2개 슬롯은 항상 편제 2입니다. 기본 5개 표적의 타입 1·2·3 구성은 2·2·1대로 무작위 순서로 섞입니다.
- 타입 1은 life 2, 점수 5이며 1대부터 타격을 시작할 수 있습니다. 10스텝 후 성공 시 1대는 life 1 감소·보상 2.5점, 2대는 life 2 감소·보상 5점을 줍니다(남은 life 이내). 참여자는 시작 시 고정되므로 이후 도착한 기체는 다음 타격에 참여할 수 있습니다.
- 타입 2는 life 1, 점수 2이고 타입 3은 life 1, 점수 1입니다.
- 실제 참여 인원은 남은 life도 초과하지 않습니다. 타입 1의 life가 1이면 다음 타격에는 1대만 참여·소모됩니다.
- 모든 표적은 타격이 시작되면 10스텝 동안 중단 없이 타격하며, 완료된 타격의 성공 확률은 기본 100%입니다. 타입 1은 타격당 최대 2대, 타입 2·3은 최대 1대만 참여하며 판정 직후 참여 기체만 소진됩니다. 같은 표적에 몰린 잉여 기체는 잠기지 않고 다음 step에 다른 표적을 선택할 수 있습니다.
- 매 step 팀 시간 페널티 계수는 `0.1`입니다.
- 모든 활성 드론은 모든 생존 표적의 현재 상태를 정확히 공유받습니다.

평가 결과의 `decisions.csv`에는 매 step 각 드론이 고른 `target_id`와 해당 표적의 점수·life·추정 거리가 기록됩니다. `score_auc`는 에피소드 동안 누적 점수를 기준 점수로 정규화해 평균한 값으로, 높은 가치의 표적을 일찍 제거할수록 커집니다. `heuristic`은 타입 2·3 표적에 한 대씩 분산하고 타입 1에는 두 대씩 조를 구성한 뒤 점수 효율과 도달 시간을 기준으로 조를 배정합니다. `replay.html`에서는 선택 변화와 파괴 순서를 확인할 수 있으며 `random`, `heuristic`, 학습된 `mappo`를 같은 seed에서 비교합니다.

구현 중심 파일은 [env.py](pointmass_rl/env.py), [strike_ppo.py](pointmass_rl/strike_ppo.py), [policies.py](pointmass_rl/policies.py), [default.json](configs/default.json)입니다.
