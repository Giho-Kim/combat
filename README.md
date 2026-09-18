# Strike 우선순위 다중 드론 RL

여러 고정익 드론이 어떤 표적을 먼저 타격할지 학습하는 2D point-mass 환경입니다. 임무 선택은 제거했으며 모든 드론은 항상 Strike를 수행합니다. 정책은 매초 드론별 `target_id`만 선택하고, 환경이 최신 추정 위치를 향한 160 km/h 접근과 교전 판정을 처리합니다.

## 설치 및 실행

Python 3.10–3.12를 권장합니다. Conda 환경을 사용합니다.

```bash
conda create -n pointmass-rl python=3.11 -y
conda activate pointmass-rl
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

`B`는 편제 1의 초기 타입·life 가치 합이고, `D`는 편제 1·2 전체에서 지금까지 감소시킨 타입별 life 가치의 누적합입니다. 타입 1 life는 2.5, 타입 2는 2, 타입 3은 1의 가치를 가집니다. 매 step 팀 보상은 `-penalty_time × (B-D)/B`이며 `max`로 자르지 않습니다. `D=B`이면 시간 보상이 0이고 임무 성공이며, `D>B`이면 이후 보상이 양수가 됩니다. 아군이 전멸해도 보상을 미리 당겨 지급하거나 조기 종료하지 않고 고정 horizon까지 진행합니다.

horizon 도달 시 `D<B`이면 임무 실패로 판정해 팀 보상 `-mission_failure_penalty`를 추가합니다. 기본값은 `-100`입니다.

학습 알고리즘은 CTDE Strike MAPPO입니다. 공유 actor는 각 드론의 local observation과 agent ID만 사용해 표적별 logit을 출력합니다. capacity-aware resolver가 고정된 agent ID 순서로 이미 찬 표적을 마스킹해 타입 1에는 최대 2대, 타입 2·3에는 최대 1대만 배정합니다. 남은 유효 life slot이 드론보다 많으면 life당 가치가 낮은 slot부터 제외하므로 기본 구성에서는 타입 1 life 4개와 타입 2 life 2개에 6대를 배정해 최대 점수 14를 보존합니다. PPO에는 실제 선택 때 사용한 conditional mask와 log-probability를 저장하므로 update도 같은 분포를 사용합니다. centralized critic은 joint observation과 타격 진행도로 공통 baseline을 추정합니다. 소진 전 선택에는 에피소드 종료까지의 팀 보상을 반영하며, 소진 이후 슬롯은 PPO loss에서 제외합니다.

환경 보상과 평가의 `team_return`은 위 공식을 그대로 사용합니다. PPO의 학습 보상에만 드론별 damage credit `damage_credit_scale × (해당 드론이 감소시킨 가치 / B)`를 더합니다. 기본 scale은 50이며, 실제 피해를 만든 드론만 credit을 받으므로 같은 팀 보상을 공유하던 유효 배치와 잉여 배치를 구분할 수 있습니다. `damage_credit_scale=0`이면 기존 순수 팀 보상 학습으로 돌아갑니다. 이 shaping은 성공 판정과 모델 입력, 평가 return을 바꾸지 않습니다.

학습 중에는 `--eval-interval` agent transition마다 `--eval-seed`부터 시작하는 동일한 고정 평가 시나리오에서 random, heuristic, deterministic MAPPO를 모두 실행합니다. 성공률은 전체 평가 에피소드의 raw mean으로, 나머지 지표는 중앙 50% IQM으로 출력하고 `training_evaluations.csv`에 기록합니다. `--eval-interval 0`으로 중간 평가를 끌 수 있습니다.

`best.pt`는 raw 성공률을 우선하고, 성공률이 같으면 IQM team return이 높은 모델로 갱신합니다. 학습 종료 시점 모델은 별도로 `strike_mappo.pt`에 저장됩니다.

`latest.pt`는 최고 성능 갱신 여부와 관계없이 매 평가 시점과 학습 종료 시 저장합니다. `--eval-interval 0`이면 종료 시에만 저장합니다. 모델 가중치 체크포인트이며 optimizer를 포함한 학습 재개 파일은 아닙니다.

Critic은 joint observation에 표적별 `strike_progress / strike_steps_per_life`와 reward margin을 추가로 받습니다. 기본 관측은 드론당 24차원, ID를 포함한 actor 입력은 30차원입니다. critic은 150차원 팀 상태와 6차원 0 padding을 받습니다. 체크포인트 구조는 `strike_mappo_v14`이며 이전 모델은 재학습해야 합니다.

아군은 6대입니다. 성공 여부는 전체 섬멸이 아니라 모든 편제의 누적 섬멸값 `D`가 편제 1 초기 구성으로 정한 `B` 이상인지로 판정합니다. 가능한 최대 `B=12`에 대해 6대의 최대 획득값은 `14`이므로 기본 시나리오는 모두 성공 가능합니다.

5대 ablation은 `configs/five_agents.json`으로 별도 실행할 수 있습니다. 이때 최대 획득값은 `D_max=12`이므로 `B=12`인 구성도 `D>=B`를 만족할 수 있습니다. 6대 기본 설정과 체크포인트 입력 차원이 다르므로 5대 모델은 별도로 학습해야 합니다.

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
- PPO 학습 전용 damage credit scale은 `50`입니다.
- `GAE lambda=1.0`을 사용해 먼 표적의 지연된 결과가 중간 bootstrap에서 추가 감쇠되지 않게 합니다.
- 모든 활성 드론은 모든 생존 표적의 현재 상태를 정확히 공유받습니다.

평가 결과의 `decisions.csv`에는 매 step 각 드론이 고른 `target_id`와 해당 표적의 점수·life·추정 거리가 기록됩니다. `score_auc`는 에피소드 동안 누적 점수를 기준 점수로 정규화해 평균한 값으로, 높은 가치의 표적을 일찍 제거할수록 커집니다. `heuristic`은 타입 2·3 표적에 한 대씩 분산하고 타입 1에는 두 대씩 조를 구성한 뒤 점수 효율과 도달 시간을 기준으로 조를 배정합니다. `replay.html`에서는 선택 변화와 파괴 순서를 확인할 수 있으며 `random`, `heuristic`, 학습된 `mappo`를 같은 seed에서 비교합니다. 타격 진행 중인 드론은 움직이는 표적 주위를 선회한 뒤 타격 효과와 함께 사라집니다. 선회는 리플레이 전용 시각 효과이며 환경의 위치, 타격 시간, 보상과 정책 입력에는 영향을 주지 않습니다.

구현 중심 파일은 [env.py](pointmass_rl/env.py), [strike_ppo.py](pointmass_rl/strike_ppo.py), [policies.py](pointmass_rl/policies.py), [default.json](configs/default.json)입니다.
