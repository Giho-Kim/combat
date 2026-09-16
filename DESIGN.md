# Strike-only MDP 설계

```text
local observation → masked target categorical actor
                  → target_id
                  → latest-known-position strike resolver
                  → 2-D speed-limited point mass
                  → sensing and combat transition
```

## MDP

결합 행동은 활성 드론마다 하나의 생존 표적 ID를 지정합니다. 사거리에서 타격을 시작한 드론은 life당 10스텝 동안 선택을 유지하고, 완료 판정 직후 성공 여부와 관계없이 제거됩니다. 성공, 전멸, 전 표적 파괴는 terminal state가 아니며 에피소드는 고정 horizon(기본 200 step)에서만 끝납니다. 이동 방향과 속도는 resolver가 결정하므로 정책의 자유도는 표적 우선순위와 드론 배치에만 쓰입니다.

관측 크기는 다음과 같습니다.

```text
4 self
+ 4 × n_targets
```

self record는 정규화된 남은 시간 `(H-t)/H`, 현재 선택 표적, 자신이 참여 중인 타격 진행도, 정규화된 reward margin으로 구성됩니다. 표적 record는 상대 위치 2, 정규화된 남은 가치, 선택 가능 여부로 구성됩니다. 제거된 슬롯은 0 padding이며 이후 PPO loss에서 제외됩니다. 환경의 개별 제거 신호와 팀 학습의 종료 신호는 구분합니다. 팀 보상 합을 학습하며 GAE는 에피소드 종료에서만 끊어, 소진 전 선택에도 이후 팀 결과를 반영합니다.

기본 state-space는 아군 5대와 표적 5대로 고정합니다. 표적 record 0–2는 편제 1, record 3–4는 편제 2에 고정되어 에피소드마다 편제 구간이 뒤섞이지 않습니다.

타입별 life 가치는 타입 1이 2.5, 타입 2가 2, 타입 3이 1입니다. `B`는 에피소드 시작 시 편제 1의 타입·life 구성으로 고정합니다. `D`는 편제 구분 없이 지금까지 감소시킨 life의 타입별 가치를 누적합니다. 매 step 팀 보상은 `-penalty_time × (B-D)/B`이고 활성 드론에게 균등 분배합니다. `D>B`이면 성공이지만 성공만으로 종료하지 않습니다. 아군 전멸 후에도 보상을 미리 당겨 지급하지 않고 horizon까지 진행합니다.

horizon 도달 시 `D<=B`이면 기본 `-100`의 임무 실패 패널티를 추가합니다.

평가의 `score_auc`는 각 step의 `score / baseline_score` 평균입니다. 같은 최종 점수라면 고가치 표적을 먼저 파괴한 정책이 더 높은 값을 얻습니다. 의사결정 CSV에는 선택 당시 표적의 점수, life, 추정 거리를 함께 저장합니다.

모든 타입은 1대부터 타격을 시작합니다. 타입 1의 타격당 참여 상한은 2대, 타입 2·3은 1대입니다. 상한보다 많이 도착하면 가까운 기체부터 참여자로 고정됩니다. 타입 1에 1대가 성공하면 life 1 감소·보상 2.5점이며, 다른 기체가 이후 남은 life를 제거할 수 있습니다. 진행 중인 타격에도 남은 참여 slot이 있으면 늦게 도착한 기체가 합류할 수 있으며, 완료 시 성공 여부와 관계없이 참여자만 소진됩니다. 잉여 기체는 잠기지 않아 다음 step에 다른 표적을 선택할 수 있습니다.

## 알고리즘

타격 참여 상한은 `min(타입별 상한, 남은 life)`입니다. 타입 1에 1대가 먼저 성공한 뒤 나머지 기체들이 도착하더라도 life 1을 제거하는 데에는 1대만 참여합니다. 따라서 명중률 100%에서 타입 1 하나에 총 2대를 초과해 소모하지 않습니다.

CTDE Strike MAPPO를 사용합니다. 공유 actor는 local observation과 agent ID만 받아 생존하며 알려진 표적에 categorical mask를 적용합니다. centralized critic은 joint observation과 타격 진행도에서 공통 팀 value를 계산합니다. critic의 기존 ID 입력 슬롯은 0으로 고정합니다. 할인율 `gamma=0.99`, `GAE lambda=0.99`로 advantage를 계산한 뒤 clipped PPO objective, centralized value loss, entropy bonus를 함께 최적화합니다.

접근 중에는 매 step 재선택을 허용하며 actor loss에 포함합니다. 실제 타격 참여자는 완료까지 action mask를 현재 표적 하나로 제한하고 actor loss에서 제외합니다. 관측의 자기 타격 진행도와 환경의 참여자 마스크를 통해 정책과 학습이 동일한 lock을 사용합니다. critic과 return은 매 step 계산하며 GAE lambda는 0.99를 사용합니다.

동시 선택의 팀 advantage가 같아도 학습 신호를 보존하도록 actor advantage는 평균을 빼지 않고 RMS(최소 1)로만 나눕니다. actor와 critic gradient는 각각 norm 0.5로 제한합니다. 잠금 때문에 실제 선택 샘플이 드물므로 장기 학습에서는 `--rollout-steps 2048`로 여러 에피소드를 모으는 설정을 권장합니다. 이 변경은 관측/체크포인트 구조를 변경하지 않습니다.

정기 평가는 학습 rollout과 다른 고정 seed에서 deterministic actor를 실행합니다. 평가 transition은 replay buffer와 optimizer에 들어가지 않으며, 각 평가 시점의 성공률·점수 비율·섬멸률·score AUC·종료 step을 별도로 기록합니다.

별도 manager, Task option, 연속 공간 goal과 task-conditioned worker는 없습니다. 체크포인트에는 `strike_mappo_v11` architecture tag를 기록해 이전 보상 규칙의 모델이 잘못 로드되지 않게 합니다.

기본 구성은 아군 5대이며 성공 기준은 전체 섬멸이 아니라 `D>B`입니다. 편제 2에서 감소시킨 life 가치도 `D`에 포함되므로 어떤 편제를 공격하든 성공 진행도와 step 보상에 반영됩니다.

모든 활성 드론은 센서 범위·FoV·오차 없이 모든 생존 표적의 현재 위치·타입·life를 정확히 관측합니다. 200스텝 제한은 남은 시간이 관측되는 유한 임무이므로 제한 시점에서 bootstrap을 끊습니다.
