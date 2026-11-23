"""Prioritize initial layouts by observed regret to an assignment upper bound."""
import json
from pathlib import Path
import numpy as np
from .strike_lookahead import completion_steps

def assignment_upper_bound(world):
    if world.c.mode!='known' or world.c.target_motion_scale!=0 or world.c.strike_probability!=1:
        raise ValueError('Oracle-gap curriculum requires known stationary deterministic targets')
    agents=np.flatnonzero(world.agent_active)
    if len(agents)>16:raise ValueError('Exact assignment oracle supports at most 16 active drones')
    masks=np.arange(1<<len(agents));dp=np.full(len(masks),-np.inf);dp[0]=0.
    feasible=completion_steps(world)[agents]<=world.c.horizon-world.t
    for target in np.flatnonzero(world.target_exists & ~world.destroyed):
        value={1:2.5,2:2.,3:1.}[int(world.target_type[target])]
        for _ in range(int(world.target_life[target])):
            nxt=dp.copy()
            for agent in np.flatnonzero(feasible[:,target]):
                bit=1<<int(agent);source=masks[(masks&bit)==0]
                np.maximum.at(nxt,source|bit,dp[source]+value)
            dp=nxt
    return float(dp.max())

class FailureCurriculum:
    def __init__(self,probability=.5,capacity=512):
        if not 0<=probability<=1 or capacity<1:raise ValueError('Invalid curriculum settings')
        self.probability=probability;self.capacity=capacity;self.pool={}
        self.bounds={};self.fresh=0;self.replayed=0;self.observations=[]
    def reset(self,world,rng):
        replay=bool(self.pool) and rng.random()<self.probability
        if replay:
            seeds=list(self.pool);weights=np.array([self.pool[s]['priority'] for s in seeds]);weights/=weights.sum()
            seed=int(rng.choice(seeds,p=weights));self.replayed+=1
        else:seed=int(rng.integers(2**31));self.fresh+=1
        world.reset(seed)
        if seed not in self.bounds:self.bounds[seed]=assignment_upper_bound(world)
        world.curriculum_seed=seed;world.curriculum_oracle=self.bounds[seed];world.curriculum_replayed=replay
    def record(self,world,score):
        seed=world.curriculum_seed;oracle=world.curriculum_oracle;gap=max(0.,oracle-float(score))
        if gap<.5:self.pool.pop(seed,None)
        else:
            self.pool[seed]=dict(seed=seed,oracle=oracle,score=float(score),gap=gap,priority=gap/max(oracle,1.))
            if len(self.pool)>self.capacity:self.pool.pop(min(self.pool,key=lambda s:self.pool[s]['priority']))
        row=dict(seed=seed,oracle_score=oracle,score=float(score),oracle_gap=gap,curriculum_replayed=world.curriculum_replayed)
        self.observations.append(row)
        return row
    def save(self,path):
        rows=sorted(self.pool.values(),key=lambda r:r['priority'],reverse=True)
        Path(path).write_text(json.dumps(dict(probability=self.probability,capacity=self.capacity,fresh_episodes=self.fresh,replayed_episodes=self.replayed,pool=rows,episodes=self.observations),indent=2))
