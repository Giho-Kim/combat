"""Paired belief-only heuristics, saved trajectories, videos and completion counts."""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.demo_polygon_mission import raster
from scripts.demo_belief_polygons import generate, contains, vertices, lanes
import scripts.demo_belief_polygons as planner
from pointmass_rl.env import Config, target_remaining_value
from pointmass_rl.belief_env import BeliefWorld, _BeliefLayoutWorld
from pointmass_rl.navigation import DubinsPath, point_path


class OppositeFormationDemoWorld(_BeliefLayoutWorld):
    """Demo-only layout: retain F1 reports and cluster F2 across the map."""
    def _reset_once(self, seed, success_threshold=None, evaluation_task=None):
        super()._reset_once(seed, success_threshold, evaluation_task)
        c=self.c;ids=np.flatnonzero(self.target_exists & (self.target_formation==1))
        f1=self.targets[self.target_exists & (self.target_formation==0)].mean(axis=0)
        margin=c.belief_search_margin+c.target_formation_radius
        center=np.clip(c.size-f1,margin,c.size-margin)
        occupied=list(self.targets[self.target_exists & (self.target_formation!=1)])
        points=[]
        for _ in ids:
            point=self._sample_cluster_member(center,c.target_formation_radius,occupied,c.target_member_min_spacing)
            points.append(point);occupied.append(point)
        self.targets[ids]=points;self.spawn_position[ids]=points
        self.formation_centers[1]=np.mean(points,axis=0)
        self.observation=self._obs()
        return self.observation.copy()


class ArcFormationDemoWorld(_BeliefLayoutWorld):
    """Sample F2's centre uniformly on the feasible half-map-radius arc."""
    def _reset_once(self,seed,success_threshold=None,evaluation_task=None):
        super()._reset_once(seed,success_threshold,evaluation_task)
        c=self.c;ids=np.flatnonzero(self.target_exists & (self.target_formation==1))
        if len(ids)!=2:raise ValueError('Arc demo expects two F2 members')
        f1=self.targets[self.target_exists & (self.target_formation==0)].mean(axis=0)
        radius=.5*c.size;cluster_radius=.5*c.target_formation_radius
        for _ in range(20000):
            angle=self.rng.uniform(0,2*np.pi);center=f1+radius*np.array([np.cos(angle),np.sin(angle)])
            if np.any(center<c.belief_search_margin+cluster_radius) or np.any(center>c.size-c.belief_search_margin-cluster_radius):continue
            orientation=self.rng.uniform(0,2*np.pi);length=self.rng.uniform(c.target_member_min_spacing/2,cluster_radius)
            offset=length*np.array([np.cos(orientation),np.sin(orientation)]);points=np.array([center-offset,center+offset])
            if not all(self._valid_spawn(point,c.min_spawn_distance) for point in points):continue
            break
        else:raise RuntimeError('No feasible F2 arc placement')
        self.targets[ids]=points;self.spawn_position[ids]=points;self.formation_centers[1]=center
        self.observation=self._obs();return self.observation.copy()


def simulate(config, seed, policy, opposite_formation_two=False, arc_formation_two=False):
    world = BeliefWorld(config)
    if opposite_formation_two:world._world=OppositeFormationDemoWorld(world._world.c)
    if arc_formation_two:world._world=ArcFormationDemoWorld(world._world.c)
    world.reset(seed)
    w, b = world._world, world.belief
    nav = [None] * w.n
    scouts = set(range(max(0, w.n-2), w.n)) if policy == 2 else set()
    frames, events = [], []
    cap = max(float(raster(b, 'total').max()), 1e-12)
    planner.SPEED=config.final_mission_speed/10
    planner.RADIUS=config.belief_turn_radius/10
    proposals=[]
    last_plan=-1000
    generation_times=[]

    def replan():
        nonlocal proposals,last_plan
        active=np.flatnonzero(w.agent_active)
        if not len(active): return
        locked=[n['polygon'] for n in nav if n and n['kind']=='search']
        available=[i for i in active if nav[i] is None or nav[i]['kind']=='search']
        anchor=available[0] if available else active[0]
        points=b.points.reshape(-1,2)*config.size/10
        proposals,ms=generate(points,b.weights.ravel(),w.pos[anchor]/10,w.heading[anchor],
                              proposals,count=12,locked_ids=locked)
        generation_times.append(ms);last_plan=w.t

    def search(i,p):
        options=[]
        for reverse in (False,True):
            for flip in (False,True):
                pos,h=w.pos[i].copy(),w.heading[i];paths=[]
                for goal,angle in lanes(p['params'],reverse,flip):
                    dest=goal*10
                    paths.append(DubinsPath(pos,h,dest,angle,config.belief_turn_radius))
                    pos,h=dest,angle
                duration=sum(path.length for path in paths)/config.final_mission_speed
                sweep=sum(path.length for path in paths[1:])/config.final_mission_speed
                if sweep<=planner.BUDGET: options.append((duration,paths))
        if not options: return False
        _,paths=min(options,key=lambda item:item[0])
        nav[i]=dict(kind='search',polygon=p['id'],params=p['params'].copy(),paths=paths,
                    goal=vertices(p['params']).mean(0)*10)
        return True

    def target(i, track):
        goal = track['mean'] * config.size
        nav[i] = dict(kind='target', handle=track['handle'], goal=goal,
                      paths=[point_path(w.pos[i], w.heading[i], goal, config.belief_turn_radius)])

    def patrol(i):
        # Split the coverage region into disjoint left/right halves.
        lo, hi = config.belief_search_margin, config.size-config.belief_search_margin
        mid = (lo+hi)/2
        side = sorted(scouts).index(i)
        xs = np.arange(lo+.5, mid if side == 0 else hi, 1.) if side == 0 else np.arange(mid+.5, hi, 1.)
        # Spread the first pass across the half-map; subsequent passes fill
        # the intervening 100m lanes rather than starting with adjacent lanes.
        xs = xs[np.concatenate([np.arange(offset, len(xs), 8) for offset in range(8)])]
        paths, pos, angle = [], w.pos[i].copy(), w.heading[i]
        for row, x in enumerate(xs):
            heading = np.pi/2 if row % 2 == 0 else -np.pi/2
            for y in ((lo-2, hi+2) if row % 2 == 0 else (hi+2, lo-2)):
                goal = np.array([x, y])
                paths.append(DubinsPath(pos, angle, goal, heading, config.belief_turn_radius))
                pos, angle = goal, heading
        left,right=(lo,mid) if side==0 else (mid,hi)
        region=np.array([[left,lo],[right,lo],[right,hi],[left,hi]])/10
        nav[i] = dict(kind='patrol', goal=pos,paths=paths,region=region)

    def assign():
        for i in scouts:
            if w.agent_active[i] and w.t < config.horizon//2 and nav[i] is None:
                patrol(i)
            if w.t >= config.horizon//2 and nav[i] and nav[i]['kind'] == 'patrol':
                nav[i] = None
        tracks = [t for t in b.tracks if t['work_units'] > 0]
        if policy == 2:
            tracks.sort(key=lambda t: (-target_remaining_value(t['type'], t['work_units']), t['type']))
        else:
            tracks.sort(key=lambda t: min(np.linalg.norm(w.pos[i]-t['mean']*config.size)
                                         for i in range(w.n) if w.agent_active[i]))
        for t in tracks:
            existing = sum(n is not None and n['kind']=='target' and n['handle']==t['handle']
                           and w.agent_active[i] for i, n in enumerate(nav))
            available = [i for i in range(w.n) if w.agent_active[i]
                         and not (i in scouts and w.t < config.horizon//2)
                         and (nav[i] is None or nav[i]['kind']=='search')]
            available.sort(key=lambda i: np.linalg.norm(w.pos[i]-t['mean']*config.size))
            for i in available[:max(0, int(t['work_units'])-existing)]:
                target(i, t)
        if w.t-last_plan>=60 or not proposals or any(
                w.agent_active[i] and nav[i] is None for i in range(w.n)):
            replan()
        points=b.points.reshape(-1,2)*config.size/10
        busy={n['polygon'] for n in nav if n and n['kind']=='search'}
        reserved=np.zeros(len(points),dtype=bool)
        for n in nav:
            if n and n['kind']=='search': reserved |= contains(n['params'],points)
        for i in range(w.n):
            if not w.agent_active[i] or nav[i] is not None:
                continue
            eligible=[p for p in proposals if p['id'] not in busy]
            def value(p):
                mask=contains(p['params'],points)
                mass=float(b.weights.ravel()[mask & ~reserved].sum())
                distance=np.linalg.norm(w.pos[i]/10-p['params'][:2])
                return mass/(1+distance)
            for p in sorted(eligible,key=value,reverse=True):
                if search(i,p):
                    busy.add(p['id']); reserved |= contains(p['params'],points);break

    def record():
        heat = np.rint(np.sqrt(np.clip(raster(b,'total')/cap,0,1))*255).astype(np.uint8)
        frames.append(dict(step=int(w.t), pos=(w.pos/10).tolist(), heading=w.heading.tolist(),
            active=w.agent_active.tolist(), heat=heat.tolist(),
            polygons=[dict(id=p['id'],vertices=vertices(p['params']).tolist(),
                assigned=next((i for i,n in enumerate(nav) if n and n['kind']=='search'
                               and n['polygon']==p['id']),-1)) for p in proposals]
                +[dict(id=-i-1,vertices=n['region'].tolist(),assigned=i)
                   for i,n in enumerate(nav) if n and n['kind']=='patrol'],
            discovered=len(b.tracks), completed=int(w.target_completed[w.target_exists].sum()),
            tracks=[dict(position=(t['mean']*10).tolist(), type=int(t['type']), work_units=int(t['work_units'])) for t in b.tracks],
            truth=[dict(position=(w.targets[j]/10).tolist(), type=int(w.target_type[j]),
                        work_units=int(w.target_work_units[j])) for j in np.flatnonzero(w.target_exists)]))

    assign(); record()
    for _ in range(config.horizon):
        old = w.pos.copy(); before = w.agent_active.copy()
        raw = np.full(w.n, -1, dtype=int)
        for i, n in enumerate(nav):
            if not w.agent_active[i] or n is None:
                continue
            if n['kind']=='target': raw[i]=n['handle']
            if w.final_mission_participants[:,i].any(): continue
            distance = config.final_mission_speed*config.dt
            while distance > 1e-9 and n['paths']:
                w.pos[i], w.heading[i], distance = n['paths'][0].move(w.pos[i],w.heading[i],distance)
                if n['paths'][0].done: n['paths'].pop(0)
            if not n['paths'] and n['kind'] != 'target': nav[i]=None
        w.vel=(w.pos-old)/config.dt; w.t+=1; w.last_progress_value_by_agent.fill(0)
        w._resolve_final_missions(raw)
        for i in np.flatnonzero(before & ~w.agent_active):
            events.append(dict(step=int(w.t), drone=int(i), target=int(raw[i]), event='service_end'))
            for t in b.tracks:
                if t['handle']==raw[i]: t.update(work_units=int(w.target_work_units[raw[i]]), seen=w.t)
            nav[i]=None
        world._sense()
        if w.agent_active.any(): assign()
        if w.t % 10 == 0 or not w.agent_active.any(): record()
        if not w.agent_active.any(): break
    if frames[-1]['step'] != w.t: record()
    summary=dict(seed=seed, heuristic=policy, completed=int(w.target_completed[w.target_exists].sum()),
                 total=int(w.target_exists.sum()), discovered=len(b.tracks), steps=int(w.t),
                 mean_polygon_generation_ms=float(np.mean(generation_times)))
    summary.update(completed_f1=int(w.target_completed[w.target_exists & (w.target_formation==0)].sum()),completed_f2=int(w.target_completed[w.target_exists & (w.target_formation==1)].sum()))
    return dict(summary=summary,initial_intel=b.initial_report,frames=frames,events=events,layout=dict(targets=w.spawn_position.tolist(),formations=w.target_formation.tolist(),centers=w.formation_centers.tolist()))


from scripts.belief_video_style import video


def add_value_metrics(data):
    first=data['frames'][0]['truth'];last=data['frames'][-1]['truth']
    earned=np.array([target_remaining_value(a['type'],a['work_units'])-target_remaining_value(b['type'],b['work_units']) for a,b in zip(first,last)])
    formation=np.array(data['layout']['formations'])[:len(earned)]
    data['summary'].update(value=float(earned.sum()),value_f1=float(earned[formation==0].sum()),value_f2=float(earned[formation==1].sum()))
    return data


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',default='runs/belief_polygon_heuristic_comparison')
    parser.add_argument('--seeds',type=int,nargs='+',default=list(range(10000,10010)))
    parser.add_argument('--opposite-formation-two',action='store_true',help='Demo only: cluster F2 opposite F1 across the map')
    parser.add_argument('--arc-formation-two',action='store_true',help='Demo only: random F2 cluster on a half-map-radius arc around F1')
    parser.add_argument('--no-video',action='store_true',help='Save comparisons without encoding videos')
    parser.add_argument('--render-only',action='store_true',help='Use saved trajectories without simulation.')
    parser.add_argument('--simulate-only',action='store_true',help='Save trajectories without video encoding.')
    args=parser.parse_args()
    if args.arc_formation_two and args.opposite_formation_two:parser.error('Choose one F2 layout')
    out=ROOT/args.out; out.mkdir(parents=True,exist_ok=True)
    config=Config.load(ROOT/'configs/belief.json')
    rows=[]
    for seed in args.seeds:
        pair=[]
        for policy in (1,2):
            cached=out/f'seed_{seed}_h{policy}.json'
            data=json.loads(cached.read_text()) if args.render_only else simulate(config,seed,policy,args.opposite_formation_two,args.arc_formation_two)
            data=add_value_metrics(data)
            pair.append(data); rows.append(data['summary'])
            (out/f'seed_{seed}_h{policy}.json').write_text(json.dumps(data))
            print(json.dumps(data['summary']),flush=True)
        if not args.simulate_only and not args.no_video: video(*pair,out/f'seed_{seed}.mp4',config)
    if len(args.seeds)==1:
        (out/'initial_report.json').write_text(json.dumps(pair[0]['initial_intel'],indent=2))
        if not args.simulate_only and not args.no_video:
            import shutil
            shutil.copy2(out/f'seed_{args.seeds[0]}.png',out/'preview.png')
    if args.simulate_only: return
    with (out/'results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0],lineterminator='\n'); writer.writeheader(); writer.writerows(rows)
    summary={f'h{p}':dict(mean_completed=float(np.mean([r['completed'] for r in rows if r['heuristic']==p])),
                         total_completed=sum(r['completed'] for r in rows if r['heuristic']==p)) for p in (1,2)}
    for policy in (1,2):
        subset=[r for r in rows if r['heuristic']==policy]
        summary[f'h{policy}'].update(mean_value=float(np.mean([r['value'] for r in subset])),mean_value_f2=float(np.mean([r['value_f2'] for r in subset])))
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    (out/'rules.json').write_text(json.dumps(dict(seeds=args.seeds,config=config.__dict__,
        h1='Existing belief-based trapezoid generator, disjoint polygon IDs, 100m lawnmower with Dubins connectors; immediate target processing.',
        h2='Two scouts cover disjoint map halves in 100m lanes until half horizon; other drones use greedy search and type-value priority. Scouts join type-value priority after half horizon.',
        formation_two_layout='uniform feasible arc, radius half map side, clustered pair' if args.arc_formation_two else 'opposite F1, clustered' if args.opposite_formation_two else 'configured',
        metric='Earned type-weighted value: type 1 = 2.5 per work_units, type 2 = 2, type 3 = 1; includes partial progress on type 1.'),indent=2))
    print(summary,flush=True)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    seeds=args.seeds; x=np.arange(len(seeds))
    fig,axes=plt.subplots(1,2 if args.arc_formation_two else 1,figsize=(13,4.4) if args.arc_formation_two else (10,4.4),squeeze=False)
    ax=axes[0,0]
    for policy,color,offset in ((1,'#2563eb',-.19),(2,'#d97706',.19)):
        values=[next(r['value'] for r in rows if r['seed']==seed and r['heuristic']==policy) for seed in seeds]
        ax.bar(x+offset,values,.36,color=color,label=f'H{policy}: mean {np.mean(values):.2f}')
    ax.set_xticks(x,[str(seed) for seed in seeds]); ax.set_xlabel('Scenario seed')
    ax.set_ylabel('Earned target value'); ax.set_ylim(0,max(1,max(r['value'] for r in rows))*1.25)
    ax.legend(frameon=False); ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',alpha=.2); ax.set_axisbelow(True)
    if args.arc_formation_two:
        ax=axes[0,1]
        for policy,color,offset in ((1,'#2563eb',-.19),(2,'#d97706',.19)):
            values=[next(r['value_f2'] for r in rows if r['seed']==seed and r['heuristic']==policy) for seed in seeds]
            ax.bar(x+offset,values,.36,color=color,label=f'H{policy}: mean {np.mean(values):.2f}')
        ax.set_xticks(x,[str(seed) for seed in seeds]);ax.set(xlabel='Scenario seed',ylabel='Formation 2 earned value',ylim=(0,max(1,max(r['value_f2'] for r in rows))*1.3));ax.legend(frameon=False);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
    fig.tight_layout()
    for ext in ('pdf','png'): fig.savefig(out/f'comparison.{ext}',dpi=180)
    plt.close(fig)
    cards='' if args.no_video else ''.join(f'<section><h2>Seed {seed}</h2><video controls preload="metadata" src="seed_{seed}.mp4"></video></section>' for seed in seeds)
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Belief heuristic comparison</title>'
        '<style>body{font:16px system-ui;max-width:1200px;margin:24px auto;padding:0 20px;background:#101826;color:#e7edf7}video,img{width:100%;border-radius:10px}section{margin:32px 0}</style>'
        '<h1>Polygon belief heuristic comparison</h1><p>H1: belief-based polygons, lawnmower search, immediate target processing.<br>'
        'H2: two systematic scouts for the first 900 seconds; remaining drones process by type value. '
        'Scouts join target processing after 900 seconds.</p><img src="comparison.png">'+cards)


if __name__=='__main__': main()
