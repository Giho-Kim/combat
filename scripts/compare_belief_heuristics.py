"""Paired belief-only heuristics, saved trajectories, videos and completion counts."""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.demo_polygon_mission import raster
from scripts.demo_belief_polygons import generate, contains, vertices, lanes
import scripts.demo_belief_polygons as planner
from pointmass_rl.env import Config, target_remaining_value
from pointmass_rl.belief_env import BeliefWorld
from pointmass_rl.navigation import DubinsPath, point_path


def simulate(config, seed, policy):
    world = BeliefWorld(config)
    world.reset(seed)
    w, b = world._world, world.belief
    nav = [None] * w.n
    scouts = set(range(max(0, w.n-2), w.n)) if policy == 2 else set()
    frames, events = [], []
    cap = max(float(raster(b, 'total').max()), 1e-12)
    planner.SPEED=config.strike_speed/10
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
                duration=sum(path.length for path in paths)/config.strike_speed
                sweep=sum(path.length for path in paths[1:])/config.strike_speed
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
        tracks = [t for t in b.tracks if t['life'] > 0]
        if policy == 2:
            tracks.sort(key=lambda t: (-target_remaining_value(t['type'], t['life']), t['type']))
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
            for i in available[:max(0, int(t['life'])-existing)]:
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
            discovered=len(b.tracks), completed=int(w.destroyed[w.target_exists].sum()),
            tracks=[dict(position=(t['mean']*10).tolist(), type=int(t['type']), life=int(t['life'])) for t in b.tracks],
            truth=[dict(position=(w.targets[j]/10).tolist(), type=int(w.target_type[j]),
                        life=int(w.target_life[j])) for j in np.flatnonzero(w.target_exists)]))

    assign(); record()
    for _ in range(config.horizon):
        old = w.pos.copy(); before = w.agent_active.copy()
        raw = np.full(w.n, -1, dtype=int)
        for i, n in enumerate(nav):
            if not w.agent_active[i] or n is None:
                continue
            if n['kind']=='target': raw[i]=n['handle']
            if w.strike_participants[:,i].any(): continue
            distance = config.strike_speed*config.dt
            while distance > 1e-9 and n['paths']:
                w.pos[i], w.heading[i], distance = n['paths'][0].move(w.pos[i],w.heading[i],distance)
                if n['paths'][0].done: n['paths'].pop(0)
            if not n['paths'] and n['kind'] != 'target': nav[i]=None
        w.vel=(w.pos-old)/config.dt; w.t+=1; w.last_damage_by_agent.fill(0)
        w._resolve_strikes(raw)
        for i in np.flatnonzero(before & ~w.agent_active):
            events.append(dict(step=int(w.t), drone=int(i), target=int(raw[i]), event='consume'))
            for t in b.tracks:
                if t['handle']==raw[i]: t.update(life=int(w.target_life[raw[i]]), seen=w.t)
            nav[i]=None
        world._sense()
        if w.agent_active.any(): assign()
        if w.t % 10 == 0 or not w.agent_active.any(): record()
        if not w.agent_active.any(): break
    if frames[-1]['step'] != w.t: record()
    summary=dict(seed=seed, heuristic=policy, completed=int(w.destroyed[w.target_exists].sum()),
                 total=int(w.target_exists.sum()), discovered=len(b.tracks), steps=int(w.t),
                 mean_polygon_generation_ms=float(np.mean(generation_times)))
    return dict(summary=summary,frames=frames,events=events)


def video(a, b, path, config):
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
    colors=['#0079bb','#8955c4','#278433','#db7725','#bf467b']
    cmd=['ffmpeg','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','1200x690',
         '-r','12','-i','-','-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p',str(path)]
    process=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    try:
        for idx in range(max(len(a['frames']),len(b['frames']))):
            im=Image.new('RGB',(1200,690),'white'); draw=ImageDraw.Draw(im)
            for side,data in enumerate((a,b)):
                frame=data['frames'][min(idx,len(data['frames'])-1)]; ox=side*600
                name='belief polygons' if side==0 else '2 scouts + type priority'
                draw.text((ox+20,12),f'H{side+1}: {name} | seed {data["summary"]["seed"]} | {frame["step"]} s',font=font,fill='#223348')
                draw.text((ox+20,40),f'Removed {frame["completed"]}/{data["summary"]["total"]}   Discovered {frame["discovered"]}',font=font,fill='#223348')
                heat=np.asarray(frame['heat'],dtype=float)/255
                stops=np.array([[242,249,252],[100,204,192],[255,226,86],[240,84,55]])
                rgb=np.stack([np.interp(heat,np.linspace(0,1,4),stops[:,c]) for c in range(3)],-1).astype('uint8')
                img=Image.fromarray(rgb[::-1]).resize((432,432),Image.Resampling.BILINEAR)
                im.paste(img,(ox+84,124)); draw.rectangle((ox+30,70,ox+570,610),outline='#8395a6',width=2)
                def xy(p): return (ox+30+p[0]*54,610-p[1]*54)
                for p in frame.get('polygons',[]):
                    v=[xy(z) for z in p['vertices']];v.append(v[0])
                    assigned=p['assigned']
                    draw.line(v,fill=colors[assigned%5] if assigned>=0 else '#a5b0ba',
                              width=3 if assigned>=0 else 1)
                placed=[]
                for t in frame['truth']:
                    x,y=xy(t['position']); r=7
                    draw.ellipse((x-r,y-r,x+r,y+r),outline='#db5290',width=2)
                    if t['life']<=0: draw.line((x-r,y-r,x+r,y+r),fill='#db5290',width=2)
                    label=f'Type {t["type"]} L{t["life"]}'
                    lx=min(x+9,ox+565-draw.textlength(label,font=font)); ly=y-10
                    width=draw.textlength(label,font=font)
                    while any(lx<px+pw and lx+width>px and ly<py+21 and ly+21>py for px,py,pw in placed):
                        ly+=22
                    placed.append((lx,ly,width))
                    if abs(ly-y)>15: draw.line((x,y,lx,ly+8),fill='#823151',width=1)
                    draw.text((lx,ly),label,font=font,fill='#823151')
                for t in frame['tracks']:
                    if t['life']>0:
                        x,y=xy(t['position']); draw.ellipse((x-4,y-4,x+4,y+4),fill='#ffd526',outline='black')
                for i,p in enumerate(frame['pos']):
                    trail=[xy(f['pos'][i]) for f in data['frames'][max(0,idx-30):min(idx+1,len(data['frames']))]]
                    if len(trail)>1: draw.line(trail,fill=colors[i%5],width=2)
                    x,y=xy(p)
                    if frame['active'][i]:
                        angle=-frame['heading'][i]*180/np.pi; half=config.sensor_fov_deg/2; r=config.sensor_range/10*54
                        draw.arc((x-r,y-r,x+r,y+r),angle-half,angle+half,fill='#788999',width=1)
                        draw.ellipse((x-4,y-4,x+4,y+4),fill=colors[i%5])
                    else: draw.line((x-4,y-4,x+4,y+4),fill='#8e98a3',width=2)
                draw.text((ox+20,628),'Belief: pale → red | pink: truth | yellow: detected',font=font,fill='#223348')
                draw.text((ox+20,655),'Polygons: grey candidates / drone color assigned',font=font,fill='#223348')
            process.stdin.write(im.tobytes())
    finally:
        process.stdin.close()
    if process.wait(): raise RuntimeError('ffmpeg failed')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',default='runs/belief_polygon_heuristic_comparison')
    parser.add_argument('--seeds',type=int,nargs='+',default=list(range(10000,10010)))
    parser.add_argument('--render-only',action='store_true',help='Use saved trajectories without simulation.')
    parser.add_argument('--simulate-only',action='store_true',help='Save trajectories without video encoding.')
    args=parser.parse_args(); out=ROOT/args.out; out.mkdir(parents=True,exist_ok=True)
    config=Config.load(ROOT/'configs/belief.json')
    rows=[]
    for seed in args.seeds:
        pair=[]
        for policy in (1,2):
            cached=out/f'seed_{seed}_h{policy}.json'
            data=json.loads(cached.read_text()) if args.render_only else simulate(config,seed,policy)
            pair.append(data); rows.append(data['summary'])
            (out/f'seed_{seed}_h{policy}.json').write_text(json.dumps(data))
            print(json.dumps(data['summary']),flush=True)
        if not args.simulate_only: video(*pair,out/f'seed_{seed}.mp4',config)
    if args.simulate_only: return
    with (out/'results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    summary={f'h{p}':dict(mean_completed=float(np.mean([r['completed'] for r in rows if r['heuristic']==p])),
                         total_completed=sum(r['completed'] for r in rows if r['heuristic']==p)) for p in (1,2)}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    (out/'rules.json').write_text(json.dumps(dict(seeds=args.seeds,config=config.__dict__,
        h1='Existing belief-based trapezoid generator, disjoint polygon IDs, 100m lawnmower with Dubins connectors; immediate target processing.',
        h2='Two scouts cover disjoint map halves in 100m lanes until half horizon; other drones use greedy search and type-value priority. Scouts join type-value priority after half horizon.',
        metric='Count of targets with zero remaining life; same scenario seeds; public belief only.'),indent=2))
    print(summary,flush=True)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    seeds=args.seeds; x=np.arange(len(seeds))
    fig,ax=plt.subplots(figsize=(10,4.4))
    for policy,color,offset in ((1,'#2563eb',-.19),(2,'#d97706',.19)):
        values=[next(r['completed'] for r in rows if r['seed']==seed and r['heuristic']==policy) for seed in seeds]
        ax.bar(x+offset,values,.36,color=color,label=f'H{policy}: mean {np.mean(values):.2f} / 5')
    ax.set_xticks(x,[str(seed) for seed in seeds]); ax.set_xlabel('Scenario seed')
    ax.set_ylabel('Targets removed'); ax.set_ylim(0,5.5); ax.set_yticks(range(6))
    ax.legend(frameon=False); ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',alpha=.2); ax.set_axisbelow(True); fig.tight_layout()
    for ext in ('pdf','png'): fig.savefig(out/f'comparison.{ext}',dpi=180)
    plt.close(fig)
    cards=''.join(f'<section><h2>Seed {seed}</h2><video controls preload="metadata" src="seed_{seed}.mp4"></video></section>' for seed in seeds)
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Belief heuristic comparison</title>'
        '<style>body{font:16px system-ui;max-width:1200px;margin:24px auto;padding:0 20px}video,img{width:100%}section{margin:32px 0}</style>'
        '<h1>Polygon belief heuristic comparison</h1><p>H1: belief-based polygons, lawnmower search, immediate target processing.<br>'
        'H2: two systematic scouts for the first 900 seconds; remaining drones process by type value. '
        'Scouts join target processing after 900 seconds.</p><img src="comparison.png">'+cards)


if __name__=='__main__': main()
