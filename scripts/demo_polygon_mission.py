"""Playable polygon search and target-consumption demo using real game rules."""
from pathlib import Path
import sys, json, time, argparse
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.demo_belief_polygons import generate, contains, vertices, lanes, route, density
import scripts.demo_belief_polygons as planner
from pointmass_rl.env import Config, DAMAGE_REWARD_RATE
from pointmass_rl.belief_env import BeliefWorld
from pointmass_rl.navigation import DubinsPath, point_path

OUT = ROOT/'runs/belief_polygon_mission'


def patrol_waypoints(position, config):
    """Systematic coverage independent of belief, reports and hidden targets.

    Start near the current position, visit every fourth lane first and then
    fill the intervening lanes. A pass includes each 100 m lane exactly once.
    """
    lo, hi = config.belief_search_margin, config.size-config.belief_search_margin
    spacing = planner.SPACING*10
    count = int(np.ceil((hi-lo)/spacing))
    axis = lo+(np.arange(count)+.5)*(hi-lo)/count
    vertical = min(abs(position[1]-lo), abs(position[1]-hi)) <= min(abs(position[0]-lo), abs(position[0]-hi))
    cross_axis = 0 if vertical else 1
    along_axis = 1-cross_axis
    first = int(np.argmin(abs(axis-position[cross_axis])))
    stride = max(1, int(np.ceil(2*config.belief_turn_radius/spacing)))
    order = []
    current = first
    for offset in range(stride):
        group = list(range((first+offset)%stride, count, stride))
        if not group:
            continue
        closest = int(np.argmin(abs(np.asarray(group)-current)))
        group = group[closest:]+group[:closest]
        order.extend(group); current = group[-1]
    extension = min(config.sensor_range, lo*.5)
    lower, upper = lo-extension, hi+extension
    forward = abs(position[along_axis]-lower) <= abs(position[along_axis]-upper)
    result = []
    for row in order:
        a, z = np.zeros(2), np.zeros(2)
        a[cross_axis] = z[cross_axis] = axis[row]
        a[along_axis], z[along_axis] = (lower,upper) if forward else (upper,lower)
        heading = (np.pi/2 if vertical else 0.)+(0 if forward else np.pi)
        result.extend([(a,heading),(z,heading)])
        forward = not forward
    return result


def raster(b, mode):
    # Reuse the continuous-density caches, without replaying sensor history.
    area = (b.c.size-2*b.c.belief_search_margin)**2/(b.cells*b.display_side**2)
    background = b.background_mass*np.exp(b._background_log_display-b._background_log_normalizer)*area
    values = b.weights if mode == 'total' else background if mode == 'background' else b.weights-background
    g,s=b.c.belief_grid,b.display_side
    return (values/area*100).reshape(g,g,s,s).transpose(0,2,1,3).reshape(g*s,g*s)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,default=7)
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    config=Config.load(ROOT/'configs/belief.json');world=BeliefWorld(config);world.reset(args.seed)
    planner.SPEED=config.strike_speed/10
    planner.RADIUS=config.belief_turn_radius/10
    w=world._world;b=world.belief
    scout = w.n-1
    nav=[None]*w.n;events=[];frames=[];plans=[];proposals=[];last_plan=-1000;known=set();generation_times=[]
    def event(kind,message,**kw):
        events.append(dict(step=int(w.t),kind=kind,message=message,**kw))
    def set_target(i,track):
        destination=track['mean']*config.size
        nav[i]=dict(kind='target',handle=track['handle'],goal=destination.copy(),
                    paths=[point_path(w.pos[i],w.heading[i],destination,config.belief_turn_radius)])
        event('select',f'D{i}: Type {track["type"]} 표적 선택',drone=i,position=destination.tolist())
    def set_patrol():
        waypoints = patrol_waypoints(w.pos[scout], config)
        pos, heading = w.pos[scout].copy(), w.heading[scout]
        paths = []
        for goal, angle in waypoints:
            paths.append(DubinsPath(pos,heading,goal,angle,config.belief_turn_radius))
            pos, heading = goal, angle
        nav[scout] = dict(kind='patrol', paths=paths, goals=[p.copy() for p,_ in waypoints],
                          goal=waypoints[0][0].copy())
        event('patrol',f'D{scout}: 전역 탐색 전용 · 발견해도 계속 탐색',drone=scout)
    def set_search(i,p):
        # Choose the shortest of the four actual sweep directions for this drone.
        options=[]
        for reverse in (False,True):
            for flip in (False,True):
                pos,h=w.pos[i].copy(),w.heading[i];paths=[]
                for goal,angle in lanes(p['params'],reverse,flip):
                    dest=goal*10;path=DubinsPath(pos,h,dest,angle,config.belief_turn_radius)
                    paths.append(path);pos,h=dest,angle
                total=sum(path.length for path in paths)/config.strike_speed
                sweep=sum(path.length for path in paths[1:])/config.strike_speed
                if sweep<=planner.BUDGET:
                    options.append((total,sweep,paths))
        if not options:
            return False
        total,sweep,paths=min(options,key=lambda x:x[0])
        nav[i]=dict(kind='search',polygon=p['id'],params=p['params'].copy(),paths=paths,
                    seconds=total,goal=vertices(p['params']).mean(0)*10)
        event('search',f'D{i}: P{p["id"]} 탐색 (이동 포함 {total/60:.1f}분)',drone=i)
        return True
    def search_mass():
        # Sensor evidence is already incorporated by the belief update.
        # Use the same current mass that is displayed in the replay.
        return b.weights.ravel()
    def replan():
        nonlocal proposals,last_plan
        active=np.flatnonzero(w.agent_active & (np.arange(w.n)!=scout))
        if not len(active):return
        points=b.points.reshape(-1,2)*config.size/10;mass=search_mass()
        locked=[n['polygon'] for n in nav if n and n['kind']=='search']
        available=[i for i in active if nav[i] is None or nav[i]['kind']=='search']
        anchor=available[0] if available else active[0]
        proposals,ms=generate(points,mass,w.pos[anchor]/10,w.heading[anchor],proposals,count=12,locked_ids=locked)
        generation_times.append(ms);last_plan=w.t
        plans.append(dict(step=int(w.t),generation_ms=ms,polygons=[dict(id=p['id'],vertices=vertices(p['params']).tolist(),
            mass=float(mass[contains(p['params'],points)].sum()),locked=p['id'] in locked) for p in proposals]))
    def assign():
        # Public tracks only. Consume capacity is the track's observed life.
        for track in b.tracks:
            if track['life']<=0:continue
            existing=[i for i,n in enumerate(nav) if w.agent_active[i] and n and n['kind']=='target' and n['handle']==track['handle']]
            slots=max(0,track['life']-len(existing))
            available=[i for i in range(w.n) if i!=scout and w.agent_active[i] and (nav[i] is None or nav[i]['kind']=='search')]
            available.sort(key=lambda i:np.linalg.norm(w.pos[i]-track['mean']*config.size))
            for i in available[:slots]:set_target(i,track)
        busy={n['polygon'] for n in nav if n and n['kind']=='search'}
        points=b.points.reshape(-1,2)*config.size/10
        for i in range(w.n):
            if i==scout:continue
            if not w.agent_active[i] or nav[i] is not None:continue
            eligible=[p for p in proposals if p['id'] not in busy]
            def value(p):
                mass=float(search_mass()[contains(p['params'],points)].sum())
                distance=np.linalg.norm(w.pos[i]/10-p['params'][:2])
                return mass/(1+distance)
            for p in sorted(eligible,key=value,reverse=True):
                if set_search(i,p):busy.add(p['id']);break
    color_limits={key:max(float(raster(b,key).max()),1e-12)*(2 if key=='background' else 1)
                  for key in ('total','reported','background')}
    def record():
        fields={key:raster(b,key).ravel() for key in ('total','reported','background')}
        images={};maximum={}
        for key,value in fields.items():
            cap=color_limits[key];maximum[key]=cap
            images[key]=np.rint(np.sqrt(np.clip(value/cap,0,1))*255).astype(np.uint8).tobytes().hex()
        frames.append(dict(step=int(w.t),pos=(w.pos/10).tolist(),heading=w.heading.tolist(),active=w.agent_active.tolist(),
            tasks=[('대기' if n is None else '전역 탐색 전용' if n['kind']=='patrol' else 'P'+str(n['polygon'])+' 탐색' if n['kind']=='search' else
                    '표적 처리 중' if w.strike_participants[:,i].any() else '표적 이동') for i,n in enumerate(nav)],
            goals=[None if n is None else (n['goal']/10).tolist() for n in nav],
            search_ids=[n['polygon'] if n and n['kind']=='search' else None for n in nav],
            progress=w.agent_strike_progress.tolist(),heat=images,maximum=maximum,plan=len(plans)-1,
            tracks=[dict(position=(t['mean']*10).tolist(),type=t['type'],life=t['life']) for t in b.tracks],
            truth=[dict(position=(w.targets[j]/10).tolist(),type=int(w.target_type[j]),life=int(w.target_life[j]),
                        formation=int(w.target_formation[j])+1) for j in np.flatnonzero(w.target_exists)],
            reported_remaining=float(b.remaining_types.sum()),background=b.background_mass,score=float(w.score)))
    begin=time.perf_counter();set_patrol();replan();assign();record()
    for step in range(config.horizon):
        before=w.agent_active.copy();old=w.pos.copy();raw=np.full(w.n,-1,dtype=int);sweep_finished=False
        for i,n in enumerate(nav):
            if not w.agent_active[i] or n is None:continue
            if n['kind']=='target':raw[i]=n['handle']
            if w.strike_participants[:,i].any():continue
            budget=config.strike_speed*config.dt
            while budget>1e-9 and n['paths']:
                w.pos[i],w.heading[i],budget=n['paths'][0].move(w.pos[i],w.heading[i],budget)
                if n['paths'][0].done:
                    n['paths'].pop(0)
                    if n['kind']=='patrol':
                        n['goals'].pop(0)
                        if n['goals']:n['goal']=n['goals'][0].copy()
            if not n['paths'] and n['kind']=='patrol':set_patrol()
            if not n['paths'] and n['kind']=='search':
                event('sweep_end',f'D{i}: P{n["polygon"]} 탐색 완료',drone=i)
                nav[i]=None;sweep_finished=True
        w.vel=(w.pos-old)/config.dt;w.t+=1;w.last_damage_by_agent.fill(0)
        w._resolve_strikes(raw)
        reward=float(DAMAGE_REWARD_RATE*w.last_damage_by_agent.sum());w.rewards_total+=reward
        w.discounted_rewards_total+=config.discount_gamma**(w.t-1)*reward
        consumed=np.flatnonzero(before & ~w.agent_active)
        for i in consumed:
            handle=raw[i]
            for track in b.tracks:
                if track['handle']==handle:track.update(life=int(w.target_life[handle]),seen=w.t)
            event('consume',f'D{i} consume · 표적 남은 life {int(w.target_life[handle])}',drone=int(i),position=w.pos[i].tolist())
            nav[i]=None
        world._sense()
        discovered=False
        for track in b.tracks:
            if track['handle'] not in known:
                known.add(track['handle']);discovered=True
                event('discover',f'Type {track["type"]} 발견 · life {track["life"]}',position=(track['mean']*config.size).tolist())
        if not w.agent_active.any():record();break
        if sweep_finished or w.t-last_plan>=60 or (not proposals):replan()
        assign()
        if w.t%3==0 or discovered or len(consumed):record()
    if frames[-1]['step']!=w.t:record()
    payload=dict(seed=args.seed,scout=scout,frames=frames,plans=plans,events=events,size=10,heat_side=32,
        sensor_range=config.sensor_range/10,sensor_fov=config.sensor_fov_deg,service_steps=config.strike_steps_per_life,
        report=w.initial_intel,summary=dict(steps=int(w.t),discovered=len(known),consumed=int((~w.agent_active).sum()),
            completed=int(w.destroyed[w.target_exists].sum()),score=float(w.score),seconds=time.perf_counter()-begin,
            mean_generation_ms=float(np.mean(generation_times))))
    (OUT/'mission.json').write_text(json.dumps(payload,ensure_ascii=False))
    (OUT/'demo.html').write_text(HTML.replace('__DATA__',json.dumps(payload,ensure_ascii=False)))
    print(json.dumps(payload['summary'],ensure_ascii=False),flush=True)
    print(OUT/'demo.html',flush=True)


HTML=r'''<!doctype html><meta charset="utf-8"><title>Polygon 탐색 · 표적 선택 · Consume</title>
<style>body{background:#101722;color:#e6edf7;font:15px system-ui;max-width:1280px;margin:22px auto;padding:0 18px}h1{font-size:25px}p{color:#b1c1d7;line-height:1.5}.layout{display:grid;grid-template-columns:minmax(500px,850px) 320px;gap:18px}canvas{width:100%;background:#101c2b;border-radius:10px}.controls{display:flex;flex-wrap:wrap;align-items:center;gap:14px;margin:18px 0}button,select{color:white;background:#263b54;padding:9px;border:1px solid #55728e;border-radius:6px}input[type=range]{width:320px}.card{background:#1a2b40;padding:16px;border-radius:10px;margin-bottom:12px}#events{max-height:450px;overflow:auto}.event{padding:9px;border-bottom:1px solid #35465b;cursor:pointer}.event:hover{background:#314660}.small{font-size:12px;color:#adbed5}.legend{height:15px;background:linear-gradient(90deg,#101c2b,#116a8c,#24ccae,#efe942,#ff5a28);border-radius:4px}b{color:#fff}@media(max-width:900px){.layout{grid-template-columns:1fr}}</style>
<h1>탐색 → 발견 → 표적 선택 → Consume</h1><p>D4는 전역 탐색 전용입니다. 발견해도 탐색을 계속하고 정보를 공유합니다. D0–D3가 표적을 선택·처리합니다. 탐색 줄 간격은 100m입니다. 실제 위치 표시는 의사결정 입력과 별도로 켜고 끌 수 있습니다.</p>
<div class="controls"><button id="play">재생</button><select id="speed"><option value="1">3초씩</option><option value="5" selected>15초씩</option><option value="10">30초씩</option></select><input id="time" type="range" min="0" value="0"><b id="clock"></b></div>
<div class="controls"><select id="layer"><option value="total">전체 belief</option><option value="reported">보고 표적 belief</option><option value="background">배경 belief</option></select><label><input id="truth" type="checkbox" checked>실제 표적 위치</label><label><input id="poly" type="checkbox" checked>탐색 polygon</label><label><input id="sensor" type="checkbox" checked>센서 시야</label><label><input id="trail" type="checkbox" checked>이동 궤적</label></div>
<div class="layout"><canvas id="map" width="850" height="850"></canvas><aside><div class="card"><b>Belief 밀도 · 밝고 붉을수록 높음</b><div class="legend"></div><div id="scale" class="small"></div><p class="small">처음부터 끝까지 같은 색상 기준입니다. 붉은색은 범례 상한 이상의 밀도입니다.</p></div><div class="card" id="status"></div><div class="card"><b>이벤트 · 클릭하면 해당 시점으로</b><div id="events"></div></div><p class="small">실제 표적: 분홍 점선 원<br>발견된 표적: 노란 원<br>처리 완료 표적: ×<br>consume된 드론: 회색 ×<br>숫자는 타입·남은 life입니다.</p></aside></div>
<script>
const data=__DATA__,canvas=document.querySelector('#map'),ctx=canvas.getContext('2d'),slider=document.querySelector('#time'),colors=['#65e4ff','#c19bff','#a8e465','#ffa470','#ff91c4'];let timer;let placed=[];
slider.max=data.frames.length-1;const xy=p=>[40+p[0]*77,810-p[1]*77];const stops=[[16,28,43],[17,106,140],[36,204,174],[239,233,66],[255,90,40]];
function color(t){let k=Math.min(3,Math.floor(t*4)),a=t*4-k;return 'rgb('+stops[k].map((v,i)=>Math.round(v*(1-a)+stops[k+1][i]*a)).join(',')+')'}
function cross(p,color,r=7){let[x,y]=xy(p);ctx.strokeStyle=color;ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(x-r,y-r);ctx.lineTo(x+r,y+r);ctx.moveTo(x+r,y-r);ctx.lineTo(x-r,y+r);ctx.stroke()}
function label(text,x,y,color){const width=ctx.measureText(text).width+8;let lx=Math.min(810-width,x+10),ly=y-9;for(let k=0;k<30;k++){if(!placed.some(r=>lx<r[0]+r[2]&&lx+width>r[0]&&ly-14<r[1]+18&&ly+4>r[1]))break;ly+=20}placed.push([lx,ly-14,width]);ctx.fillStyle='rgba(12,22,36,.82)';ctx.fillRect(lx-3,ly-14,width,19);ctx.fillStyle=color;ctx.fillText(text,lx,ly);if(Math.abs(ly-y)>25){ctx.strokeStyle=color;ctx.lineWidth=.8;ctx.beginPath();ctx.moveTo(x,y);ctx.lineTo(lx,ly-7);ctx.stroke()}}
function draw(){placed=[];let ix=+slider.value,f=data.frames[ix],layer=document.querySelector('#layer').value;ctx.clearRect(0,0,850,850);let h=f.heat[layer],n=data.heat_side;const surface=document.createElement('canvas');surface.width=n;surface.height=n;const sc=surface.getContext('2d');for(let j=0;j<n;j++)for(let i=0;i<n;i++){let v=parseInt(h.slice(2*(j*n+i),2*(j*n+i)+2),16)/255;sc.fillStyle=color(v);sc.fillRect(i,n-1-j,1,1)}ctx.imageSmoothingEnabled=true;ctx.imageSmoothingQuality='high';ctx.drawImage(surface,117,117,616,616);ctx.strokeStyle='#6f849e';ctx.lineWidth=1;ctx.strokeRect(40,40,770,770);ctx.fillStyle='#c3d2e5';ctx.font='12px system-ui';for(let k=0;k<=10;k++){ctx.fillText(k,35+k*77,833);ctx.fillText(k,18,815-k*77)}
if(document.querySelector('#poly').checked&&f.plan>=0){data.plans[f.plan].polygons.forEach(p=>{ctx.beginPath();p.vertices.forEach((v,j)=>j?ctx.lineTo(...xy(v)):ctx.moveTo(...xy(v)));ctx.closePath();let assigned=(f.search_ids||[]).indexOf(p.id);ctx.strokeStyle=assigned>=0?colors[assigned%5]:'#b9d6f160';ctx.lineWidth=assigned>=0?2.5:1;ctx.stroke();let m=p.vertices.reduce((a,v)=>[a[0]+v[0]/4,a[1]+v[1]/4],[0,0]);})}
if(document.querySelector('#truth').checked){f.truth.forEach(t=>{if(t.life<=0){cross(t.position,'#ff87b4');return}let[x,y]=xy(t.position);ctx.setLineDash([4,3]);ctx.strokeStyle='#ff75b0';ctx.lineWidth=2.5;ctx.beginPath();ctx.arc(x,y,11,0,2*Math.PI);ctx.stroke();ctx.setLineDash([]);ctx.fillStyle='#ffb3d0';ctx.font='bold 13px system-ui';label('Type '+t.type+' · L'+t.life,x,y,'#ffb3d0')})}
f.tracks.forEach(t=>{if(t.life<=0)return;let[x,y]=xy(t.position);ctx.fillStyle='#fff182';ctx.strokeStyle='#263745';ctx.lineWidth=2;ctx.beginPath();ctx.arc(x,y,6,0,2*Math.PI);ctx.fill();ctx.stroke();ctx.fillStyle='#fff7bf';ctx.font='bold 13px system-ui';label('발견 Type '+t.type+' · L'+t.life,x,y,'#fff7bf')});
f.pos.forEach((p,i)=>{let[x,y]=xy(p);if(document.querySelector('#trail').checked){ctx.strokeStyle=colors[i%5];ctx.lineWidth=1.7;ctx.beginPath();for(let k=Math.max(0,ix-70);k<=ix;k++){let z=xy(data.frames[k].pos[i]);k===Math.max(0,ix-70)?ctx.moveTo(...z):ctx.lineTo(...z)}ctx.stroke()}if(!f.active[i]){cross(p,'#8f9caa',5);ctx.fillStyle='#bcc9d7';;return}if(document.querySelector('#sensor').checked){let a=-f.heading[i],half=data.sensor_fov*Math.PI/360;ctx.fillStyle='rgba(255,255,255,.12)';ctx.strokeStyle='rgba(255,255,255,.45)';ctx.beginPath();ctx.moveTo(x,y);ctx.arc(x,y,data.sensor_range*77,a-half,a+half);ctx.closePath();ctx.fill();ctx.stroke()}if(f.goals[i]){ctx.strokeStyle=colors[i%5]+'90';ctx.setLineDash([3,5]);ctx.beginPath();ctx.moveTo(x,y);ctx.lineTo(...xy(f.goals[i]));ctx.stroke();ctx.setLineDash([])}ctx.fillStyle=colors[i%5];ctx.beginPath();ctx.arc(x,y,5,0,2*Math.PI);ctx.fill();ctx.font='bold 13px system-ui';});
document.querySelector('#clock').textContent=Math.floor(f.step/60)+':'+String(f.step%60).padStart(2,'0')+' / '+Math.floor(data.summary.steps/60)+':'+String(data.summary.steps%60).padStart(2,'0');document.querySelector('#scale').textContent='0 → '+f.maximum[layer].toFixed(3)+' 질량/km²';document.querySelector('#status').innerHTML='<b>seed '+data.seed+'</b><p>D'+data.scout+' 전역 탐색 전용<br>활성 드론 '+f.active.filter(Boolean).length+' / '+f.active.length+'<br>발견 '+f.tracks.length+'개 · 처리 완료 '+f.tracks.filter(t=>t.life===0).length+'개<br>보고 미발견 예상 '+f.reported_remaining.toFixed(2)+'<br>배경 총질량 '+f.background.toFixed(2)+'<br>점수 '+f.score.toFixed(1)+'</p><span class="small">드론은 도착 후 '+data.service_steps+'스텝 동안 처리하고 consume됩니다.</span>'}
document.querySelector('#events').innerHTML=data.events.filter(e=>e.kind!=='search').map(e=>'<div class="event" data-step="'+e.step+'"><b>'+Math.floor(e.step/60)+':'+String(e.step%60).padStart(2,'0')+'</b> '+e.message+'</div>').join('');document.querySelectorAll('.event').forEach(el=>el.onclick=()=>{slider.value=Math.max(0,data.frames.findIndex(f=>f.step>=+el.dataset.step));draw()});slider.oninput=draw;document.querySelectorAll('input[type=checkbox],#layer').forEach(e=>e.onchange=draw);document.querySelector('#play').onclick=()=>{if(timer){clearInterval(timer);timer=null;document.querySelector('#play').textContent='재생';return}document.querySelector('#play').textContent='정지';timer=setInterval(()=>{slider.value=Math.min(+slider.max,+slider.value+(+document.querySelector('#speed').value));draw();if(+slider.value===+slider.max){clearInterval(timer);timer=null;document.querySelector('#play').textContent='재생'}},120)};const seek=Number(new URLSearchParams(location.search).get('step')||0);if(seek>0)slider.value=Math.max(0,data.frames.findIndex(f=>f.step>=seek));draw();
</script>'''
if __name__=='__main__':main()
