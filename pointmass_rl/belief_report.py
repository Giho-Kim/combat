"""Standalone replay of the public belief state in the partially observed game."""
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np


def frame(world, observation, chosen=None):
    """Record what the policy saw; truth is a separate, opt-in audit overlay."""
    belief = world.belief
    hidden = world._world
    cells = belief.cells
    components = []
    for slot in np.flatnonzero(belief.valid):
        record = belief.records[slot]
        components.append(dict(slot=int(slot), kind='region' if slot < cells else 'track',
                               mean=(record[:2] * world.c.size).tolist(),
                               goal=belief.goals[slot].tolist(),
                               count=float(record[9]), uncertainty=float(record[2] + record[4]),
                               type_prob=record[5:8].tolist(), life=float(record[8] * 2)))
    truth = [dict(position=hidden.targets[j].tolist(), formation=int(hidden.target_formation[j])+1,
                  type=int(hidden.target_type[j]),
                  life=int(hidden.target_life[j]), active=bool(hidden.active[j]))
             for j in np.flatnonzero(hidden.target_exists)]
    metrics = world.metrics()
    expected = ((world.c.n_targets + world.c.min_targets)/2
                if world.c.randomize_counts else world.c.n_targets)
    initial_weight = expected/belief.weights.size
    strength = np.clip(belief.weights/initial_weight, 0, 1)
    # Fixed initial scale: clearing one point never brightens another point.
    # Two hex digits per support point keep replay files compact.
    heat = np.where(strength > 0, np.maximum(1, np.rint(strength*255)), 0).astype(np.uint8)
    return dict(step=int(hidden.t), drones=hidden.pos.tolist(),
                belief_heat=heat.tobytes().hex(), headings=hidden.heading.tolist(),
                goals=hidden.last_goal.tolist(),
                sweeping=[bool(nav and nav['sweeping'] and belief.valid[nav['component']])
                          for nav in world.navigation],
                active=hidden.agent_active.tolist(), components=components,
                chosen=(None if chosen is None else np.asarray(chosen, dtype=int).tolist()),
                truth=truth, formation_centers=hidden.formation_centers.tolist(),
                score=metrics['score'], threshold=metrics['success_threshold'],
                team_return=metrics['team_return'], discovery_fraction=metrics['discovery_fraction'],
                unseen_expected_count=metrics['unseen_expected_count'])


HTML = r'''<!doctype html><html lang="ko"><meta charset="utf-8">
<title>Belief mode · 평가 리플레이</title>
<style>
body{font:15px system-ui;margin:20px auto;max-width:1040px;padding:0 16px;background:#101826;color:#e7edf7}
h1{font-size:23px;margin:0 0 8px}p{color:#b5c2d4;line-height:1.5}button,select,input{font:inherit}
button,select{padding:7px 11px;border:1px solid #53627a;border-radius:6px;background:#202e43;color:white}
.controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:16px 0}
input[type=range]{width:min(320px,55vw)}label{white-space:nowrap}canvas{width:min(800px,100%);background:#162238;border-radius:10px}
#stats{line-height:1.7;margin:12px 0 24px;white-space:pre-wrap}
</style>
<h1>Belief mode · 평가 리플레이</h1>
<p id="motion"></p>
<p>파란 작은 칸이 진할수록 미발견 belief가 많이 남아 있습니다. 관측으로 belief가 줄면 옅어지고, 0이 되면 사라집니다. 굵은 선은 16개 탐색 구역의 경계, 부채꼴은 센서 범위, 노란 원은 발견된 개체입니다.</p>
<div class="controls"><select id="policy"></select><button id="play">재생</button><input id="time" type="range" min="0" value="0"><span id="step"></span><label><input id="truth" type="checkbox"> 실제 위치 보기</label></div>
<div class="controls"><label><input id="belief" type="checkbox" checked> 미발견 belief</label><label><input id="sensor" type="checkbox" checked> 센서 범위</label><span id="belief-status"></span></div>
<canvas id="map" width="800" height="800"></canvas><div id="stats"></div>
<script>
const data=__DATA__,sel=document.getElementById('policy'),slider=document.getElementById('time'),
      ctx=document.getElementById('map').getContext('2d'),truthToggle=document.getElementById('truth'),
      beliefToggle=document.getElementById('belief'),sensorToggle=document.getElementById('sensor');
const searchMargin=data.config.belief_search_margin??0,searchSize=data.config.size-2*searchMargin;
document.getElementById('motion').textContent=`Dubins 이동 · lawnmower ${(data.config.belief_lawnmower??true)?'ON':'OFF'} · 맵 ${data.config.size/10}×${data.config.size/10}km · 탐색 영역 ${searchSize/10}×${searchSize/10}km`;
truthToggle.checked=new URLSearchParams(location.search).has('truth') ||
  Object.values(data.runs).every(frames=>frames.length===1);
for(const name of Object.keys(data.runs)){const option=document.createElement('option');option.value=name;option.textContent=name;sel.appendChild(option)}
sel.value=data.runs.none?'none':Object.keys(data.runs)[0];
const margin=30,span=740,scale=span/data.config.size;
function xy(point){return[margin+point[0]*scale,margin+(data.config.size-point[1])*scale]}
function draw(){const frames=data.runs[sel.value],index=Number(slider.value),f=frames[index];
  slider.max=frames.length-1;ctx.clearRect(0,0,800,800);
  const grid=data.config.belief_grid,cell=searchSize/grid;
  ctx.strokeStyle='#35445d';ctx.lineWidth=1;
  ctx.strokeRect(margin,margin,span,span);
  if(beliefToggle.checked){
    if(f.belief_heat){const sub=data.config.belief_subcells,small=cell/sub;
      for(let j=0;j<grid*grid;j++)for(let k=0;k<sub*sub;k++){
        const offset=2*(j*sub*sub+k),weight=parseInt(f.belief_heat.slice(offset,offset+2),16)/255;
        if(!weight)continue;
        const col=j%grid,row=Math.floor(j/grid),sx=k%sub,sy=Math.floor(k/sub),
              a=xy([searchMargin+col*cell+sx*small,searchMargin+row*cell+(sy+1)*small]);
        ctx.fillStyle=`rgba(55,155,232,${.72*weight})`;
        ctx.fillRect(a[0]+.4,a[1]+.4,small*scale-.8,small*scale-.8)}
    }else{for(const item of f.components){if(item.kind!=='region')continue;
      const col=item.slot%grid,row=Math.floor(item.slot/grid),a=xy([searchMargin+col*cell,searchMargin+(row+1)*cell]);
      ctx.fillStyle=`rgba(55,155,232,${Math.min(.55,.06+.4*item.count)})`;
      ctx.fillRect(a[0],a[1],cell*scale,cell*scale)}}
  }
  ctx.strokeStyle='#7186a2';ctx.lineWidth=1.5;
  for(let i=0;i<=grid;i++){const p=searchMargin+i*cell,a=xy([p,searchMargin]),b=xy([p,data.config.size-searchMargin]),c=xy([searchMargin,p]),d=xy([data.config.size-searchMargin,p]);
    ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.moveTo(...c);ctx.lineTo(...d);ctx.stroke()}
  if(sensorToggle.checked && f.headings){
    const half=data.config.sensor_fov_deg*Math.PI/360,radius=data.config.sensor_range*scale;
    for(let i=0;i<f.drones.length;i++){if(!f.active[i])continue;
      const [x,y]=xy(f.drones[i]),angle=-f.headings[i];
      ctx.fillStyle='rgba(100,255,195,.12)';ctx.strokeStyle='rgba(100,255,195,.6)';ctx.lineWidth=1;
      ctx.beginPath();ctx.moveTo(x,y);ctx.arc(x,y,radius,angle-half,angle+half);ctx.closePath();ctx.fill();ctx.stroke()}}
  const initial=data.config.randomize_counts?(data.config.n_targets+data.config.min_targets)/2:data.config.n_targets;
  document.getElementById('belief-status').textContent=f.belief_heat?
    `미발견 belief 잔량 ${(100*f.unseen_expected_count/initial).toFixed(1)}% (초기 대비)`:
    '구역별 합계 표시 · 세부 belief는 새 평가부터 기록됩니다';
  for(const item of f.components){if(item.kind!=='track')continue;
    const [x,y]=xy(item.mean);ctx.fillStyle='#ffd166';ctx.strokeStyle='#fff';ctx.lineWidth=1.5;
    ctx.beginPath();ctx.arc(x,y,6,0,2*Math.PI);ctx.fill();ctx.stroke();
    const type=item.type_prob.indexOf(Math.max(...item.type_prob))+1;
    ctx.fillStyle='#ffd166';ctx.fillText(`Type ${type} · L${item.life.toFixed(0)}`,x+9,y-7)}
  if(truthToggle.checked){const formationColors=['#ff6778','#ffb347'];
    for(let k=0;k<f.formation_centers.length;k++){const [x,y]=xy(f.formation_centers[k]);
      ctx.strokeStyle=formationColors[k%formationColors.length];ctx.lineWidth=1.5;ctx.setLineDash([6,4]);
      ctx.beginPath();ctx.arc(x,y,data.config.target_formation_radius*scale,0,2*Math.PI);ctx.stroke();ctx.setLineDash([]);
      ctx.fillStyle=ctx.strokeStyle;ctx.font='bold 16px system-ui';ctx.fillText(`편제 ${k+1}`,x+8,y-11)}
    for(let j=0;j<f.truth.length;j++){const item=f.truth[j];if(!item.active)continue;
      const [x,y]=xy(item.position);ctx.strokeStyle=formationColors[(item.formation-1)%formationColors.length];ctx.lineWidth=3;
      ctx.beginPath();ctx.moveTo(x-7,y-7);ctx.lineTo(x+7,y+7);ctx.moveTo(x+7,y-7);ctx.lineTo(x-7,y+7);ctx.stroke();
      ctx.fillStyle=ctx.strokeStyle;ctx.font='bold 12px system-ui';ctx.fillText(`F${item.formation}·${j}`,x+9,y+5)}}
  for(let i=0;i<f.drones.length;i++){const trail=frames.slice(Math.max(0,index-30),index+1);
    ctx.strokeStyle='rgba(105,206,255,.35)';ctx.lineWidth=1.5;ctx.beginPath();
    trail.forEach((past,j)=>{const [tx,ty]=xy(past.drones[i]);j?ctx.lineTo(tx,ty):ctx.moveTo(tx,ty)});ctx.stroke();
    if(!f.active[i])continue;
    const [x,y]=xy(f.drones[i]);ctx.fillStyle='#62caff';ctx.beginPath();ctx.arc(x,y,5,0,2*Math.PI);ctx.fill();
    ctx.fillText(`D${i}${f.sweeping?.[i]?' · 탐색 중':''}`,x+8,y+12);
    const slot=f.chosen?.[i],item=f.components.find(component=>component.slot===slot);
    if(item){const [gx,gy]=xy(f.goals?.[i]??item.goal);ctx.strokeStyle='rgba(105,206,255,.45)';ctx.beginPath();ctx.moveTo(x,y);ctx.lineTo(gx,gy);ctx.stroke();
      ctx.beginPath();ctx.arc(gx,gy,2.5,0,2*Math.PI);ctx.stroke()}}
  document.getElementById('step').textContent=`step ${f.step}/${data.config.horizon}`;
  const choices=(f.chosen??[]).map((slot,i)=>`D${i}: ${slot<data.config.belief_grid**2?'구역':'발견 개체'} ${slot}`).join(' · ');
  document.getElementById('stats').textContent=`점수 ${f.score.toFixed(2)} / 기준 ${f.threshold.toFixed(2)}  ·  return ${f.team_return.toFixed(2)}\n발견률 ${(100*f.discovery_fraction).toFixed(0)}%  ·  미발견 예상 개체 수 ${f.unseen_expected_count.toFixed(2)}\n${choices}`;
}
let playing=false;sel.onchange=()=>{slider.value=0;draw()};slider.oninput=draw;truthToggle.onchange=draw;
beliefToggle.onchange=draw;sensorToggle.onchange=draw;
document.getElementById('play').onclick=()=>{playing=!playing;document.getElementById('play').textContent=playing?'일시정지':'재생'};
setInterval(()=>{if(playing){slider.value=(Number(slider.value)+1)%(Number(slider.max)+1);draw()}},100);draw();
</script></html>'''


def replay(path, config, runs, seed):
    payload = json.dumps(dict(config=asdict(config), seed=seed, runs=runs), separators=(',', ':'))
    Path(path).write_text(HTML.replace('__DATA__', payload.replace('<', '\\u003c')), encoding='utf-8')
