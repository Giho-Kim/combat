"""Standalone offline replay and scientific comparison plot."""
import json
from pathlib import Path

import numpy as np


HTML = r'''<!doctype html><html lang="ko"><meta charset="utf-8">
<title>Point-mass RL · 실험 리플레이</title>
<style>
body{font:15px system-ui;margin:24px auto;max-width:1280px;padding:0 16px;background:#101826;color:#e7edf7}h1{font-size:23px;margin:0 0 8px}h3{margin:8px 0}p{color:#b5c2d4;margin:8px 0}button,select,input{font:inherit}button,select{padding:7px 12px;border:1px solid #53627a;border-radius:6px;background:#202e43;color:white}.controls{display:flex;gap:8px;align-items:center;flex-wrap:wrap}.legend{display:flex;gap:10px 18px;align-items:center;flex-wrap:wrap;margin-top:12px;padding:9px 12px;background:#162238;border-radius:8px;color:#dce7f5}.legend-group{display:flex;gap:10px;align-items:center;flex-wrap:wrap}.legend-title{color:#93a4ba}.swatch{width:12px;height:12px;display:inline-block;border:1px solid #eef5ff;margin-right:5px;vertical-align:-1px}.swatch.friendly{background:#59c3ff;border-radius:50%}.swatch.f1{background:#ff5d73}.swatch.f2{background:#ffc857}.shape{font-size:18px;line-height:12px;margin-right:4px}.map-layout{display:grid;grid-template-columns:minmax(0,700px) 260px;gap:18px;align-items:start;margin-top:16px}canvas{background:#162238;border-radius:10px}#map{width:100%;max-width:700px}#overview{width:260px;height:260px}.overview-panel{width:260px}.row{display:flex;gap:28px;flex-wrap:wrap;margin-top:18px}.panel{flex:1;min-width:320px}table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:8px;border-bottom:1px solid #34405a}input[type=range]{width:260px}#stats{line-height:1.7}@media(max-width:1020px){.map-layout{grid-template-columns:1fr}.overview-panel{width:auto;display:flex;gap:14px;align-items:center;flex-wrap:wrap}}
</style>
<h1>Point-mass RL · Strike 표적 우선순위</h1>
<p>메인 맵은 선택 드론 주변 확대 화면입니다. 드론은 타격 준비 중 표적 주위를 선회한 뒤 타격하며, 이 선회 궤적은 리플레이 전용 시각 효과입니다. Random은 비교용 정책이라 접근 중에도 표적을 자주 바꿀 수 있습니다.</p>
<div class="controls"><select id="policy"></select><select id="focus"></select><button id="play">재생</button><input id="time" type="range" min="0" value="0"><span id="step"></span></div>
<div class="legend" aria-label="전장 기호 범례"><div class="legend-group"><span class="legend-title">소속 색상</span><span><i class="swatch friendly"></i>아군 드론</span><span><i class="swatch f1"></i>적 편제 1</span><span><i class="swatch f2"></i>적 편제 2</span></div><div class="legend-group"><span class="legend-title">표적 타입 도형</span><span><i class="shape">△</i>Type 1</span><span><i class="shape">□</i>Type 2</span><span><i class="shape">◇</i>Type 3</span></div></div>
<div class="map-layout"><canvas id="map" width="700" height="700"></canvas><div class="overview-panel"><h3>전체 맵</h3><canvas id="overview" width="260" height="260"></canvas><p>흰 사각형은 메인 맵의 확대 범위입니다.</p></div></div>
<div class="row"><div class="panel"><h3>드론의 선택</h3><table><thead><tr><th>드론</th><th>행동</th></tr></thead><tbody id="agents"></tbody></table></div><div class="panel"><h3>누적 지표</h3><div id="stats"></div><p>정책은 모든 생존 표적의 현재 상태를 정확히 관측합니다.</p></div></div>
<script>
const data=__DATA__,friendlyColor='#59c3ff',formationColors=['#ff5d73','#ffc857'],loiterRadius=.8;
const sel=document.getElementById('policy'),focus=document.getElementById('focus'),slider=document.getElementById('time'),ctx=document.getElementById('map').getContext('2d'),mini=document.getElementById('overview').getContext('2d');
const policyLabels={mappo:'MAPPO (학습 정책)',heuristic:'Heuristic (안정적 데모)',random:'Random (비교 기준)'};
for(const key of Object.keys(data.runs)){const option=document.createElement('option');option.value=key;option.textContent=policyLabels[key]??key;sel.appendChild(option)}
sel.value=['mappo','heuristic','random'].find(key=>data.runs[key])??Object.keys(data.runs)[0];
function updateFocus(){const frame=data.runs[sel.value][0],old=focus.value;focus.innerHTML='';frame.agent_active.forEach((active,i)=>{if(active){const option=document.createElement('option');option.value=i;option.textContent=`D${i} 주변 확대`;focus.appendChild(option)}});if([...focus.options].some(x=>x.value===old))focus.value=old}
function grid(context,xy,bounds){context.strokeStyle='#344460';context.lineWidth=1;for(let value=0;value<=data.config.size;value+=10){if(value>=bounds[0]&&value<=bounds[1]){let a=xy([value,bounds[2]]),b=xy([value,bounds[3]]);context.beginPath();context.moveTo(...a);context.lineTo(...b);context.stroke()}if(value>=bounds[2]&&value<=bounds[3]){let a=xy([bounds[0],value]),b=xy([bounds[1],value]);context.beginPath();context.moveTo(...a);context.lineTo(...b);context.stroke()}}}
function targetPath(context,x,y,type,r){context.beginPath();if(type===1){context.moveTo(x,y-r);context.lineTo(x+r*.9,y+r*.75);context.lineTo(x-r*.9,y+r*.75);context.closePath()}else if(type===2){context.rect(x-r,y-r,r*2,r*2)}else{context.moveTo(x,y-r);context.lineTo(x+r,y);context.lineTo(x,y+r);context.lineTo(x-r,y);context.closePath()}}
function drawTarget(context,xy,frame,j,r){const [x,y]=xy(frame.targets[j]),formation=frame.target_formation[j]??0;color=formationColors[formation];targetPath(context,x,y,frame.target_type[j],r);context.fillStyle=color;context.strokeStyle='#f4f8ff';context.lineWidth=1.5;context.fill();context.stroke();return{x,y,color,formation}}
function drawDrone(context,xy,position,r,focused=false){const [x,y]=xy(position);context.save();context.strokeStyle=focused?'#ffffff':'#082438';context.fillStyle=friendlyColor;context.lineWidth=focused?2.5:1.5;context.beginPath();context.arc(x,y,r,0,Math.PI*2);context.fill();context.stroke();context.beginPath();context.moveTo(x-r*1.45,y);context.lineTo(x+r*1.45,y);context.moveTo(x,y-r*1.45);context.lineTo(x,y+r*1.45);context.strokeStyle=friendlyColor;context.lineWidth=2;context.stroke();context.restore();return{x,y}}
function loiter(frame,i){const j=frame.selected_target[i];if(j<0||!frame.agent_active[i]||!frame.strike_participants?.[j]?.[i])return null;const members=frame.strike_participants[j].map((on,k)=>on?k:-1).filter(k=>k>=0),rank=members.indexOf(i),steps=data.config.strike_steps_per_life,phase=(frame.strike_progress[j]-1)/Math.max(1,steps-2),angle=2*Math.PI*(phase+rank/members.length);return{target:j,progress:frame.strike_progress[j],position:[frame.targets[j][0]+loiterRadius*Math.cos(angle),frame.targets[j][1]+loiterRadius*Math.sin(angle)]}}
function dronePosition(frame,i){return loiter(frame,i)?.position??frame.drones[i]}
function recentStrike(frames,index,i){for(let age=0;age<3&&index-age>0;age++){const frame=frames[index-age];if(frame.metrics.agent_terminated[i]){const previous=frames[index-age-1],target=previous.selected_target[i];if(target>=0)return{position:frame.targets[target],age}}}return null}
function strikeFlash(context,xy,event){if(!event)return;const [x,y]=xy(event.position),radius=8+event.age*7;context.save();context.strokeStyle='#ff5f56';context.fillStyle='#ffd166';context.lineWidth=3;context.globalAlpha=1-event.age*.28;context.beginPath();context.arc(x,y,radius,0,Math.PI*2);context.stroke();for(let a=0;a<8;a++){const angle=a*Math.PI/4;context.beginPath();context.moveTo(x+Math.cos(angle)*(radius+2),y+Math.sin(angle)*(radius+2));context.lineTo(x+Math.cos(angle)*(radius+10),y+Math.sin(angle)*(radius+10));context.stroke()}context.beginPath();context.arc(x,y,4,0,Math.PI*2);context.fill();context.restore()}
function draw(){const frames=data.runs[sel.value],index=+slider.value,f=frames[index],S=data.config.size,drone=+focus.value,span=Math.min(S,35),focusPosition=dronePosition(f,drone),cx=focusPosition[0],cy=focusPosition[1],x0=Math.max(0,Math.min(S-span,cx-span/2)),y0=Math.max(0,Math.min(S-span,cy-span/2)),bounds=[x0,x0+span,y0,y0+span],scale=660/span,xy=p=>[20+(p[0]-x0)*scale,680-(p[1]-y0)*scale];slider.max=frames.length-1;ctx.clearRect(0,0,700,700);ctx.save();ctx.beginPath();ctx.rect(20,20,660,660);ctx.clip();grid(ctx,xy,bounds);
let [bx,by]=xy(f.base);ctx.fillStyle='#fff';ctx.fillRect(bx-5,by-5,10,10);
for(let i=0;i<f.drones.length;i++){if(!f.agent_active[i])continue;ctx.strokeStyle=friendlyColor;ctx.globalAlpha=.42;ctx.beginPath();for(let t=Math.max(0,index-40);t<=index;t++){let [x,y]=xy(dronePosition(frames[t],i));t===Math.max(0,index-40)?ctx.moveTo(x,y):ctx.lineTo(x,y)}ctx.stroke();ctx.globalAlpha=1}
f.targets.forEach((p,j)=>{if(!f.active[j])return;const mark=drawTarget(ctx,xy,f,j,7),label=`F${mark.formation+1} · T${j} · Type ${f.target_type[j]} · L${f.target_life[j]} · P${f.target_score[j]}`;ctx.fillStyle=mark.color;let width=ctx.measureText(label).width,lx=Math.max(24,Math.min(676-width,mark.x+10)),ly=Math.max(32,Math.min(676,mark.y-7));ctx.fillText(label,lx,ly)});
f.drones.forEach((p,i)=>{if(!f.agent_active[i])return;const orbit=loiter(f,i);let [x,y]=xy(dronePosition(f,i)),[gx,gy]=xy(f.goal_positions[i]);ctx.strokeStyle=friendlyColor;ctx.globalAlpha=.34;if(orbit){let [tx,ty]=xy(f.targets[orbit.target]);ctx.setLineDash([5,5]);ctx.beginPath();ctx.arc(tx,ty,loiterRadius*scale,0,Math.PI*2);ctx.stroke();ctx.setLineDash([])}else{ctx.beginPath();ctx.moveTo(x,y);ctx.lineTo(gx,gy);ctx.stroke();ctx.beginPath();ctx.arc(gx,gy,3,0,Math.PI*2);ctx.stroke()}ctx.globalAlpha=1;drawDrone(ctx,xy,dronePosition(f,i),i===drone?7:5,i===drone);ctx.fillStyle=friendlyColor;ctx.fillText('D'+i,x+9,y+16)});for(let i=0;i<f.drones.length;i++)strikeFlash(ctx,xy,recentStrike(frames,index,i));ctx.restore();ctx.strokeStyle='#53627a';ctx.strokeRect(20,20,660,660);
const mscale=230/S,mxy=p=>[15+p[0]*mscale,245-p[1]*mscale];mini.clearRect(0,0,260,260);grid(mini,mxy,[0,S,0,S]);[bx,by]=mxy(f.base);mini.fillStyle='#fff';mini.fillRect(bx-3,by-3,6,6);f.targets.forEach((p,j)=>{if(f.active[j])drawTarget(mini,mxy,f,j,4)});f.drones.forEach((p,i)=>{if(f.agent_active[i])drawDrone(mini,mxy,dronePosition(f,i),i===drone?4:2.7,i===drone)});for(let i=0;i<f.drones.length;i++)strikeFlash(mini,mxy,recentStrike(frames,index,i));let topLeft=mxy([x0,y0+span]);mini.strokeStyle='#fff';mini.lineWidth=1.5;mini.strokeRect(topLeft[0],topLeft[1],span*mscale,span*mscale);
document.getElementById('step').textContent=`step ${f.step}/${data.config.horizon}`;document.getElementById('agents').innerHTML=f.drones.map((_,i)=>{if(!f.agent_active[i])return'';const orbit=loiter(f,i);return`<tr><td style="color:${friendlyColor}">D${i}</td><td>${orbit?`T${orbit.target} 주위 선회 ${orbit.progress}/${data.config.strike_steps_per_life} → 타격`:`표적 T${f.selected_target[i]} 접근 → (${f.goal_positions[i][0].toFixed(1)}, ${f.goal_positions[i][1].toFixed(1)})`}</td></tr>`}).join('');let m=f.metrics;document.getElementById('stats').innerHTML=`점수 ${m.score}/${m.baseline_score}<br>임무 성공 ${m.mission_success?'예':'아니오'}<br>표적 섬멸률 ${(100*m.destroyed_fraction).toFixed(1)}%<br>우선순위 score AUC ${(100*m.score_auc).toFixed(1)}%<br>첫 득점 step ${m.first_score_step??'-'}<br>타격 성공 ${m.strike_hits}/${m.strike_attempts}`}
let playing=false;sel.onchange=()=>{slider.value=0;updateFocus();draw()};focus.onchange=draw;slider.oninput=draw;document.getElementById('play').onclick=()=>{playing=!playing;document.getElementById('play').textContent=playing?'일시정지':'재생'};setInterval(()=>{if(playing){slider.value=(+slider.value+1)%(+slider.max+1);draw()}},100);updateFocus();draw();
</script></html>'''


def replay(path, config, runs):
    from dataclasses import asdict
    data = json.dumps(dict(config=asdict(config), runs=runs), separators=(',', ':'))
    Path(path).write_text(HTML.replace('__DATA__', data), encoding='utf-8')


def plot(path, summaries):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    metrics = [('mission_success', 'Mission success'),
               ('destroyed_fraction', 'Targets destroyed'),
               ('score_auc', 'Early-value score AUC')]
    fig, axes = plt.subplots(1, len(metrics), figsize=(11, 3.8), layout='constrained')
    for ax, (key, label) in zip(axes, metrics):
        labels = list(summaries)
        means = [summaries[p][key]['iqm'] for p in labels]
        ax.bar(labels, means, color=['#7c899e', '#55a798', '#647ddd'][:len(labels)])
        ax.set_title(label, fontsize=10)
        ax.set_ylim(0, 1.12)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle('Held-out scenario seeds · interquartile mean', fontsize=12)
    fig.savefig(path, dpi=150)
    plt.close(fig)
