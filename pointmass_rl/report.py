"""Standalone offline replay and scientific comparison plot."""
import json
from pathlib import Path

import numpy as np


HTML = r'''<!doctype html><html lang="ko"><meta charset="utf-8">
<title>Point-mass RL · 실험 리플레이</title>
<style>
body{font:15px system-ui;margin:24px auto;max-width:1280px;padding:0 16px;background:#101826;color:#e7edf7}h1{font-size:23px;margin:0 0 8px}h3{margin:8px 0}p{color:#b5c2d4;margin:8px 0}button,select,input{font:inherit}button,select{padding:7px 12px;border:1px solid #53627a;border-radius:6px;background:#202e43;color:white}.controls{display:flex;gap:8px;align-items:center;flex-wrap:wrap}.map-layout{display:grid;grid-template-columns:minmax(0,700px) 260px;gap:18px;align-items:start;margin-top:16px}canvas{background:#162238;border-radius:10px}#map{width:100%;max-width:700px}#overview{width:260px;height:260px}.overview-panel{width:260px}.row{display:flex;gap:28px;flex-wrap:wrap;margin-top:18px}.panel{flex:1;min-width:320px}table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:8px;border-bottom:1px solid #34405a}input[type=range]{width:260px}#stats{line-height:1.7}@media(max-width:1020px){.map-layout{grid-template-columns:1fr}.overview-panel{width:auto;display:flex;gap:14px;align-items:center;flex-wrap:wrap}}
</style>
<h1>Point-mass RL · Strike 표적 우선순위</h1>
<p>메인 맵은 선택 드론 주변 확대 화면입니다. 우측 작은 맵에서 전체 배치와 현재 확대 영역을 확인할 수 있습니다.</p>
<div class="controls"><select id="policy"></select><select id="focus"></select><button id="play">재생</button><input id="time" type="range" min="0" value="0"><span id="step"></span></div>
<div class="map-layout"><canvas id="map" width="700" height="700"></canvas><div class="overview-panel"><h3>전체 맵</h3><canvas id="overview" width="260" height="260"></canvas><p>흰 사각형은 메인 맵의 확대 범위입니다.</p></div></div>
<div class="row"><div class="panel"><h3>드론의 선택</h3><table><thead><tr><th>드론</th><th>행동</th></tr></thead><tbody id="agents"></tbody></table></div><div class="panel"><h3>누적 지표</h3><div id="stats"></div><p>정책은 모든 생존 표적의 현재 상태를 정확히 관측합니다.</p></div></div>
<script>
const data=__DATA__,colors=['#65d6bb','#7aa9ff','#ffb86c','#dc9df5','#ff7e89','#cbdc6b','#a6e3ff','#e3bfaa','#ef9ad1','#d1d7e0'];
const sel=document.getElementById('policy'),focus=document.getElementById('focus'),slider=document.getElementById('time'),ctx=document.getElementById('map').getContext('2d'),mini=document.getElementById('overview').getContext('2d');
for(const key of Object.keys(data.runs)){const option=document.createElement('option');option.value=key;option.textContent=key;sel.appendChild(option)}
function updateFocus(){const frame=data.runs[sel.value][0],old=focus.value;focus.innerHTML='';frame.agent_active.forEach((active,i)=>{if(active){const option=document.createElement('option');option.value=i;option.textContent=`D${i} 주변 확대`;focus.appendChild(option)}});if([...focus.options].some(x=>x.value===old))focus.value=old}
function grid(context,xy,bounds){context.strokeStyle='#344460';context.lineWidth=1;for(let value=0;value<=data.config.size;value+=10){if(value>=bounds[0]&&value<=bounds[1]){let a=xy([value,bounds[2]]),b=xy([value,bounds[3]]);context.beginPath();context.moveTo(...a);context.lineTo(...b);context.stroke()}if(value>=bounds[2]&&value<=bounds[3]){let a=xy([bounds[0],value]),b=xy([bounds[1],value]);context.beginPath();context.moveTo(...a);context.lineTo(...b);context.stroke()}}}
function draw(){const frames=data.runs[sel.value],index=+slider.value,f=frames[index],S=data.config.size,drone=+focus.value,span=Math.min(S,35),cx=f.drones[drone][0],cy=f.drones[drone][1],x0=Math.max(0,Math.min(S-span,cx-span/2)),y0=Math.max(0,Math.min(S-span,cy-span/2)),bounds=[x0,x0+span,y0,y0+span],scale=660/span,xy=p=>[20+(p[0]-x0)*scale,680-(p[1]-y0)*scale];slider.max=frames.length-1;ctx.clearRect(0,0,700,700);ctx.save();ctx.beginPath();ctx.rect(20,20,660,660);ctx.clip();grid(ctx,xy,bounds);
let [bx,by]=xy(f.base);ctx.fillStyle='#fff';ctx.fillRect(bx-5,by-5,10,10);
for(let i=0;i<f.drones.length;i++){if(!f.agent_active[i])continue;ctx.strokeStyle=colors[i];ctx.globalAlpha=.55;ctx.beginPath();for(let t=Math.max(0,index-40);t<=index;t++){let [x,y]=xy(frames[t].drones[i]);t===Math.max(0,index-40)?ctx.moveTo(x,y):ctx.lineTo(x,y)}ctx.stroke();ctx.globalAlpha=1}
f.targets.forEach((p,j)=>{if(!f.active[j])return;let [x,y]=xy(p),label=`T${j} type${f.target_type[j]} L${f.target_life[j]} P${f.target_score[j]}`;ctx.strokeStyle='#fbd38d';ctx.lineWidth=2;ctx.beginPath();ctx.arc(x,y,6,0,Math.PI*2);ctx.stroke();ctx.fillStyle=ctx.strokeStyle;let width=ctx.measureText(label).width,lx=Math.max(24,Math.min(676-width,x+8)),ly=Math.max(32,Math.min(676,y-5));ctx.fillText(label,lx,ly)});
f.drones.forEach((p,i)=>{if(!f.agent_active[i])return;let [x,y]=xy(p),[gx,gy]=xy(f.goal_positions[i]);ctx.strokeStyle=colors[i];ctx.globalAlpha=.3;ctx.beginPath();ctx.moveTo(x,y);ctx.lineTo(gx,gy);ctx.stroke();ctx.beginPath();ctx.arc(gx,gy,3,0,Math.PI*2);ctx.stroke();ctx.globalAlpha=1;ctx.fillStyle=colors[i];ctx.beginPath();ctx.arc(x,y,i===drone?7:5,0,Math.PI*2);ctx.fill();ctx.fillText('D'+i,x+8,y+15)});ctx.restore();ctx.strokeStyle='#53627a';ctx.strokeRect(20,20,660,660);
const mscale=230/S,mxy=p=>[15+p[0]*mscale,245-p[1]*mscale];mini.clearRect(0,0,260,260);grid(mini,mxy,[0,S,0,S]);[bx,by]=mxy(f.base);mini.fillStyle='#fff';mini.fillRect(bx-3,by-3,6,6);f.targets.forEach((p,j)=>{if(!f.active[j])return;let [x,y]=mxy(p);mini.fillStyle=f.target_formation[j]?'#ffb86c':'#9ddcc7';mini.beginPath();mini.arc(x,y,3,0,Math.PI*2);mini.fill()});f.drones.forEach((p,i)=>{if(!f.agent_active[i])return;let [x,y]=mxy(p);mini.fillStyle=colors[i];mini.beginPath();mini.arc(x,y,i===drone?4:2.5,0,Math.PI*2);mini.fill()});let topLeft=mxy([x0,y0+span]);mini.strokeStyle='#fff';mini.lineWidth=1.5;mini.strokeRect(topLeft[0],topLeft[1],span*mscale,span*mscale);
document.getElementById('step').textContent=`step ${f.step}/${data.config.horizon}`;document.getElementById('agents').innerHTML=f.drones.map((_,i)=>f.agent_active[i]?`<tr><td style="color:${colors[i]}">D${i}</td><td>표적 T${f.selected_target[i]} → (${f.goal_positions[i][0].toFixed(1)}, ${f.goal_positions[i][1].toFixed(1)})</td></tr>`:'').join('');let m=f.metrics;document.getElementById('stats').innerHTML=`점수 ${m.score}/${m.baseline_score}<br>임무 성공 ${m.mission_success?'예':'아니오'}<br>표적 섬멸률 ${(100*m.destroyed_fraction).toFixed(1)}%<br>우선순위 score AUC ${(100*m.score_auc).toFixed(1)}%<br>첫 득점 step ${m.first_score_step??'-'}<br>타격 성공 ${m.strike_hits}/${m.strike_attempts}`}
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
