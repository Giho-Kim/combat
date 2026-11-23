"""Current belief environment and arc-demo spawning on one map figure."""
import sys,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Circle
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from pointmass_rl.env import Config
from pointmass_rl.belief_env import BeliefWorld
from scripts.compare_belief_heuristics import ArcFormationDemoWorld
out=ROOT/'runs/belief_spawn_layout';out.mkdir(exist_ok=True);c=Config.load(ROOT/'configs/belief.json')
fig,axes=plt.subplots(1,2,figsize=(11.5,6.2));records=[]
for ax,arc,title in zip(axes,[False,True],['Belief training environment','Recent heuristic demo: random arc']):
 world=BeliefWorld(c)
 if arc:world._world=ArcFormationDemoWorld(world._world.c)
 world.reset(10000);w=world._world;f1=w.target_formation==0;f2=w.target_formation==1;p=w.targets/10;center=p[f1].mean(0)
 ax.set_facecolor('#f0f2f5');ax.add_patch(Rectangle((1,1),8,8,facecolor='white',edgecolor='#89939e',ls='--',zorder=0))
 if arc:
  ax.add_patch(Circle(center,5,fill=False,color='#c47536',ls='--',lw=1.4,zorder=1))
  other=p[f2].mean(0);ax.plot([center[0],other[0]],[center[1],other[1]],color='#c47536',lw=1,zorder=1)
  mid=(center+other)/2;ax.text(mid[0]+.15,mid[1],'5 km',color='#9a561f',fontsize=11)
 ax.scatter(w.pos[:,0]/10,w.pos[:,1]/10,marker='^',s=28,c='#3c9b78',label='Drone start',zorder=3)
 ax.scatter(p[f1,0],p[f1,1],s=65,c='#2878b8',label='Formation 1: actual',edgecolors='white',zorder=4)
 ax.scatter(p[f2,0],p[f2,1],s=65,c='#e38428',label='Formation 2: actual',edgecolors='white',zorder=4)
 reports=np.array([r['position'] for r in w.initial_intel['objects']])/10
 ax.scatter(reports[:,0],reports[:,1],marker='+',s=120,c='#9954b2',linewidths=2,label='Initial reports (F1)',zorder=5)
 for actual,reported in zip(p[f1],reports):ax.plot([actual[0],reported[0]],[actual[1],reported[1]],color='#aeb7c2',lw=.8,zorder=2)
 ax.set(xlim=(0,10),ylim=(0,10),xticks=range(0,11,2),yticks=range(0,11,2),xlabel='x (km)',ylabel='y (km)',title=title);ax.set_aspect('equal');ax.grid(alpha=.14)
 records.append(dict(mode='arc demo' if arc else 'belief environment',seed=10000,targets_km=p.tolist(),formation=w.target_formation.tolist(),reports_km=reports.tolist()))
handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=4,frameon=False,bbox_to_anchor=(.5,.015),fontsize=10)
fig.subplots_adjust(bottom=.19,top=.88,wspace=.2);fig.suptitle('10 × 10 km map · spawns inside 8 × 8 km · seed 10000',fontsize=13,y=.98)
for ext in ['png','pdf']:fig.savefig(out/f'spawn_layout.{ext}',dpi=180,bbox_inches='tight')
(out/'layout.json').write_text(json.dumps(records,indent=2));print(out/'spawn_layout.png')
