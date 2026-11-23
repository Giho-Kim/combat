"""Two compact schematics of deadline-related allocation failures."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'runs/target_allocation_attention_anatomy/SLIDES';out.mkdir(exist_ok=True)
fig,axes=plt.subplots(1,2,figsize=(12.8,5.6))
ink='#17263b';red='#dc2626';green='#059669';gray='#64748b';blue='#2563eb'
def arrow(ax,a,b,color,style='-',rad=0):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=17,lw=2.3,
        linestyle=style,color=color,connectionstyle=f'arc3,rad={rad}',shrinkA=23,shrinkB=23,zorder=2))
def node(ax,x,y,label,drone=False):
    ax.scatter([x],[y],s=700 if drone else 620,marker='o' if drone else 's',
               color='#eff6ff' if drone else '#fff7ed',edgecolor=blue if drone else '#d97706',lw=1.8,zorder=4)
    ax.text(x,y,label,ha='center',va='center',fontsize=12,weight='bold',color=ink,zorder=5)
def text(ax,x,y,label,color=ink,fs=11,background=True):
    ax.text(x,y,label,ha='center',va='center',fontsize=fs,color=color,
        bbox=dict(fc='white',ec='none',pad=2) if background else None,zorder=6)
for ax in axes:
    ax.set_xlim(0,10);ax.set_ylim(0,7.5);ax.axis('off')
    ax.add_patch(FancyBboxPatch((.08,.28),9.84,6.75,boxstyle='round,pad=0,rounding_size=.2',fc='#f8fafc',ec='#e2e8f0'))
a=axes[0]
a.set_title('1. Deadline feasibility',fontsize=17,weight='bold',loc='left',pad=12,color=ink)
node(a,1.6,3.65,'D1',True);node(a,8.1,5.1,'T3');node(a,6.1,1.85,'T7')
arrow(a,(1.6,3.65),(8.1,5.1),red)
arrow(a,(1.6,3.65),(6.1,1.85),green,'--')
text(a,5.2,5.15,'104 steps > 100',red,13)
text(a,4.0,1.88,'48 steps ≤ 100',green,13)
text(a,5,.85,'An unreachable target stays selected',ink,12,background=False)
b=axes[1]
b.set_title('2. Deadline-aware coordination',fontsize=17,weight='bold',loc='left',pad=12,color=ink)
node(b,1.55,5.05,'D1',True);node(b,1.55,2.05,'D2',True)
node(b,8.25,5.05,'T5');node(b,8.25,2.05,'T6')
arrow(b,(1.55,2.05),(8.25,5.05),red)
arrow(b,(1.55,5.05),(8.25,5.05),green,'--')
arrow(b,(1.55,2.05),(8.25,2.05),green,'--')
arrow(b,(1.55,5.05),(8.25,2.05),'#94a3b8',':',rad=.10)
text(b,5,5.38,'91 steps',green,12)
text(b,5,1.63,'99 steps',green,12)
text(b,4.85,3.85,'88 steps',red,12)
text(b,5.6,2.95,'102 steps > 100',gray,11)
text(b,5,.85,'Individual choices miss a feasible team assignment',ink,12,background=False)
fig.legend(handles=[Line2D([0],[0],color=red,lw=2.3,label='Actual choice'),
    Line2D([0],[0],color=green,lw=2.3,ls='--',label='Better assignment'),
    Line2D([0],[0],color='#94a3b8',lw=2,ls=':',label='Cannot finish in time')],
    loc='lower center',ncol=3,frameon=False,fontsize=11,bbox_to_anchor=(.5,.01))
fig.subplots_adjust(left=.03,right=.98,top=.87,bottom=.10,wspace=.10)
for ext in ('pdf','svg','png'):fig.savefig(out/f'failure_modes.{ext}',dpi=220,facecolor='white')
plt.close(fig)
