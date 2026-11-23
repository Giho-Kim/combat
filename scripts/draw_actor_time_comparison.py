"""Compact vector diagram of the actor time-input alternatives."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'runs/target_allocation_attention_anatomy/SLIDES'
fig,ax=plt.subplots(figsize=(13,6.2))
fig.patch.set_facecolor('white');ax.set_xlim(0,13);ax.set_ylim(0,5.8);ax.axis('off')
ink='#17263b';blue='#2563eb';orange='#ea580c';gray='#64748b'
def box(x,y,w,h,text,fc='#eff6ff',ec='#bfdbfe',fs=11):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.02,rounding_size=0.09',linewidth=1,edgecolor=ec,facecolor=fc))
    if text.startswith(('Query inputs', 'Peer inputs')) and '\n' in text:
        name, note = text.split('\n', 1)
        ax.text(x+w/2,y+h*.66,name,ha='center',va='center',fontsize=12,color=ink)
        ax.text(x+w/2,y+h*.26,note,ha='center',va='center',fontsize=8,color=gray)
    else:
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fs,color=ink,linespacing=1.35)
def arrow(a,b,color=gray):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=12,lw=1.35,color=color,connectionstyle='arc3'))
def plus(x,y):
    ax.text(x,y,'+',ha='center',va='center',color=orange,fontsize=19,zorder=7)
for y,label,color in [(3.1,'Actor V1',blue),(.24,'Actor V2',orange)]:
    ax.add_patch(FancyBboxPatch((.18,y),12.62,2.30 if y>1 else 2.52,boxstyle='round,pad=0.0,rounding_size=.13',facecolor='#f8fafc',edgecolor='#e2e8f0'))
    ax.text(.25,y+(2.42 if y>1 else 2.64),label,fontsize=15,weight='bold',color=color)
def cells(x,y,colors,width=.13,height=.24,gap=.025):
    for i,color in enumerate(colors):
        ax.add_patch(Rectangle((x+i*(width+gap),y-height/2),width,height,
                               facecolor=color,edgecolor='white',linewidth=.7))
# Old path: time is one scalar among the query features.
box(.5,4.22,2.55,.65,'Query inputs\nOwn state · candidate target · prior intent',fs=9)
box(.5,3.30,2.55,.65,'Peer inputs\nPeer state · intent for this target',fs=9)
box(4.25,4.22,1.25,.65,'Query\nencoder',fs=10)
box(3.65,3.30,1.85,.65,'Peer encoder')
arrow((3.07,4.545),(3.22,4.545))
cells(3.25,4.545,[blue]*4+[orange])
arrow((4.04,4.545),(4.23,4.545))
ax.text(2.85,5.08,'Remaining time',fontsize=10,color=orange,ha='center',va='center')
cells(3.87,5.08,[orange])
arrow((3.935,4.94),(3.935,4.69),orange)
arrow((3.07,3.625),(3.63,3.625))
box(7.15,3.55,2.15,1.10,'Attention')
arrow((5.52,4.545),(7.13,4.32));arrow((5.52,3.625),(7.13,3.82))
box(10.10,3.68,2.15,.84,'Candidate score',fc='white',ec='#cbd5e1')
arrow((9.32,4.1),(10.08,4.1))
# New path: the scalar is removed from query concatenation; one vector
# is shared across all candidate queries and all peer representations.
box(.5,1.42,2.55,.60,'Query inputs\nOwn state · candidate target · prior intent',fs=9)
box(.5,.55,2.55,.60,'Peer inputs\nPeer state · intent for this target',fs=9)
box(3.65,1.42,1.85,.60,'Query encoder')
box(3.65,.55,1.85,.60,'Peer encoder')
arrow((3.07,1.72),(3.63,1.72));arrow((3.07,.85),(3.63,.85))
cells(2.4,2.37,[orange])
box(3.35,2.16,1.2,.42,'Time encoder',fc='#fff7ed',ec='#fed7aa',fs=10)
arrow((4.57,2.37),(4.91,2.37),orange)
cells(4.95,2.37,[orange]*4)

ax.text(.5,2.37,'Remaining time',fontsize=11,color=orange,va='center')
arrow((2.55,2.37),(3.33,2.37),orange)
# Shared time-vector bus.
ax.plot([5.60,6.12],[2.37,2.37],color=orange,lw=1.35)
arrow((6.12,2.37),(6.12,1.98),orange)
ax.add_patch(FancyBboxPatch((5.94,.59),.36,1.39,
    boxstyle='round,pad=0.0,rounding_size=.05',fc='#fff7ed',ec=orange,lw=1.5,zorder=5))
plus(6.12,1.72);plus(6.12,.85)
arrow((5.52,1.72),(5.94,1.72));arrow((5.52,.85),(5.94,.85))
box(7.15,.78,2.15,1.10,'Attention')
arrow((6.30,1.72),(7.13,1.55));arrow((6.30,.85),(7.13,1.05))
box(10.10,.91,2.15,.84,'Candidate score',fc='white',ec='#cbd5e1')
arrow((9.32,1.33),(10.08,1.33))
fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
for ext in ('pdf','svg','png'):
    fig.savefig(out/f'actor_time_comparison.{ext}',dpi=220,facecolor='white')
plt.close(fig)
