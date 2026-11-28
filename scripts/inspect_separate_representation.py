"""Frozen no-resolver anatomy: input interventions, pair probes and representations."""
import argparse, csv, hashlib, json, shutil, sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from pointmass_rl.env import Config,World,evaluation_task_for_episode
from pointmass_rl.final_mission_ppo import (model_from_checkpoint,checkpoint_payload,DecisionSchedule,
    _model_observation,scheduled_action,_evaluation_threshold)

def write_csv(path, rows):
    if not rows:return
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def parts(state,n,t):
    a=1+7*n;b=a+6*t;c=b+n*t
    return state[...,1:a].reshape(*state.shape[:-1],n,7),state[...,a:b].reshape(*state.shape[:-1],t,6),state[...,b:c].reshape(*state.shape[:-1],t,n).transpose(-1,-2),state[...,c:].reshape(*state.shape[:-1],t,n).transpose(-1,-2)

def offdiag_cos(x):
    if len(x)<2:return float('nan')
    z=torch.nn.functional.normalize(x,dim=-1);m=z@z.T
    return float(m[~torch.eye(len(x),dtype=torch.bool)].mean())

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--checkpoint',default='checkpoint_3000000.pt');p.add_argument('--episodes',type=int,default=30);p.add_argument('--seed',type=int,default=20000);a=p.parse_args()
    run=ROOT/a.run;out=run/'representation_anatomy'/Path(a.checkpoint).stem;out.mkdir(parents=True,exist_ok=True)
    for name in (a.checkpoint,'config.json'):shutil.copy2(run/name,out/name)
    model=model_from_checkpoint(out/a.checkpoint,Config.load(out/'config.json'));model.eval();torch.set_num_threads(1)
    c=Config.load(out/'config.json');n,t=model.n_agents,model.n_targets
    def encode(s,critic=False):return model.q_body.encode(s,n,t,**({} if critic else {'encoder':model.actor_encoder}))
    def logits(s):return model.logits_from_encoded(*encode(s)[:2])
    contexts=[];effects=[];probes=[];reps=[];rollouts=[];saved={};rng=np.random.default_rng(7)
    for ep in range(a.episodes):
        world=World(c);world.reset(a.seed+ep,success_threshold=_evaluation_threshold(c,ep),evaluation_task=evaluation_task_for_episode(c,ep));schedule=DecisionSchedule(c)
        while not world.done:
            if world.t in (0,1,5,10,20,40,60,80):
                obs=torch.as_tensor(_model_observation(model,world,schedule),dtype=torch.float32)
                state=model.state_from_observation(obs[0],n,t);bat=state[None]
                available=model.target_mask(obs);mask,decision=schedule.mask(world,available.numpy());mask=torch.as_tensor(mask)
                eligible=decision & (mask.sum(-1).numpy()>1);idx=np.flatnonzero(eligible)
                base=logits(bat)[0];torch.testing.assert_close(base,model.actor_logits(obs),atol=2e-4,rtol=1e-4)
                prob=torch.softmax(base.masked_fill(~mask,-torch.inf),-1);choice=prob.argmax(-1)
                dr,ta,pr,prev=parts(state,n,t);alive=dr[:,2]>.5;exists=ta[:,2:5].sum(-1)>.5;valid_targets=torch.where(exists)[0]
                stage='initial' if world.t==0 else 'in_progress';context=len(contexts)
                contexts.append(dict(context=context,seed=a.seed+ep,step=world.t,eligible=int(len(idx)),score=float(world.score)))
                saved[f'state_{context}']=state.numpy();saved[f'mask_{context}']=mask.numpy();saved[f'prob_{context}']=prob.numpy()
                # Input shuffles preserve observed values; history reset tests memory dependence.
                changes={}
                for name,cols in [('target_position',slice(0,2)),('target_type',slice(2,5)),('target_work_units',slice(5,6))]:
                    z=state.clone();_,zt,_,_=parts(z,n,t);perm=valid_targets[torch.as_tensor(rng.permutation(len(valid_targets)))];zt[valid_targets,cols]=ta[perm,cols];changes[name]=z
                z=state.clone();zd,_,_,_=parts(z,n,t);di=torch.where(alive)[0];perm=di[torch.as_tensor(rng.permutation(len(di)))];zd[di,:2]=dr[perm,:2];changes['drone_position']=z
                z=state.clone();_,_,zp,zv=parts(z,n,t);zp.zero_();zp[:,exists]=1/max(1,int(exists.sum()));zv.zero_();changes['all_previous_intent_reset']=z
                for name,z in changes.items():
                    q=torch.softmax(logits(z[None])[0].masked_fill(~mask,-torch.inf),-1)
                    for i in idx:effects.append(dict(context=context,stage=stage,feature=name,flip=int(q[i].argmax()!=choice[i]),tv=float((q[i]-prob[i]).abs().sum()/2)))
                for side in ['actor','critic']:
                    enc=encode(bat,side=='critic');dv,tv=enc[0][0,alive],enc[1][0,exists]
                    reps.append(dict(context=context,stage=stage,side=side,drone_cos=offdiag_cos(dv),target_cos=offdiag_cos(tv),drone_spread=float(dv.std(0,unbiased=False).norm()),target_spread=float(tv.std(0,unbiased=False).norm())))
                # Controlled probes: modify just one candidate's position or one drone's history.
                variants=[];keys=[]
                for i in idx:
                    for j in torch.where(mask[i])[0].tolist():
                        for label,factor in [('nearer',.5),('farther',1.5)]:
                            z=state.clone();_,zt,_,_=parts(z,n,t);zt[j,:2]=dr[i,:2]+factor*(ta[j,:2]-dr[i,:2]);variants.append(z);keys.append((i,j,'distance',label))
                        other=next(k for k in torch.where(mask[i])[0].tolist() if k!=j)
                        for label,k in [('same',j),('other',other)]:
                            z=state.clone();_,_,zp,zv=parts(z,n,t);zp[i].zero_();zv[i].zero_();zp[i,k]=1;zv[i,k]=1;variants.append(z);keys.append((i,j,'own_history',label))
                if variants:
                    pp=torch.softmax(logits(torch.stack(variants)).masked_fill(~mask[None],-torch.inf),-1)
                    for k,(i,j,feature,label) in enumerate(keys):probes.append(dict(context=context,stage=stage,agent=i,target=j,feature=feature,condition=label,probability=float(pp[k,i,j])))
            action,_=scheduled_action(model,world,schedule,resolver=False);world.step(action)
        rollouts.append(dict(seed=a.seed+ep,**world.metrics()));print('episode',ep+1,'return',world.metrics()['team_return'],flush=True)
    np.savez_compressed(out/'inputs.npz',**saved)
    for name,rows in [('contexts',contexts),('interventions',effects),('controlled_probes',probes),('representations',reps),('rollouts',rollouts)]:write_csv(out/f'{name}.csv',rows)
    summary={'checkpoint':a.checkpoint,'sha256':hashlib.sha256((out/a.checkpoint).read_bytes()).hexdigest(),'metadata':checkpoint_payload(out/a.checkpoint)['metadata'],'episodes':a.episodes,'seed_start':a.seed,'resolver':False,'contexts':len(contexts),'decision_samples':sum(x['eligible'] for x in contexts),'team_return':float(np.mean([r['team_return'] for r in rollouts])),'mission_success':float(np.mean([r['mission_success'] for r in rollouts])),'input_effects':{},'controlled_effects':{}}
    for feature in changes:
        summary['input_effects'][feature]={}
        for stage in ['initial','in_progress']:
            rows=[r for r in effects if r['feature']==feature and r['stage']==stage]
            summary['input_effects'][feature][stage]={'samples':len(rows),'flip_rate':float(np.mean([r['flip'] for r in rows])),'tv':float(np.mean([r['tv'] for r in rows]))}
    grouped={}
    for r in probes:grouped.setdefault((r['context'],r['agent'],r['target'],r['feature'],r['stage']),{})[r['condition']]=r['probability']
    for feature,pos,neg in [('distance','nearer','farther'),('own_history','same','other')]:
        summary['controlled_effects'][feature]={}
        for stage in ['initial','in_progress']:
            dif=[v[pos]-v[neg] for k,v in grouped.items() if k[-2:]==(feature,stage)]
            summary['controlled_effects'][feature][stage]={'samples':len(dif),'mean_probability_difference':float(np.mean(dif)),'positive_fraction':float(np.mean(np.array(dif)>0))}
    summary['representations']={side:{k:float(np.nanmean([r[k] for r in reps if r['side']==side])) for k in ['drone_cos','target_cos','drone_spread','target_spread']} for side in ['actor','critic']}
    summary['caveats']=['Interventions measure sensitivity, not correctness; fixed action masks isolate scoring changes.','Controlled modified states are synthetic and may be outside the training distribution.','Cosine similarity alone cannot establish representation collapse.','Decision samples within an episode are correlated.']
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    features=list(changes);stages=['initial','in_progress'];fig,axes=plt.subplots(1,2,figsize=(13,5))
    v=np.array([[100*summary['input_effects'][f][g]['flip_rate'] for g in stages] for f in features]);axes[0].imshow(v,cmap='YlOrRd',vmin=0,vmax=100,aspect='auto');axes[0].set_yticks(range(len(features)),[f.replace('_',' ') for f in features]);axes[0].set_xticks([0,1],stages);axes[0].set_title('Choice changes after input intervention (%)')
    for i in range(len(features)):
        for j in range(2):axes[0].text(j,i,f'{v[i,j]:.1f}',ha='center',va='center')
    labels=['Nearer vs farther','Own history: same vs other'];v=[100*summary['controlled_effects'][f]['in_progress']['mean_probability_difference'] for f in ['distance','own_history']];axes[1].barh(labels,v);axes[1].axvline(0,color='black',lw=1);axes[1].set_xlabel('Probability difference (percentage points)');axes[1].set_title('Controlled probes: in progress');fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(out/f'input_and_controlled_effects.{ext}',dpi=180)
    plt.close(fig)
    # Save representative state geometry, policy and critic action scores together.
    pick=max((x for x in contexts if x['step']==20),key=lambda x:x['eligible'])['context'];st=torch.from_numpy(saved[f'state_{pick}']);mask=torch.from_numpy(saved[f'mask_{pick}']);pr=torch.from_numpy(saved[f'prob_{pick}']);dr,ta,_,_=parts(st,n,t);di=torch.where(dr[:,2]>.5)[0];ti=torch.where(ta[:,2:5].sum(-1)>.5)[0];act=pr.argmax(-1);enc=encode(st[None],True);qs=[]
    for i in di:
        acts=act.repeat(t,1);acts[:,i]=torch.arange(t);q=model.q_head(model.q_body.from_encoded(*(v.expand(t,*v.shape[1:]) for v in enc),acts)).squeeze(-1);qs.append(q)
    fig,axes=plt.subplots(1,3,figsize=(15,5));axes[0].scatter(dr[di,0],dr[di,1],label='Drones');axes[0].scatter(ta[ti,0],ta[ti,1],marker='x',label='Targets')
    for i in di:
        j=act[i];axes[0].plot([dr[i,0],ta[j,0]],[dr[i,1],ta[j,1]],alpha=.3,color='gray');axes[0].annotate(f'D{int(i)}',dr[i,:2])
    for j in ti:axes[0].annotate(f'T{int(j)}',ta[j,:2])
    axes[0].legend();axes[0].set_title(f'Context {pick}: step 20, greedy choices');axes[0].set_aspect('equal')
    for ax,data,title in [(axes[1],pr[di][:,ti],'Actor probabilities'),(axes[2],torch.stack(qs)[:,ti],'Critic Q: change one drone action')]:
        im=ax.imshow(data.numpy(),aspect='auto');ax.set_xticks(range(len(ti)),ti.tolist());ax.set_yticks(range(len(di)),di.tolist());ax.set_xlabel('Target');ax.set_ylabel('Drone');ax.set_title(title);fig.colorbar(im,ax=ax)
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(out/f'decision_case.{ext}',dpi=180)
    print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':
    with torch.no_grad():main()
