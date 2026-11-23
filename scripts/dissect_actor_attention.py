"""Inspect a saved Target Allocation actor using messages and interventions."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import types

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pointmass_rl.env import Config, World
from pointmass_rl.strike_ppo import (DecisionSchedule,_model_observation,
    model_from_checkpoint,scheduled_action)


def csv_file(path,rows):
    if not rows:return
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


class Inspector:
    def __init__(self,model):
        self.model=model;self.drop=set();self.capture=False;self.records={};self.peer_mode=None
        def preprocess(module,args):
            if self.peer_mode is None:return
            query,peers,mask=args
            if self.peer_mode=='mean':
                peers=(peers*mask[...,None]).sum(-2,keepdim=True)/mask.sum(-1,keepdim=True)[...,None].clamp_min(1)
                peers=peers.expand_as(args[1])
            else:
                peers=torch.zeros_like(peers);peers[...,2]=1;peers[...,5]=1/query.shape[-2]
            return query,peers,mask
        self.peer_hook=model.actor_body.register_forward_pre_hook(preprocess)
        self.original=[]
        for index,layer in enumerate(model.actor_body.layers):
            self.original.append(layer.aggregate)
            def aggregate(module,query,peers,mask,index=index):
                q=module.query(query).reshape(*query.shape[:-1],module.heads,module.width).unsqueeze(-3)
                k=module.key(peers).reshape(*peers.shape[:-1],module.heads,module.width)
                v=torch.nn.functional.softplus(module.value(peers)).reshape_as(k)
                gate=torch.sigmoid((q*k).sum(-1)/module.width**.5)
                weighted=gate.unsqueeze(-1)*v*mask[...,None,None]
                message=weighted.sum(-3)
                if self.capture:
                    self.records[index]=dict(gate=gate.detach().numpy(),
                        strength=(gate*v.norm(dim=-1)*mask[...,None]).detach().numpy(),
                        message_norm=message.norm(dim=-1).detach().numpy())
                for head in range(module.heads):
                    if (index,head) in self.drop:message[...,head,:]=0
                return message.flatten(-2)
            layer.aggregate=types.MethodType(aggregate,layer)

    def restore(self):
        self.peer_hook.remove()
        for layer,original in zip(self.model.actor_body.layers,self.original):layer.aggregate=original


@torch.no_grad()
def extract(model,world,schedule):
    saved=[]
    def hook(module,args):saved.extend(t.detach().clone() for t in args)
    handle=model.actor_body.register_forward_pre_hook(hook)
    logits=model.actor_logits(torch.as_tensor(_model_observation(model,world,schedule)))
    handle.remove()
    query,peers,mask=saved
    reconstructed=model.target_head(model.actor_body(query,peers,mask)).squeeze(-1)
    torch.testing.assert_close(logits,reconstructed,atol=1e-6,rtol=1e-6)
    return query.numpy(),peers.numpy(),mask.numpy(),logits.numpy()


@torch.no_grad()
def collect(model,config,seeds,out):
    pool=[];trajectories=[]
    for seed in seeds:
        w=World(config);w.reset(seed);s=DecisionSchedule(config);life=w.target_life.copy()
        while not w.done:
            available=model.available_actions(w);_,free=s.mask(w,available)
            after_consumption=bool(np.any(life!=w.target_life))
            take=w.t in (0,1,5,10,20,40,60,80) or after_consumption
            if take:q,p,mask,logits=extract(model,w,s)
            action,_=scheduled_action(model,w,s,resolver=True)
            if take:
                stage='initial' if w.t==0 else 'after_consumption' if after_consumption else 'negotiation' if w.t<=10 else 'travel'
                for i in np.flatnonzero(w.agent_active & free & (available.sum(-1)>1)):
                    choice=int(np.where(available[i],logits[i],-np.inf).argmax())
                    nearest=int(np.where(available[i],np.linalg.norm(w.targets-w.pos[i],axis=-1),np.inf).argmin())
                    pool.append(dict(query=q[i],peers=p[i],peer_mask=mask[i],available=available[i],
                        meta=dict(seed=seed,step=int(w.t),stage=stage,agent=int(i),actor_choice=choice,
                            executed_choice=int(action['target'][i]),nearest=nearest,
                            own_position=(w.pos[i]/config.size).tolist(),positions=(w.pos/config.size).tolist(),
                            targets=(w.targets/config.size).tolist(),active=w.agent_active.tolist(),
                            types=w.target_type.tolist(),life=w.target_life.tolist(),
                            previous_choices=s.targets.tolist())))
                trajectories.append(dict(seed=seed,step=int(w.t),score=float(w.score),
                    pos=(w.pos/config.size).tolist(),targets=(w.targets/config.size).tolist(),
                    active=w.agent_active.tolist(),types=w.target_type.tolist(),life=w.target_life.tolist(),
                    selected=action['target'].tolist()))
            life=w.target_life.copy();w.step(action)
        print('Collected seed',seed,'return',float(w.score),flush=True)
    rng=np.random.default_rng(7);selected=[]
    for stage in ('initial','negotiation','travel','after_consumption'):
        candidates=[r for r in pool if r['meta']['stage']==stage]
        indices=rng.choice(len(candidates),min(160,len(candidates)),replace=False)
        selected.extend(candidates[i] for i in indices)
    np.savez_compressed(out/'inputs.npz',query=np.stack([r['query'] for r in selected]),
        peers=np.stack([r['peers'] for r in selected]),peer_mask=np.stack([r['peer_mask'] for r in selected]),
        available=np.stack([r['available'] for r in selected]))
    (out/'contexts.json').write_text(json.dumps([r['meta'] for r in selected]))
    (out/'trajectories.json').write_text(json.dumps(trajectories))


@torch.no_grad()
def scores(model,q,p,mask):
    return model.target_head(model.actor_body(torch.as_tensor(q),torch.as_tensor(p),
        torch.as_tensor(np.array(mask,copy=True)))).squeeze(-1).numpy()


def probabilities(logits,available):
    x=np.where(available,logits,-1e30);x=x-x.max(-1,keepdims=True)
    p=np.exp(x)*available;return p/p.sum(-1,keepdims=True)


@torch.no_grad()
def feature_and_head_tests(model,inspector,data,contexts,out):
    q,p,mask,available=[data[k] for k in ('query','peers','peer_mask','available')]
    base=[];gates={i:[] for i in range(2)};strength={i:[] for i in range(2)}
    inspector.capture=True
    for start in range(0,len(q),32):
        base.extend(scores(model,q[start:start+32],p[start:start+32],mask[start:start+32]))
        for i in range(2):
            gates[i].append(inspector.records[i]['gate']);strength[i].append(inspector.records[i]['strength'])
    inspector.capture=False;base=np.array(base);base_prob=probabilities(base,available)
    choice=np.argmax(base_prob,-1)
    gate_arrays={i:np.concatenate(gates[i]) for i in range(2)}
    strength_arrays={i:np.concatenate(strength[i]) for i in range(2)}
    np.savez_compressed(out/'attention.npz',baseline_logits=base,
        **{f'layer_{i}_gate':gate_arrays[i] for i in range(2)},
        **{f'layer_{i}_strength':strength_arrays[i] for i in range(2)})
    rows=[];head_rows=[];changed_logits={}
    features=['target_position','target_type','target_life','time','own_previous_probability',
        'own_previous_choice','own_velocity','peer_previous_intent','peer_position',
        'peer_velocity','peer_idle','peer_records_initial','peer_records_mean',
        'all_peer_messages','layer_1_messages','layer_2_messages']
    features += [f'layer_{i+1}_head_{h+1}' for i in range(2) for h in range(4)]
    for feature in features:
        inspector.drop=set();changed=[]
        for start in range(0,len(q),32):
            a=q[start:start+32].copy();b=p[start:start+32].copy();m=mask[start:start+32].copy()
            if feature=='target_position':a[...,:2]=0
            elif feature=='target_type':a[...,2:5]=1/3
            elif feature=='target_life':a[...,5]=.5
            elif feature=='time':a[...,6]=1
            elif feature=='own_previous_probability':a[...,7]=1/a.shape[1]
            elif feature=='own_previous_choice':a[...,8]=0
            elif feature=='own_velocity':a[...,11:13]=0
            elif feature=='peer_previous_intent':b[...,5]=1/a.shape[1];b[...,6]=0
            elif feature=='peer_position':b[...,:2]=0
            elif feature=='peer_velocity':b[...,3:5]=0
            elif feature=='peer_idle':b[...,2]=1
            elif feature=='peer_records_initial':
                b[:]=0;b[...,2]=1;b[...,5]=1/a.shape[1]
            elif feature=='peer_records_mean':
                b=np.broadcast_to((b*m[...,None]).sum(-2,keepdims=True)/np.maximum(1,m.sum(-1,keepdims=True)[...,None]),b.shape).astype(b.dtype)
            elif feature=='all_peer_messages':m[:]=False
            elif feature in ('layer_1_messages','layer_2_messages'):
                layer=0 if feature=='layer_1_messages' else 1
                inspector.drop={(layer,h) for h in range(4)}
            else:
                parts=feature.split('_');inspector.drop={(int(parts[1])-1,int(parts[3])-1)}
            changed.extend(scores(model,a,b,m))
        changed=np.array(changed);changed_logits[feature]=changed;prob=probabilities(changed,available)
        flip=np.argmax(prob,-1)!=choice
        for stage in ['all','initial','negotiation','travel','after_consumption']:
            selected=np.ones(len(q),dtype=bool) if stage=='all' else np.array([c['stage']==stage for c in contexts])
            if not selected.any():continue
            kl=(base_prob*(np.log(np.maximum(base_prob,1e-30))-np.log(np.maximum(prob,1e-30)))).sum(-1)
            rows.append(dict(feature=feature,stage=stage,samples=int(selected.sum()),
                decision_flip_rate=float(flip[selected].mean()),mean_kl=float(kl[selected].mean()),
                original_choice_probability_drop=float((base_prob[np.arange(len(q)),choice]-prob[np.arange(len(q)),choice])[selected].mean())))
        print('Intervention',feature,'flip',float(flip.mean()),flush=True)
    inspector.drop=set()
    for layer in range(2):
        for head in range(4):
            valid=mask & available[...,None]
            values=gate_arrays[layer][...,head][valid]
            selected=np.arange(len(q))
            magnitude=strength_arrays[layer][selected,choice,:,head].sum(-1)
            head_rows.append(dict(layer=layer+1,head=head+1,mean_gate=float(values.mean()),
                sd_gate=float(values.std()),fraction_below_005=float((values<.05).mean()),
                fraction_above_095=float((values>.95).mean()),mean_message_magnitude=float(magnitude.mean())))
    csv_file(out/'interventions.csv',rows);csv_file(out/'heads.csv',head_rows)
    np.savez_compressed(out/'intervention_logits.npz',**changed_logits)
    return base


@torch.no_grad()
def controlled_tests(model,data,contexts,base,out):
    rng=np.random.default_rng(7);rows=[]
    selected=rng.choice(len(contexts),min(240,len(contexts)),replace=False)
    for index in selected:
        q=data['query'][index];p=data['peers'][index];mask=data['peer_mask'][index];available=data['available'][index]
        legal=np.flatnonzero(available);choice=int(np.where(available,base[index],-np.inf).argmax())
        ranked=sorted(legal,key=lambda k:-base[index,k])[:2]
        for k in ranked:
            alt=next(int(t) for t in legal if t!=k)
            variants=[];names=[]
            def add(name,a,b):names.append(name);variants.append((a,b))
            for label,scale in [('target_nearer',.8),('target_farther',1.2)]:
                a=q.copy();a[k,:2]*=scale;add(label,a,p)
            if q[k,5]<=.51:
                for typ in (1,2,3):
                    a=q.copy();a[k,2:5]=0;a[k,1+typ]=1;add(f'type_{typ}',a,p)
            for target,label in [(k,'own_stay'),(alt,'own_other')]:
                a=q.copy();a[:,7:9]=0;a[target,7:9]=1;add(label,a,p)
            peers=np.flatnonzero(mask[k] & (p[k,:,2]>.5))
            peer=None
            if len(peers):
                peer=int(min(peers,key=lambda j:np.linalg.norm(p[k,j,:2]-q[k,:2])))
                for target,label in [(k,'peer_same'),(alt,'peer_other')]:
                    b=p.copy();b[:,peer,5:7]=0;b[target,peer,5:7]=1;add(label,q,b)
                origin=np.asarray(contexts[index]['own_position'])
                if ((origin-.5*q[k,:2]>=0)&(origin-.5*q[k,:2]<=1)).all():
                    for factor,label in [(.5,'claiming_peer_nearer'),(-.5,'claiming_peer_farther')]:
                        b=p.copy();b[:,peer,:2]=factor*q[k,:2];b[:,peer,5:7]=0;b[k,peer,5:7]=1;add(label,q,b)
            batchq=np.stack([v[0] for v in variants]);batchp=np.stack([v[1] for v in variants])
            logits=scores(model,batchq,batchp,np.broadcast_to(mask,(len(variants),*mask.shape)))
            probs=probabilities(logits,np.broadcast_to(available,logits.shape))
            for n,label in enumerate(names):
                row=dict(context=int(index),seed=contexts[index]['seed'],step=contexts[index]['step'],
                    stage=contexts[index]['stage'],agent=contexts[index]['agent'],target=int(k),
                    type=contexts[index]['types'][k],life=contexts[index]['life'][k],intervention=label,
                    probability=float(probs[n,k]),score_gap=float(logits[n,k]-np.mean(logits[n,legal[legal!=k]])),
                    selected=int(np.argmax(probs[n])),peer=-1 if peer is None else peer)
                if peer is None:row['peer_closer']=False
                else:row['peer_closer']=bool(np.linalg.norm(p[k,peer,:2]-q[k,:2])<np.linalg.norm(q[k,:2]))
                rows.append(row)
    csv_file(out/'controlled_probes.csv',rows)


@torch.no_grad()
def evaluate_ablation(model,inspector,config,seeds,drop,resolver,peer_mode=None):
    inspector.drop=drop;inspector.peer_mode=peer_mode;values=[]
    for seed in seeds:
        w=World(config);w.reset(seed);schedule=DecisionSchedule(config)
        while not w.done:
            action,_=scheduled_action(model,w,schedule,resolver=resolver);w.step(action)
        values.append((seed,float(w.score)))
    inspector.drop=set();inspector.peer_mode=None;return values


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',default='runs/target_allocation_variable_aligned_basic_warmstart')
    parser.add_argument('--out',default='runs/target_allocation_attention_anatomy')
    parser.add_argument('--episodes',type=int,default=30);parser.add_argument('--static-only',action='store_true')
    args=parser.parse_args();run=ROOT/args.run;out=ROOT/args.out;out.mkdir(parents=True,exist_ok=True)
    config=Config.load(run/'config.json');torch.set_num_threads(1)
    model=model_from_checkpoint(run/'latest.pt',config);model.eval()
    if not (out/'inputs.npz').exists():collect(model,config,range(20000,20000+args.episodes),out)
    data=np.load(out/'inputs.npz');contexts=json.loads((out/'contexts.json').read_text())
    inspector=Inspector(model)
    try:
        # Validate the instrumented implementation against the unmodified actor.
        q,p,m=[torch.as_tensor(data[k][:8]) for k in ('query','peers','peer_mask')]
        actual=model.target_head(model.actor_body(q,p,m));inspector.restore()
        expected=model.target_head(model.actor_body(q,p,m))
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        inspector=Inspector(model)
        base=feature_and_head_tests(model,inspector,data,contexts,out)
        controlled_tests(model,data,contexts,base,out)
        if not args.static_only:
            interventions=list(csv.DictReader((out/'interventions.csv').open()))
            heads=sorted([r for r in interventions if 'head' in r['feature'] and r['stage']=='all'],
                         key=lambda r:-float(r['decision_flip_rate']))
            top=heads[0]['feature'].split('_');one_head={(int(top[1])-1,int(top[3])-1)}
            variants={'baseline':set(),'no_peer_messages':{(l,h) for l in range(2) for h in range(4)},
                'no_layer_1_messages':{(0,h) for h in range(4)},'no_layer_2_messages':{(1,h) for h in range(4)},
                'drop_most_sensitive_head':one_head}
            rows=[]
            for resolver in (False,True):
                for label,drop in variants.items():
                    values=evaluate_ablation(model,inspector,config,range(20000,20000+args.episodes),drop,resolver)
                    rows.extend(dict(seed=seed,resolver=resolver,ablation=label,team_return=value) for seed,value in values)
                    csv_file(out/'rollout_ablations.csv',rows)
                    print('Rollout',resolver,label,'return',np.mean([v for _,v in values]),flush=True)
    finally:inspector.restore()
    (out/'protocol.json').write_text(json.dumps(dict(checkpoint=str(run/'latest.pt'),
        sha256=hashlib.sha256((run/'latest.pt').read_bytes()).hexdigest(),episodes=args.episodes,
        seed_start=20000,context_samples=len(contexts),
        method='Saved forward inputs; gate and value-message inspection; fixed-mask feature perturbations; controlled pair probes; optional full-trajectory message ablations.',
        note='Gates are sigmoid, not normalized. Feature clearing is a sensitivity diagnostic; controlled probes and rollout ablations provide separate intervention evidence.'),indent=2))


if __name__=='__main__':main()
