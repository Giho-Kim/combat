"""Locate entity homogenization in cached no-resolver anatomy states."""
import csv,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.inspect_separate_representation import parts,offdiag_cos,write_csv
from pointmass_rl.strike_ppo import model_from_checkpoint
from pointmass_rl.env import Config
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
@torch.no_grad()
def main():
    out=Path(sys.argv[1]);run=out.parent.parent;config=Config.load(out/'config.json');states=np.load(out/'inputs.npz');contexts=list(csv.DictReader((out/'contexts.csv').open()));torch.set_num_threads(1);rows=[];decisions=[];q_audits=[]
    for cp in ['checkpoint_100000.pt','checkpoint_1500000.pt','checkpoint_3000000.pt']:
        model=model_from_checkpoint(out/cp if (out/cp).exists() else run/cp,config);model.eval();n,t=model.n_agents,model.n_targets
        stages={};handles=[]
        for name in ['drone_encoder','drone_input','drone_self','drone_cross','drone_norm']:
            def hook(module,args,result,name=name):stages[name]=result.detach()
            handles.append(getattr(model.actor_encoder,name).register_forward_hook(hook))
        for cx in contexts:
            k=cx['context'];state=torch.from_numpy(states[f'state_{k}'])[None];dr,ta,_,_=parts(state,n,t);alive=dr[0,:,2]>.5;exists=ta[0,:,2:5].sum(-1)>.5;mask=torch.from_numpy(states[f'mask_{k}']);enc=model.q_body.encode(state,n,t,encoder=model.actor_encoder);logits=model.logits_from_encoded(*enc[:2])[0];pr=torch.softmax(logits.masked_fill(~mask,-torch.inf),-1);valid=alive & (mask.sum(-1)>1);ids=torch.where(valid)[0]
            if len(ids)>1:
                ch=pr[ids].argmax(-1);counts=torch.bincount(ch,minlength=t)
                decisions.append(dict(checkpoint=cp,context=k,step=cx['step'],drone_count=len(ids),distinct_choices=len(ch.unique()),largest_choice_fraction=float(counts.max()/len(ids)),logit_drone_std=float(logits[ids][:,exists].std(0,unbiased=False).mean())))
            if cp=='checkpoint_3000000.pt' and len(ids):
                actions=pr.argmax(-1);encoded=model.q_body.encode(state,n,t);variants=[];keys=[]
                for i in ids:
                    for j in torch.where(mask[i])[0]:
                        act=actions.clone();act[i]=j;variants.append(act);keys.append((int(i),int(j)))
                q=model.q_head(model.q_body.from_encoded(*(v.expand(len(variants),*v.shape[1:]) for v in encoded),torch.stack(variants))).squeeze(-1)
                for i in ids.tolist():
                    selected=[h for h,key in enumerate(keys) if key[0]==i];v=q[selected];q_audits.append(dict(context=k,step=cx['step'],agent=i,q_range=float(v.max()-v.min()),q_mean=float(v.mean()),probability_range=float(pr[i,mask[i]].max()-pr[i,mask[i]].min())))
            for name,x in stages.items():
                v=x[0,alive];rows.append(dict(checkpoint=cp,context=k,step=cx['step'],layer=name,cos=offdiag_cos(v),spread=float(v.std(0,unbiased=False).norm()),mean_norm=float(v.norm(dim=-1).mean())))
        for h in handles:h.remove()
    write_csv(out/'critic_action_sensitivity.csv',q_audits)
    write_csv(out/'layer_representations.csv',rows);write_csv(out/'drone_choice_agreement.csv',decisions)
    summary={}
    for cp in sorted(set(r['checkpoint'] for r in rows)):
        rr=[r for r in decisions if r['checkpoint']==cp];summary[cp]={'contexts':len(rr),'all_same_target_fraction':float(np.mean([r['distinct_choices']==1 for r in rr])),'mean_largest_choice_fraction':float(np.mean([r['largest_choice_fraction'] for r in rr])),'mean_logit_drone_std':float(np.mean([r['logit_drone_std'] for r in rr])),'layers':{name:{key:float(np.mean([r[key] for r in rows if r['checkpoint']==cp and r['layer']==name])) for key in ['cos','spread','mean_norm']} for name in stages}}
    (out/'layer_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
    fig,axes=plt.subplots(1,2,figsize=(12,4.6))
    for cp,v in summary.items():
        axes[0].plot(list(stages),[v['layers'][s]['spread'] for s in stages],marker='o',label=cp.removeprefix('checkpoint_').removesuffix('.pt'));axes[1].plot(list(stages),[max(1e-9,1-v['layers'][s]['cos']) for s in stages],marker='o')
    axes[0].set_ylabel('Across-drone embedding spread');axes[1].set_ylabel('1 - mean pairwise cosine');axes[1].set_yscale('log')
    for ax in axes:ax.tick_params(axis='x',rotation=25);ax.grid(alpha=.2)
    axes[0].legend(title='Training transitions');fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(out/f'drone_representation_layers.{ext}',dpi=180)
if __name__=='__main__':main()
