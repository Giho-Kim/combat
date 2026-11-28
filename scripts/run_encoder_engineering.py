"""Matched 500k encoder experiments; no extra observations or reward changes."""
import concurrent.futures,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/encoder_engineering_500k'
OUT.mkdir(exist_ok=True)
variants=['baseline','residual_small','token_norm']
args='--mode known --algorithm target_allocation --config configs/target_allocation_variable.json --device cpu --steps 500000 --seed 7 --gamma 1.0 --gae-lambda 0.97 --n-envs 32 --rollout-steps 200 --learning-rate 1e-4 --policy-clip 0.2 --entropy-coef 0.01 --epochs 5 --critic-epochs 5 --minibatch-size 256 --critic-minibatch-size 256 --coma-advantage q --actor-samples all --decision-interval 1 --eval-interval 100000 --eval-episodes 30 --eval-seed 10000 --shared-attention --separate-encoders --target-kl 0.03'.split()
def run(v):
    path=OUT/v
    if (path/'training_metadata.json').exists():return v,0
    if (path/'config.json').exists():raise RuntimeError(f'Incomplete existing run: {path}')
    command=[sys.executable,'-m','pointmass_rl','train',*args,'--encoder-engineering',v,'--out',str(path)]
    (OUT/f'{v}_command.json').write_text(json.dumps(command,indent=2))
    with (OUT/f'{v}.log').open('w') as f:
        result=subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MPLCONFIGDIR':'/tmp/pointmass_mpl'})
    return v,result.returncode
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    for v,code in pool.map(run,variants):print(v,code,flush=True)
