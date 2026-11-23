"""Ten paired belief scenarios with F2 sampled on the feasible 5km arc."""
import argparse,json,sys,subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
def run_seed(args):
    seed,out=args
    from pointmass_rl.env import Config
    from scripts.compare_belief_heuristics import simulate
    c=Config.load(ROOT/'configs/belief.json');rows=[]
    for policy in [1,2]:
        data=simulate(c,seed,policy,arc_formation_two=True)
        (out/f'seed_{seed}_h{policy}.json').write_text(json.dumps(data));rows.append(data['summary'])
    return rows
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='runs/belief_arc_heuristic_comparison');p.add_argument('--workers',type=int,default=4);p.add_argument('--seeds',type=int,nargs='+',default=list(range(10000,10010)));a=p.parse_args();out=ROOT/a.out;out.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for rows in pool.map(run_seed,[(seed,out) for seed in a.seeds]):print(json.dumps(rows),flush=True)
    subprocess.run([sys.executable,str(ROOT/'scripts/compare_belief_heuristics.py'),'--out',str(out),'--seeds',*[str(s) for s in a.seeds],'--render-only','--no-video','--arc-formation-two'],check=True)
