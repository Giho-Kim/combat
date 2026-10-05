"""Model-free episodic parameter search on a trained actor's final linear layer.

Each candidate is one fixed policy run through complete environment episodes.
No counterfactual state simulation, dynamics predictor, deadline mask, or oracle.
The resolver is explicitly part of the deployment policy optimized here.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import shutil

import numpy as np
import torch

from pointmass_rl.env import Config
from pointmass_rl.strike_ppo import model_from_checkpoint, evaluate_model, save_checkpoint


_MODEL = _CONFIG = None


def initialize(model_path, config_path):
    global _MODEL, _CONFIG
    torch.set_num_threads(1)
    _CONFIG = Config.load(config_path)
    _MODEL = model_from_checkpoint(model_path, _CONFIG).eval()
    if any(_MODEL.training_settings.get(key, 0) for key in
           ('lookahead_credit', 'guided_training', 'deadline_mask')):
        raise ValueError('Model-assisted checkpoint is outside this experiment')


def evaluate_candidate(task):
    weights, seed, episodes = task
    with torch.no_grad():
        _MODEL.target_head.weight.copy_(torch.as_tensor(weights).reshape_as(_MODEL.target_head.weight))
    result = evaluate_model(_MODEL, _CONFIG, episodes, seed, resolver=True)
    cases = [result[f'case{i}_damage'] for i in (1, 2, 3)]
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--generations', type=int, default=8)
    parser.add_argument('--population', type=int, default=12)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--episodes', type=int, default=18)
    parser.add_argument('--sigma', type=float, default=.025)
    parser.add_argument('--seed', type=int, default=101)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'initial.pt').exists():
        raise ValueError(f'Refusing to overwrite {out}')
    shutil.copy2(args.model, out / 'initial.pt')
    shutil.copy2(args.config, out / 'config.json')
    (out / 'arguments.json').write_text(json.dumps(vars(args), indent=2) + '\n')
    initialize(out / 'initial.pt', out / 'config.json')
    rng = np.random.default_rng(args.seed)
    center = _MODEL.target_head.weight.detach().numpy().reshape(-1).copy()
    best_key = -float('inf')
    history = []
    _MODEL.training_settings.update(
        actor_finetuning='model_free_episodic_parameter_search',
        finetuning_resolver=True, finetuning_fit_seed_base=1_000_000,
        finetuning_selection_seed=10000,
        critic_frozen_during_actor_finetuning=True)
    initial_cases = evaluate_candidate((center, 10000, 30))
    best_key = sum(initial_cases)
    save_checkpoint(out / 'best.pt', _MODEL, dict(
        generation=-1, case_damage=initial_cases, eval_seed=10000,
        eval_episodes=30, algorithm='model_free_episodic_parameter_search'))
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn'),
                             initializer=initialize,
                             initargs=(str(out / 'initial.pt'), str(out / 'config.json'))) as pool:
        for generation in range(args.generations):
            noise = rng.normal(size=(args.population, len(center))).astype(np.float32)
            sigma = args.sigma * .9 ** generation
            candidates = np.concatenate((center[None], center + sigma * noise))
            fit_seed = 1_000_000 + 1000 * generation
            cases = list(pool.map(evaluate_candidate,
                                 [(weights, fit_seed, args.episodes) for weights in candidates]))
            scores = np.asarray(cases).sum(axis=1)
            winner = int(scores.argmax())
            selection = evaluate_candidate((candidates[winner], 10000, 30))
            # Selection uses the same tuning set as PPO; independent seeds are
            # evaluated only after selecting the final checkpoint.
            selection_key = sum(selection)
            if selection_key > best_key:
                best_key = selection_key
                save_checkpoint(out / 'best.pt', _MODEL, dict(
                    generation=generation, case_damage=selection, eval_seed=10000,
                    eval_episodes=30, algorithm='model_free_episodic_parameter_search'))
            row = dict(generation=generation, sigma=sigma, fit_seed=fit_seed,
                       fit_cases=cases, winner=winner, selection_cases=selection,
                       best_selection_sum=best_key)
            history.append(row)
            (out / 'generations.json').write_text(json.dumps(history, indent=2) + '\n')
            print(json.dumps(row), flush=True)
            center = candidates[winner].copy()
            if selection_key >= 27. - 1e-6:
                break
    print('FINISHED', out, 'best sum', best_key, flush=True)


if __name__ == '__main__':
    main()
