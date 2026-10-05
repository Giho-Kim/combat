"""Model-free v12 tuning and independent evaluation; no rollout model or oracle.

Run from the repository root with python -m scripts.tune_target_coma.
"""
import argparse
import csv
from dataclasses import replace, asdict
import json
from pathlib import Path
import shutil
import time

import torch

from pointmass_rl.env import Config
from pointmass_rl.strike_ppo import train_mappo, save_checkpoint, model_from_checkpoint, evaluate_model


VARIANTS = {
    'q_stable': dict(coma_advantage='q', gae_lambda=.97, decision_interval=1),
    'mc_step': dict(coma_advantage='sampled_return', gae_lambda=1., decision_interval=1),
    'mc_hold5': dict(coma_advantage='sampled_return', gae_lambda=1., decision_interval=5),
    'mc_hold10': dict(coma_advantage='sampled_return', gae_lambda=1., decision_interval=10),
    'mc_commit': dict(coma_advantage='sampled_return', gae_lambda=1., decision_interval=1,
                      commit_target=True),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=VARIANTS, default='mc_hold5')
    parser.add_argument('--steps', type=int, default=1_000_000)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--out', required=True)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--critic-epochs', type=int, default=12)
    parser.add_argument('--n-envs', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--entropy-coef', type=float, default=.005)
    parser.add_argument('--gamma', type=float, default=1.)
    parser.add_argument('--eval-interval', type=int, default=100_000)
    parser.add_argument('--init-model', help='Warm-start weights with fresh optimizers')
    parser.add_argument('--actor-distances', action='store_true',
                        help='v13 actor: add observed distances; Q stays the v12 pair architecture')
    parser.add_argument('--b12-probability', type=float, default=0.,
                        help='Fraction of training resets forced to B12, still random distance 25-58')
    parser.add_argument('--verify', help='Only evaluate this checkpoint, without training')
    parser.add_argument('--config', default='configs/five_agents.json')
    parser.add_argument('--eval-seed', type=int, default=20000)
    parser.add_argument('--eval-episodes', type=int, default=90)
    args = parser.parse_args()
    torch.set_num_threads(1)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.verify:
        config = Config.load(args.config)
        snapshot = out / 'model.pt'
        if Path(args.verify).resolve() != snapshot.resolve():
            shutil.copy2(args.verify, snapshot)
        config.save(out / 'config.json')
        model = model_from_checkpoint(snapshot, config)
        if any(model.training_settings.get(k, 0) for k in
               ('lookahead_credit', 'guided_training', 'deadline_mask')):
            raise ValueError('Model-assisted checkpoint is outside this model-free audit')
        report = {}
        for mode, resolver in [('no_resolver', False), ('resolver', True)]:
            report[mode] = evaluate_model(model, config, args.eval_episodes,
                                          args.eval_seed, resolver=resolver)
            print(mode, [report[mode][f'case{i}_damage'] for i in (1, 2, 3)], flush=True)
        (out / 'evaluation.json').write_text(json.dumps(dict(
            checkpoint=args.verify, seed=args.eval_seed, episodes=args.eval_episodes,
            snapshot=str(snapshot), training_settings=model.training_settings,
            config=asdict(config), results=report), indent=2) + '\n')
        return
    if (out / 'config.json').exists():
        raise ValueError(f'Refusing to overwrite existing experiment: {out}')
    variant = dict(VARIANTS[args.variant])
    config = replace(Config.load(args.config), discount_gamma=args.gamma,
                     gae_lambda=variant.pop('gae_lambda'),
                     decision_interval=variant.pop('decision_interval'),
                     commit_target=variant.pop('commit_target', False))
    config.save(out / 'config.json')
    (out / 'arguments.json').write_text(json.dumps(vars(args), indent=2) + '\n')
    initial_checkpoint = None
    if args.init_model:
        # Snapshot the exact source even if another run later replaces best.pt.
        initial_checkpoint = out / 'initial.pt'
        shutil.copy2(args.init_model, initial_checkpoint)
        initial = model_from_checkpoint(initial_checkpoint, config)
        if any(initial.training_settings.get(k, 0) for k in
               ('lookahead_credit', 'guided_training', 'deadline_mask')):
            raise ValueError('Model-assisted initialization is outside this model-free sweep')
    start = time.monotonic()
    model, episodes, evaluations, actual = train_mappo(
        config, args.steps, seed=args.seed, algorithm='target_coma',
        n_envs=args.n_envs, rollout_steps=100, epochs=args.epochs,
        critic_epochs=args.critic_epochs, minibatch_size=1024,
        critic_minibatch_size=512, learning_rate=args.learning_rate,
        entropy_coef=args.entropy_coef, policy_clip=.15,
        eval_interval=args.eval_interval, eval_episodes=30, eval_seed=10000,
        best_metric='suite_damage', best_path=out / 'best.pt',
        latest_path=out / 'latest.pt', diagnostics_path=out / 'training_updates.csv',
        guided_training=False, lookahead_credit=0., deadline_mask=False,
        initial_checkpoint=initial_checkpoint,
        actor_target_distances=args.actor_distances,
        actor_decision_minibatches=True,
        target_coma_critic='pair',
        training_b12_probability=args.b12_probability,
        **variant)
    metadata = dict(arguments=vars(args), config=asdict(config),
                    training_settings=model.training_settings,
                    actual_agent_transitions=actual, elapsed_seconds=time.monotonic() - start)
    save_checkpoint(out / 'final.pt', model, metadata)
    (out / 'training_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    for name, rows in [('training_episodes', episodes), ('training_evaluations', evaluations)]:
        if rows:
            with (out / f'{name}.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    print('FINISHED', out, metadata['elapsed_seconds'], flush=True)


if __name__ == '__main__':
    main()
