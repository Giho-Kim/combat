"""Render nine baseline MP4s directly with Pillow/ffmpeg (no browser capture)."""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pointmass_rl.env import Config, EVALUATION_TASKS, World, strike_action
from pointmass_rl.cli import make_policy


class ScaledDraw:
    """Draw at presentation coordinates on a native-resolution canvas."""

    def __init__(self, image, scale):
        self.draw = ImageDraw.Draw(image)
        self.scale = scale

    def coords(self, value):
        if isinstance(value, (tuple, list)):
            return [self.coords(item) for item in value]
        return value * self.scale

    def text(self, xy, text, **kwargs):
        self.draw.text(self.coords(xy), text, **kwargs)

    def rectangle(self, xy, *, width=1, **kwargs):
        self.draw.rectangle(self.coords(xy), width=width * self.scale, **kwargs)

    def line(self, xy, *, width=1, **kwargs):
        self.draw.line(self.coords(xy), width=width * self.scale, **kwargs)

    def polygon(self, xy, **kwargs):
        self.draw.polygon(self.coords(xy), **kwargs)

    def ellipse(self, xy, *, width=1, **kwargs):
        self.draw.ellipse(self.coords(xy), width=width * self.scale, **kwargs)

    def arc(self, xy, start, end, *, width=1, **kwargs):
        self.draw.arc(self.coords(xy), start, end, width=width * self.scale, **kwargs)


def staged_reveal_action(policy, obs, world, seen, pending, activated):
    """Keep the policy's original F2 target, but approach F1 until reveal."""
    original = policy.predict(obs)['target']
    targets = original.copy()
    formation_one = np.flatnonzero((world.target_formation == 0) & (world.target_life > 0))
    for agent in np.flatnonzero(world.agent_active):
        if activated[agent] >= 0:
            if world.target_life[activated[agent]] > 0:
                targets[agent] = activated[agent]
                continue
            activated[agent] = -1
        if (pending[agent] < 0 and world.target_formation[original[agent]] == 1
                and not seen[original[agent]]):
            pending[agent] = original[agent]
        if pending[agent] < 0:
            continue
        remembered = pending[agent]
        if seen[remembered]:
            activated[agent] = remembered
            pending[agent] = -1
            targets[agent] = remembered
        elif len(formation_one):
            distances = np.linalg.norm(world.targets[formation_one] - world.pos[agent], axis=1)
            targets[agent] = formation_one[np.argmin(distances)]
        else:
            # No valid F1 remains to approach; wait without revealing the F2 goal.
            targets[agent] = int(np.flatnonzero(world.target_formation == 0)[0])
    if np.any(world.agent_active & (world.target_formation[targets] == 1) & ~seen[targets]):
        raise AssertionError('A drone selected formation 2 before discovery')
    return strike_action(targets)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='runs/baseline_videos_hd')
    parser.add_argument('--cases', type=int, nargs='+', choices=(1, 2, 3), default=(1, 2, 3))
    parser.add_argument('--policies', nargs='+', choices=('nearest', 'approximate_dp', 'type_priority'),
                        default=('nearest', 'approximate_dp', 'type_priority'))
    parser.add_argument('--overwrite', action='store_true',
                        help='Replace existing MP4s only after the new encoding succeeds')
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    c = Config.load('configs/default.json')
    render_scale = 2
    fonts = {s: ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
                                   s * render_scale)
             for s in (12, 15, 19, 26)}
    colors = ['#56c8ff', '#a39bff', '#5ce1ae', '#ffcc66', '#ff8dae']
    target_colors = {1: '#ff7c75', 2: '#ffbd69', 3: '#b8d788'}
    rows = []
    for case, task in enumerate(EVALUATION_TASKS, 1):
        if case not in args.cases:
            continue
        for name, title in [('nearest', 'Nearest'), ('approximate_dp', 'Lookahead Planner'),
                            ('type_priority', 'Type Priority')]:
            if name not in args.policies:
                continue
            seed = 10000 + case - 1
            w = World(c)
            obs = w.reset(seed, evaluation_task=task)
            policy = make_policy(name, c, seed)
            frames = []
            seen = w.target_formation == 0
            discoveries = {}
            pending = np.full(c.n_agents, -1, dtype=int)
            activated = np.full(c.n_agents, -1, dtype=int)
            while True:
                dist = np.linalg.norm(w.pos[:, None] - w.targets[None], axis=-1)
                detected = ((dist <= c.sensor_range) & w.agent_active[:, None]).any(axis=0)
                for j in np.flatnonzero(detected & ~seen):
                    discoveries[int(j)] = w.t
                seen |= detected
                frames.append((w.snapshot(), seen.copy()))
                if w.done:
                    break
                action = staged_reveal_action(policy, obs, w, seen, pending, activated)
                obs, *_ = w.step(action)
            points = np.vstack((np.array(frames[0][0]['drones']), w.targets))
            lo, hi = points.min(axis=0) - 8, points.max(axis=0) + 8
            span = max(hi - lo)
            center = (lo + hi) / 2
            lo = center - span / 2
            scale = 560 / span
            def xy(p):
                return (40 + (p[0] - lo[0]) * scale, 660 - (p[1] - lo[1]) * scale)
            path = out / f'case{case}_{name}.mp4'
            if path.exists() and not args.overwrite:
                raise FileExistsError(path)
            render_path = path
            if path.exists():
                with tempfile.NamedTemporaryFile(dir=out, prefix=f'.{path.stem}.',
                                                 suffix='.mp4', delete=False) as temporary:
                    render_path = Path(temporary.name)
            cmd = ['ffmpeg', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                   '-s', '1920x1440', '-r', '10', '-i', '-', '-an', '-c:v', 'libx264',
                   '-preset', 'slow', '-crf', '16', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(render_path)]
            process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            for step, (state, visible) in enumerate(frames):
                im = Image.new('RGB', (960 * render_scale, 720 * render_scale), '#101924')
                d = ScaledDraw(im, render_scale)
                def label(p, txt, size=15, color='#e7edf5'):
                    d.text(p, txt, font=fonts[size], fill=color)
                label((30, 20), f'CASE {case}', 26)
                label((30, 57), '5 drones | 100-second mission | same layout across policies', 15, '#9aabba')
                map_image = Image.new('RGB', im.size, '#101924')
                map_draw = ScaledDraw(map_image, render_scale)
                def map_label(p, txt, size=15, color='#e7edf5'):
                    map_draw.text(p, txt, font=fonts[size], fill=color)
                for grid in range(10, 100, 10):
                    gx, gy = xy([grid, grid])
                    if 40 <= gx <= 600:
                        map_draw.line((gx, 100, gx, 660), fill='#1d2c3c')
                    if 100 <= gy <= 660:
                        map_draw.line((40, gy, 600, gy), fill='#1d2c3c')
                pos = np.array(state['drones'])
                for agent in range(c.n_agents):
                    trail = [xy(f[0]['drones'][agent]) for f in frames[:step+1]]
                    if len(trail) > 1:
                        map_draw.line(trail, fill=colors[agent], width=2)
                    x, y = xy(pos[agent])
                    if state['agent_active'][agent]:
                        map_draw.polygon([(x, y-7), (x-5, y+6), (x+5, y+6)], fill=colors[agent])
                        map_label((x+8, y-15), f'D{agent+1}', 12, colors[agent])
                        progress = state['agent_strike_progress'][agent]
                        if progress:
                            map_draw.arc((x-12, y-12, x+12, y+12), -90,
                                         -90 + 360*progress/c.strike_steps_per_life,
                                         fill=colors[agent], width=3)
                    else:
                        map_draw.line((x-4,y-4,x+4,y+4), fill=colors[agent], width=2)
                        map_draw.line((x-4,y+4,x+4,y-4), fill=colors[agent], width=2)
                for j in range(c.n_targets):
                    if not visible[j]:
                        continue
                    x, y = xy(w.targets[j])
                    dead = state['destroyed'][j]
                    color = '#596576' if dead else target_colors[int(w.target_type[j])]
                    map_draw.ellipse((x-9,y-9,x+9,y+9), fill='#101924', outline=color, width=3)
                    map_label((x+12,y-13), f'T{j+1}', 15, color)
                    map_label((x+12,y+2), f'F{w.target_formation[j]+1}  life {state["target_life"][j]}', 12, color)
                    if j in discoveries and 0 <= step-discoveries[j] < 5:
                        map_draw.ellipse((x-18,y-18,x+18,y+18), outline='#69ddff', width=2)
                map_box = tuple(value * render_scale for value in (30, 90, 610, 670))
                im.paste(map_image.crop(map_box), map_box[:2])
                d.rectangle((30, 90, 610, 670), outline='#354557', width=2)
                label((640, 110), f'TIME   {step:03d} / 100 s', 19)
                label((640, 150), f'DAMAGE   {state["metrics"]["score"]:.1f}', 26)
                label((640, 190), f'B = {w.formation_one_initial_score}', 19)
                label((640, 224), 'SUCCESS' if state['metrics']['mission_success'] else 'IN PROGRESS' if step<100 else 'TIME LIMIT',
                      19, '#5ce1ae' if state['metrics']['mission_success'] else '#ffcc66')
                label((640, 280), 'DRONE STATUS', 19)
                for i in range(c.n_agents):
                    parts = np.flatnonzero(np.array(state['strike_participants'])[:,i])
                    assigned = np.flatnonzero(np.array(state['target_assignment'])[:,i])
                    status = 'expended' if not state['agent_active'][i] else (
                        f'striking T{parts[0]+1}' if len(parts) else
                        f'approaching T{assigned[0]+1}' if len(assigned) else 'selecting')
                    # Avoid disclosing hidden formation targets in the HUD.
                    if len(assigned) and not visible[assigned[0]] and state['agent_active'][i]:
                        status = 'en route'
                    label((640, 316+30*i), f'D{i+1}   {status}', 15, colors[i])
                label((640, 457), 'TARGET TYPES', 19)
                for j in range(c.n_targets):
                    known = bool(visible[j])
                    label((640, 490 + 35*j),
                          f'T{j+1}  TYPE {int(w.target_type[j])}' if known else f'T{j+1}  TYPE ?',
                          26, target_colors[int(w.target_type[j])] if known else '#596576')
                d.rectangle((30, 692, 930, 700), fill='#293747')
                if step:
                    d.rectangle((30, 692, 30+900*step/100, 700), fill='#56c8ff')
                if step in (0, 50, 100):
                    im.save(out / f'case{case}_{name}_{step:03d}.png')
                for _ in range(3 if step < 100 else 15):
                    process.stdin.write(im.tobytes())
            process.stdin.close()
            if process.wait() != 0:
                if render_path != path:
                    render_path.unlink(missing_ok=True)
                raise RuntimeError(f'ffmpeg failed: {render_path}')
            if render_path != path:
                render_path.replace(path)
            row = dict(case=case, policy=title, seed=seed, video=str(path),
                       damage=w.score, discoveries=discoveries, metrics=w.metrics())
            rows.append(row)
            print(json.dumps(row), flush=True)
    (out / 'manifest.json').write_text(json.dumps(dict(
        disclosure='Baselines retain full information and original F2 choices; drones temporarily approach F1 until each remembered F2 target is revealed.',
        videos=rows), indent=2))


if __name__ == '__main__':
    main()
