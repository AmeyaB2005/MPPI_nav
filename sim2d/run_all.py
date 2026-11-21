"""Run every offline scenario closed-loop and print a results table.

    python3 sim2d/run_all.py                    # all cases
    python3 sim2d/run_all.py --cases case3 case4

Each case: robot starts on the first path point facing along the path, the
simulated lidar feeds mppi_core at 5 Hz, MPPI runs at control_rate (20 Hz).
Outputs go to sim2d/results/: one PNG per case (recorded path green, traversed
red, obstacles blue) and metrics.csv.
"""
from __future__ import annotations

import argparse
import copy
import csv
import os
import sys
from typing import Dict, List

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mppi_nav.mppi_core import MPPIConfig, MPPIController  # noqa: E402
from mppi_nav.utils import load_mppi_params, scan_to_points  # noqa: E402
from sim2d.scenarios import SPACING, Scenario, scenarios  # noqa: E402
from sim2d.world import Box, LidarModel, World, step_unicycle  # noqa: E402

CLEAR_DIST = 1.0      # [m] cross-track error is measured only this far from every obstacle
COVER_DIST = 0.15     # [m] a path point counts as "visited" if the robot came this close
MIN_SPEED = 0.08      # [m/s] time limit = path length / MIN_SPEED + TIME_MARGIN
TIME_MARGIN = 30.0    # [s]


def point_to_polyline(p: np.ndarray, path: np.ndarray) -> float:
    """Distance from point p to the polyline `path` (for the cross-track metric)."""
    a, b = path[:-1], path[1:]
    ab = b - a
    u = np.clip(((p - a) * ab).sum(1) / np.maximum((ab ** 2).sum(1), 1e-12), 0, 1)
    return float(np.linalg.norm(a + u[:, None] * ab - p, axis=1).min())


def run_case(sc: Scenario, params: dict, seed: int) -> Dict:
    """Simulate one scenario. Returns metrics plus the data needed for the plot."""
    cfg = MPPIConfig.from_dict(params)
    cfg.seed = seed
    rng = np.random.default_rng(seed + 1)
    world = World(copy.deepcopy(sc.boxes))
    lidar = LidarModel()
    ctrl = MPPIController(cfg, sc.path)

    path = sc.path
    state = np.array([*path[0], np.arctan2(*(path[1] - path[0])[::-1])])
    control_dt = 1.0 / cfg.control_rate
    length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
    t, t_max, next_scan = 0.0, length / MIN_SPEED + TIME_MARGIN, 0.0
    pending = list(sc.drops)
    traj: List[np.ndarray] = [state.copy()]
    clearance: List[float] = []
    compute_ms: List[float] = []
    collisions = recoveries = 0
    done = False

    while t < t_max:
        # Dynamic obstacles: drop a box ON the path 0.8-1.2 m ahead of the robot
        # (measured along the path, so it really blocks the route on curves too).
        for drop in [d for d in pending if ctrl.progress_idx * SPACING >= d.at_s]:
            pending.remove(drop)
            ahead, lat = rng.uniform(*drop.ahead), rng.uniform(*drop.lateral)
            i = min(ctrl.progress_idx + int(round(ahead / SPACING)), len(path) - 2)
            yaw = np.arctan2(*(path[i + 1] - path[i])[::-1])
            cx, cy = path[i] + lat * np.array([-np.sin(yaw), np.cos(yaw)])
            world.add_box(Box(cx=cx, cy=cy, sx=(sz := rng.uniform(*drop.size)), sy=sz,
                              yaw=yaw, t_added=t))

        # Lidar at its own (slower) rate, converted exactly as the ROS node will do it.
        if t >= next_scan:
            sx = state[0] + lidar.offset_x * np.cos(state[2])
            sy = state[1] + lidar.offset_x * np.sin(state[2])
            ranges = world.raycast((sx, sy), state[2], lidar, rng)
            pts = scan_to_points(ranges, lidar.angle_min, lidar.angle_increment,
                                 lidar.range_min, lidar.range_max, (sx, sy, state[2]))
            ctrl.set_obstacles(pts, state[:2])
            next_scan += 1.0 / lidar.rate

        res = ctrl.compute(state)
        if res.goal_reached:
            done = True
            break
        compute_ms.append(res.compute_ms)
        recoveries += res.all_collide
        state = step_unicycle(state, res.v, res.w, control_dt)
        t += control_dt
        traj.append(state.copy())
        d = world.distance(state[:2])
        clearance.append(d - cfg.robot_radius)
        collisions += d < cfg.robot_radius

    traj_a = np.array(traj)
    # Cross-track error, only where the robot is well clear of every obstacle.
    clear_mask = np.array([world.distance(p) > CLEAR_DIST for p in traj_a[:, :2]])
    cte = np.array([point_to_polyline(p, path) for p in traj_a[clear_mask, :2]])
    # Coverage: path points away from obstacles that the robot actually passed close to.
    free_pts = np.array([p for p in path if world.distance(p) > cfg.inflation_radius])
    visited = [np.linalg.norm(traj_a[:, :2] - p, axis=1).min() < COVER_DIST for p in free_pts]
    return {
        'case': sc.name,
        'done': done,
        'time_s': round(t, 1),
        'min_clear_m': round(min(clearance), 3) if world.boxes and clearance else float('nan'),
        'cte_mean_cm': round(100 * cte.mean(), 1) if len(cte) else float('nan'),
        'cte_max_cm': round(100 * cte.max(), 1) if len(cte) else float('nan'),
        'coverage_pct': round(100 * float(np.mean(visited)), 1),
        'collisions': int(collisions),
        'recoveries': int(recoveries),
        'mppi_ms_mean': round(float(np.mean(compute_ms)), 1),
        'mppi_ms_max': round(float(np.max(compute_ms)), 1),
        '_traj': traj_a, '_world': world, '_sc': sc,
    }


def plot_case(r: Dict, out_png: str) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    sc, traj, world = r['_sc'], r['_traj'], r['_world']
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(sc.path[:, 0], sc.path[:, 1], '-', color='limegreen', lw=3, label='recorded path')
    ax.plot(traj[:, 0], traj[:, 1], '-', color='red', lw=1.5, label='traversed path')
    for i, b in enumerate(world.boxes):
        ax.add_patch(Polygon(b.corners(), closed=True, color='royalblue', alpha=0.8,
                             label='obstacle' if i == 0 else None))
        if b.t_added > 0:
            ax.annotate(f'dropped t={b.t_added:.1f}s', (b.cx, b.cy + b.sy / 2 + 0.05),
                        ha='center', fontsize=8, color='royalblue')
    ax.plot(*sc.path[0], 'ko', ms=6, label='start')
    ax.plot(*sc.path[-1], 'k*', ms=12, label='goal')
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize=8)
    status = 'GOAL REACHED' if r['done'] else 'NOT COMPLETED'
    ax.set_title(f"{sc.name}: {sc.description} | {status} in {r['time_s']} s | "
                 f"collisions {r['collisions']} | min clearance {r['min_clear_m']} m", fontsize=9)
    fig.savefig(out_png, dpi=110, bbox_inches='tight')
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cases', nargs='*', default=None, help='subset of cases to run')
    parser.add_argument('--params', default=os.path.join(ROOT, 'config', 'mppi_params.yaml'))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', default=os.path.join(ROOT, 'sim2d', 'results'))
    args = parser.parse_args()

    params = load_mppi_params(args.params)
    all_sc = scenarios()
    names = args.cases or list(all_sc)
    os.makedirs(args.out, exist_ok=True)

    results = []
    for name in names:
        r = run_case(all_sc[name], params, args.seed)
        plot_case(r, os.path.join(args.out, f'{name}.png'))
        results.append({k: v for k, v in r.items() if not k.startswith('_')})
        print(f'  {name:8s} done={r["done"]} t={r["time_s"]}s collisions={r["collisions"]}',
              flush=True)

    cols = list(results[0])
    widths = [max(len(c), *(len(str(r[c])) for r in results)) for c in cols]
    print('\n' + ' | '.join(c.ljust(w) for c, w in zip(cols, widths)))
    print('-+-'.join('-' * w for w in widths))
    for r in results:
        print(' | '.join(str(r[c]).ljust(w) for c, w in zip(cols, widths)))
    with open(os.path.join(args.out, 'metrics.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(results)
    ok = all(r['done'] and r['collisions'] == 0 for r in results)
    print(f'\nALL PASSED: {ok}   (PNGs + metrics.csv in {args.out})')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
