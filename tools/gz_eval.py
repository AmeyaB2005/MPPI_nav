#!/usr/bin/env python3
"""Gazebo evaluation: run every case N times headless and measure it with ground truth.

    source /opt/ros/jazzy/setup.bash && source <your_ws>/install/setup.bash
    export TURTLEBOT3_MODEL=burger
    python3 tools/gz_eval.py                       # all cases, 3 runs each
    python3 tools/gz_eval.py --cases case1 case2 --runs 1

For each run: launches mppi.launch.py (gui:=false, use_rviz:=false), records
/ground_truth (Gazebo's true robot pose, never used by the controller) until the
tracker logs "Goal reached" or a timeout, then shuts everything down.

Metrics (all from ground truth, robot radius 0.105 m):
  done            goal reached before the timeout
  time_s          tracking time (sim time) from the tracker's log
  min_clear_m     closest distance robot BODY <-> obstacle surface (static + dropped boxes)
  collisions      ground-truth samples where the body overlaps an obstacle (must be 0)
  cte_mean/max    distance to the recorded path, only where > CLEAR_DIST from every obstacle
  coverage_pct    recorded path points the robot passed within COVER_DIST
  goal_err_cm     final distance to the last path point
  mppi_ms_*       MPPI step time from the tracker's log
Outputs: results/gazebo_metrics.csv, results/gazebo_metrics.md, results/gz_<case>.png
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import signal
import subprocess
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG)
from mppi_nav.utils import load_path_csv  # noqa: E402
from sim2d.run_all import point_to_polyline  # noqa: E402
from sim2d.scenarios import scenarios  # noqa: E402

ROBOT_RADIUS = 0.105   # [m]
CLEAR_DIST = 1.0       # [m] cross-track error only counted this far from every obstacle
COVER_DIST = 0.15      # [m]
MIN_SPEED = 0.08       # [m/s] timeout = path length / MIN_SPEED + TIMEOUT_MARGIN
TIMEOUT_MARGIN = 40.0  # [s]
SHUTDOWN_WAIT = 15.0   # [s]

# case name -> (launch arguments, path CSV)
CASES: Dict[str, Tuple[List[str], str]] = {
    'case1': (['case:=case1'], 'case1.csv'),
    'case2': (['case:=case2'], 'case2.csv'),
    'case3': (['case:=case3'], 'case3.csv'),
    'case4': (['case:=case4'], 'case4.csv'),
    'case4_recorded': (['path:=case4_recorded.csv'], 'case4_recorded.csv'),
    'loop': (['case:=loop'], 'loop.csv'),
    'dynamic': (['case:=dynamic'], 'dynamic.csv'),
}
Box = Tuple[float, float, float, float, float]     # cx, cy, sx, sy, yaw


def box_distance(p: np.ndarray, box: Box) -> np.ndarray:
    """Distance from points p (N, 2) to the surface of a rectangle (0 inside)."""
    cx, cy, sx, sy, yaw = box
    c, s = np.cos(yaw), np.sin(yaw)
    d = p - [cx, cy]
    lx, ly = c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]
    return np.hypot(np.maximum(np.abs(lx) - sx / 2, 0), np.maximum(np.abs(ly) - sy / 2, 0))


def dropped_boxes(log: str, path: np.ndarray) -> List[Box]:
    """Boxes the obstacle spawner dropped, parsed from its log (yaw = path direction)."""
    boxes = []
    for size, x, y in re.findall(r'Dropped \S+: ([\d.]+) m box.*?at \(([-\d.]+), ([-\d.]+)\)', log):
        size, x, y = float(size), float(x), float(y)
        i = int(np.argmin(np.linalg.norm(path - [x, y], axis=1)))
        d = path[min(i + 1, len(path) - 1)] - path[max(i - 1, 0)]
        boxes.append((x, y, size, size, float(np.arctan2(d[1], d[0]))))
    return boxes


class GroundTruth:
    """Collects /ground_truth (x, y) in a background rclpy node."""

    def __init__(self) -> None:
        import rclpy
        from nav_msgs.msg import Odometry
        self.rclpy = rclpy
        rclpy.init()
        self.node = rclpy.create_node('gz_eval_ground_truth')
        self.rows: List[Tuple[float, float]] = []
        self.node.create_subscription(
            Odometry, '/ground_truth',
            lambda m: self.rows.append((m.pose.pose.position.x, m.pose.pose.position.y)), 50)

    def spin_for(self, seconds: float) -> None:
        end = time.time() + seconds
        while time.time() < end:
            self.rclpy.spin_once(self.node, timeout_sec=0.05)

    def take(self) -> np.ndarray:
        rows, self.rows = self.rows, []
        return np.array(rows) if rows else np.zeros((0, 2))


def stop_launch(proc: subprocess.Popen) -> None:
    """SIGINT the whole launch process group, then make sure nothing is left."""
    try:
        os.killpg(proc.pid, signal.SIGINT)
        proc.wait(timeout=SHUTDOWN_WAIT)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    # Gazebo's ruby wrapper can leave its children behind: anchored patterns only.
    subprocess.run(['pkill', '-9', '-f', '^gz sim'], check=False)
    time.sleep(2.0)


def run_once(name: str, run: int, gt: GroundTruth, out_dir: str) -> Dict:
    """One headless Gazebo run of a case; returns its metrics."""
    launch_args, csv_name = CASES[name]
    path = load_path_csv(os.path.join(PKG, 'paths', csv_name))
    length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
    timeout = length / MIN_SPEED + TIMEOUT_MARGIN
    log_file = os.path.join(out_dir, 'logs', f'{name}_run{run}.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    cmd = ['ros2', 'launch', 'mppi_nav', 'mppi.launch.py', *launch_args, 'gui:=false',
           'use_rviz:=false', 'start_delay:=2']
    with open(log_file, 'w') as log_fh:
        proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT,
                                start_new_session=True)
    gt.take()
    t0 = time.time()
    done = False
    while time.time() - t0 < timeout:
        gt.spin_for(1.0)
        text = open(log_file, errors='replace').read()
        if 'Goal reached' in text:
            done = True
            gt.spin_for(1.0)
            break
        if 'Traceback' in text or proc.poll() is not None:     # crashed / launch exited
            break
    truth = gt.take()
    stop_launch(proc)

    log = re.sub(r'\x1b\[[0-9;]*m', '', open(log_file, errors='replace').read())
    m = re.search(r'Goal reached in ([\d.]+) s .*?mean ([\d.]+) ms, max ([\d.]+) ms', log)
    static = scenarios()[name].boxes if name in scenarios() else []
    boxes = [(b.cx, b.cy, b.sx, b.sy, b.yaw) for b in static]
    boxes += dropped_boxes(log, path)
    np.savetxt(os.path.join(out_dir, 'logs', f'{name}_run{run}_truth.csv'), truth,
               delimiter=',', fmt='%.4f')

    res = {'case': name, 'run': run, 'done': done,
           'time_s': float(m.group(1)) if m else float('nan'),
           'boxes': len(boxes), 'min_clear_m': float('nan'), 'collisions': 0,
           'cte_mean_cm': float('nan'), 'cte_max_cm': float('nan'), 'coverage_pct': 0.0,
           'goal_err_cm': float('nan'),
           'mppi_ms_mean': float(m.group(2)) if m else float('nan'),
           'mppi_ms_max': float(m.group(3)) if m else float('nan')}
    if len(truth) < 10:
        return {**res, '_truth': truth, '_boxes': boxes, '_path': path}
    d_obs = (np.min([box_distance(truth, b) for b in boxes], axis=0) if boxes
             else np.full(len(truth), np.inf))
    clear = d_obs > CLEAR_DIST
    cte = np.array([point_to_polyline(p, path) for p in truth[clear]])
    res.update({
        'min_clear_m': round(float(d_obs.min() - ROBOT_RADIUS), 3) if boxes else float('nan'),
        'collisions': int((d_obs < ROBOT_RADIUS).sum()),
        'cte_mean_cm': round(100 * float(cte.mean()), 1) if len(cte) else float('nan'),
        'cte_max_cm': round(100 * float(cte.max()), 1) if len(cte) else float('nan'),
        'coverage_pct': round(100 * float(np.mean(
            [np.linalg.norm(truth - p, axis=1).min() < COVER_DIST
             for p in path if not boxes or min(box_distance(p[None], b)[0] for b in boxes) > 0.5])), 1),
        'goal_err_cm': round(100 * float(np.linalg.norm(truth[-1] - path[-1])), 1),
    })
    return {**res, '_truth': truth, '_boxes': boxes, '_path': path}


def plot_case(name: str, runs: List[Dict], out_png: str) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    fig, ax = plt.subplots(figsize=(8, 5))
    path = runs[0]['_path']
    ax.plot(path[:, 0], path[:, 1], color='limegreen', lw=4, label='recorded path')
    shown = set()
    for k, r in enumerate(runs):
        if len(r['_truth']):
            ax.plot(r['_truth'][:, 0], r['_truth'][:, 1], '-', color='red', alpha=0.9 - 0.25 * k,
                    lw=1.5, label=f'driven, run {r["run"]} (ground truth)')
        for b in r['_boxes']:
            if b in shown:
                continue
            shown.add(b)
            cx, cy, sx, sy, yaw = b
            c, s = np.cos(yaw), np.sin(yaw)
            corners = np.array([[-sx, -sy], [sx, -sy], [sx, sy], [-sx, sy]]) / 2 @ \
                np.array([[c, s], [-s, c]]) + [cx, cy]
            ax.add_patch(Polygon(corners, color='royalblue', alpha=0.8,
                                 label='obstacle' if len(shown) == 1 else None))
    ax.plot(*path[0], 'ko', label='start')
    ax.plot(*path[-1], 'k*', ms=12, label='goal')
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc='upper left', bbox_to_anchor=(1.01, 1.0))
    ok = sum(r['done'] for r in runs)
    ax.set_title(f'Gazebo {name}: {ok}/{len(runs)} runs reached the goal', fontsize=10)
    fig.savefig(out_png, dpi=110, bbox_inches='tight')
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cases', nargs='*', default=list(CASES))
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--out', default=os.path.join(PKG, 'results'))
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    gt = GroundTruth()
    results: List[Dict] = []
    for name in args.cases:
        runs = []
        for run in range(1, args.runs + 1):
            r = run_once(name, run, gt, args.out)
            runs.append(r)
            print(f'{name} run {run}: done={r["done"]} t={r["time_s"]}s '
                  f'clear={r["min_clear_m"]} collisions={r["collisions"]} '
                  f'cte={r["cte_mean_cm"]}/{r["cte_max_cm"]}cm cov={r["coverage_pct"]}%', flush=True)
        plot_case(name, runs, os.path.join(args.out, f'gz_{name}.png'))
        results += [{k: v for k, v in r.items() if not k.startswith('_')} for r in runs]

    cols = list(results[0])
    with open(os.path.join(args.out, 'gazebo_metrics.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(results)
    with open(os.path.join(args.out, 'gazebo_metrics.md'), 'w') as f:
        f.write('| ' + ' | '.join(cols) + ' |\n|' + '---|' * len(cols) + '\n')
        for r in results:
            f.write('| ' + ' | '.join(str(r[c]) for c in cols) + ' |\n')
    print(open(os.path.join(args.out, 'gazebo_metrics.md')).read())
    ok = all(r['done'] and r['collisions'] == 0 for r in results)
    print(f'ALL RUNS OK: {ok}  (results in {args.out})')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
