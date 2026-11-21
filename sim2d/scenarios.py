"""Test scenarios (paths + obstacles) for the offline simulator.

Obstacle layouts follow the assignment figure:
  case1   thin wall across the path, shorter way round is on the left (+y)
  case2   symmetric T lying on its side: the T's leg lies ON the path (the robot starts
          in line with it) and the cross bar is centred on the leg. The robot has to leave
          the path before the leg, drive alongside it, round one end of the bar, and rejoin
  case3   long box lying on the path -> long detour (right)
  case4   smooth self-crossing loop (same path as `loop`), NO obstacles: replay a
          complicated path in order, no shortcut at the crossing
  case4b  path with a loop that crosses itself (like the sketch), no obstacles
  dynamic gently curved path; boxes are dropped 0.8-1.2 m in front of the robot
  loop    smooth self-crossing loop (like the assignment sketch): straight run-in, one
          0.7 m-radius loop, clean 56 deg crossing, straight run-out; boxes dropped on
          the straights. Built from a curvature profile, so there are no sharp turns

Run `python3 sim2d/scenarios.py` to (re)write paths/<case>.csv and worlds/<case>.sdf.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mppi_nav.utils import resample_path, save_path_csv  # noqa: E402
from sim2d.world import Box  # noqa: E402

SPACING = 0.1  # [m] same as the path recorder


@dataclass
class DynamicDrop:
    """Drop a box on the path 0.8-1.2 m ahead of the robot once it has progressed `at_s` m."""

    at_s: float
    ahead: Tuple[float, float] = (0.8, 1.2)       # [m] ahead of the robot, along the path
    lateral: Tuple[float, float] = (-0.15, 0.15)  # [m] sideways offset from the path
    size: Tuple[float, float] = (0.25, 0.4)       # [m] square side length


@dataclass
class Scenario:
    name: str
    description: str
    path: np.ndarray
    boxes: List[Box] = field(default_factory=list)
    drops: List[DynamicDrop] = field(default_factory=list)


def straight(length: float) -> np.ndarray:
    return resample_path(np.array([[0.0, 0.0], [length, 0.0]]), SPACING)


def loop_path() -> np.ndarray:
    """Lead-in, one loop that crosses itself (a prolate trochoid), lead-out."""
    t = np.linspace(-np.pi, 3 * np.pi, 2000)
    x = 0.35 * t - 0.6 * np.sin(t)
    y = 0.6 * (1.0 - np.cos(t))
    return resample_path(np.column_stack((x, y)), SPACING)


def loop_from_curvature(radius: float = 0.7, ramp: float = 1.2, run_in: float = 2.5,
                        run_out: float = 2.5, turn_deg: float = 305.0) -> np.ndarray:
    """Smooth loop built from its curvature kappa(s) (s = distance along the path):

        straight (kappa = 0) -> kappa ramps up linearly to 1/radius -> constant arc
        -> ramps back to 0 -> straight.

    The heading is the integral of kappa, so the total turn is the area under kappa(s).
    It is set to turn_deg < 360 so the run-out leaves at an angle and crosses the run-in
    cleanly (305 deg -> a 56 deg crossing). The linear ramps mean the turn rate never
    jumps, and the tightest radius anywhere is `radius`.
    """
    ds = 0.005
    k = 1.0 / radius
    hold = np.radians(turn_deg) / k - ramp          # ramps contribute k*ramp/2 each
    pieces = [np.zeros(int(run_in / ds)),
              k * np.linspace(0.0, 1.0, int(ramp / ds)),
              np.full(int(hold / ds), k),
              k * np.linspace(1.0, 0.0, int(ramp / ds)),
              np.zeros(int(run_out / ds))]
    heading = np.cumsum(np.concatenate(pieces)) * ds
    xy = np.column_stack((np.cumsum(np.cos(heading)), np.cumsum(np.sin(heading)))) * ds
    return resample_path(np.vstack(([0.0, 0.0], xy)), SPACING)


def curved(length: float = 8.0, amplitude: float = 0.4) -> np.ndarray:
    x = np.linspace(0.0, length, 2000)
    return resample_path(np.column_stack((x, amplitude * np.sin(2 * np.pi * x / length))),
                         SPACING)


def scenarios() -> Dict[str, Scenario]:
    return {
        'case1': Scenario(
            'case1', 'Thin wall across the path',
            straight(6.0),
            boxes=[Box(cx=3.0, cy=-0.1, sx=0.05, sy=1.0)]),          # y from -0.6 to +0.4
        'case2': Scenario(
            'case2', 'Symmetric T, leg on the path',
            straight(6.0),
            boxes=[Box(cx=2.6, cy=0.0, sx=1.2, sy=0.05),             # T's leg ON the path, x 2.0-3.2
                   Box(cx=3.2, cy=0.0, sx=0.05, sy=1.4)]),            # cross bar centred on the
                                                                       # leg: y -0.7 to +0.7
        'case3': Scenario(
            'case3', 'Long box on the path',
            straight(6.0),
            boxes=[Box(cx=3.0, cy=0.05, sx=2.0, sy=0.3)]),            # x 2.0-4.0
        'case4': Scenario('case4', 'Smooth self-crossing loop', loop_from_curvature()),
        'case4b': Scenario('case4b', 'Loop with a self-crossing', loop_path()),
        'dynamic': Scenario(
            'dynamic', 'Boxes dropped in front of the robot',
            curved(),
            drops=[DynamicDrop(at_s=1.5), DynamicDrop(at_s=4.5)]),
        'loop': Scenario(
            'loop', 'Smooth self-crossing loop + dropped boxes',
            loop_from_curvature(),
            drops=[DynamicDrop(at_s=0.5), DynamicDrop(at_s=7.3)]),   # on the run-in / run-out
    }


def write_paths(out_dir: str) -> None:
    """Write every scenario path to <out_dir>/<name>.csv."""
    for name, sc in scenarios().items():
        print('wrote', save_path_csv(os.path.join(out_dir, f'{name}.csv'), sc.path))


# --------------------------------------------------------------------------- #
# Gazebo worlds, generated from the SAME boxes the offline simulator uses
# --------------------------------------------------------------------------- #
BOX_HEIGHT = 0.4          # [m] taller than the lidar (0.19 m above the ground)
PHYSICS_STEP = 0.004      # [s] 250 Hz. TB3's world uses 1 ms (1 kHz /clock), which costs the
                          # Python tracker a lot of CPU under use_sim_time; 4 ms is plenty
                          # for a 0.22 m/s differential-drive robot.


def box_sdf(name: str, b: Box) -> str:
    """Static blue box model."""
    size = f'{b.sx} {b.sy} {BOX_HEIGHT}'
    geom = f'<geometry><box><size>{size}</size></box></geometry>'
    return f"""
    <model name="{name}">
      <static>true</static>
      <pose>{b.cx} {b.cy} {BOX_HEIGHT / 2} 0 0 {b.yaw}</pose>
      <link name="link">
        <collision name="collision">{geom}</collision>
        <visual name="visual">{geom}
          <material><ambient>0.1 0.3 1 1</ambient><diffuse>0.1 0.3 1 1</diffuse></material>
        </visual>
      </link>
    </model>"""


def world_sdf(sc: Scenario) -> str:
    """Complete SDF world: the systems TurtleBot3 needs, ground, light, top-down camera, boxes."""
    xs, ys = sc.path[:, 0], sc.path[:, 1]
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    cam_z = max(4.0, 1.2 * max(xs.ptp(), ys.ptp()))
    boxes = ''.join(box_sdf(f'obstacle_{i}', b) for i, b in enumerate(sc.boxes))
    return f"""<?xml version="1.0"?>
<!-- Generated by sim2d/scenarios.py from scenario "{sc.name}": {sc.description}.
     Do not edit by hand: change the scenario and re-run `python3 sim2d/scenarios.py`. -->
<sdf version="1.8">
  <world name="default">
    <physics name="default_physics" type="ode">
      <max_step_size>{PHYSICS_STEP}</max_step_size>
      <real_time_update_rate>{1.0 / PHYSICS_STEP:.0f}</real_time_update_rate>
      <real_time_factor>1</real_time_factor>
    </physics>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>
    <plugin filename="gz-sim-imu-system" name="gz::sim::systems::Imu"/>

    <scene><shadows>false</shadows><grid>true</grid></scene>
    <gui fullscreen="0">
      <camera name="user_camera">
        <pose>{cx:.2f} {cy:.2f} {cam_z:.1f} 0 1.5707 0</pose>
      </camera>
    </gui>

    <light type="directional" name="sun">
      <cast_shadows>false</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <direction>-0.5 0.1 -0.9</direction>
    </light>

    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry><plane><normal>0 0 1</normal><size>100 100</size></plane></geometry>
        </collision>
        <visual name="visual">
          <geometry><plane><normal>0 0 1</normal><size>100 100</size></plane></geometry>
          <material><ambient>0.8 0.8 0.8 1</ambient><diffuse>0.8 0.8 0.8 1</diffuse></material>
        </visual>
      </link>
    </model>
{boxes}
  </world>
</sdf>
"""


def write_worlds(out_dir: str) -> None:
    """Write worlds/<name>.sdf for every scenario (no boxes = empty world)."""
    os.makedirs(out_dir, exist_ok=True)
    worlds = dict(scenarios())
    worlds['empty'] = Scenario('empty', 'No obstacles', straight(4.0))
    for name, sc in worlds.items():
        out = os.path.join(out_dir, f'{name}.sdf')
        with open(out, 'w', encoding='utf-8') as f:
            f.write(world_sdf(sc))
        print('wrote', out)


if __name__ == '__main__':
    pkg = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    write_paths(os.path.join(pkg, 'paths'))
    write_worlds(os.path.join(pkg, 'worlds'))
