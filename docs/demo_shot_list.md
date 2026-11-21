# Demo video: shot list

Target length: **6–8 min**. Record with OBS on Windows (WSLg windows are normal Windows
windows). Speed up the long waits 2× when editing, and say in a caption that you did.

## Screen layout (same for every shot)
```
+----------------------------+----------------------------+
|                            |                            |
|   Gazebo (top view)        |   RViz (top-down, odom)    |
|                            |                            |
+----------------------------+----------------------------+
|  Terminal A: ros2 launch ...   |  Terminal B: teleop / service calls |
+--------------------------------+-------------------------------------+
```
- Gazebo: left half. Our worlds open with a top-down camera; zoom so the whole path fits.
- RViz: right half (`config/mppi.rviz` is already top-down). Show the Displays panel once in the RViz legend shot, then hide it.
- Terminals: a thin strip at the bottom, large font (≥ 14 pt) so the log lines are readable.

## Before every take
```bash
pkill -f '^gz sim'          # leftover Gazebo servers from the previous take
source /opt/ros/jazzy/setup.bash && source ~/ros2_ws/install/setup.bash && export TURTLEBOT3_MODEL=burger
```
The 10 s start delay gives you time to arrange the windows. If you need longer, use `start_delay:=20`.

---

## Shots
| # | Shot | Length | Commands | What to show / say |
|---|---|---|---|---|
| 0 | Title card | 5 s | – | "MPPI path tracking with obstacle avoidance, TurtleBot3, ROS 2 Jazzy. ataraxy" |
| 1 | The problem | 20 s | – (voice over a Case 1 screenshot) | "The reference tracker (Regulated Pure Pursuit) stops at obstacles. I replaced it with an MPPI controller that goes around them and returns to the path." |
| 2 | **Recording a path** (task 1) | 60 s | A: `ros2 launch mppi_nav record.launch.py output:=demo_rec.csv`  B: `ros2 run turtlebot3_teleop teleop_keyboard` | Drive a gentle curve. The green line grows in RViz. Ctrl+C in A shows `Saved N points … spacing 0.100/0.100/0.100 m`. Then `head paths/demo_rec.csv`: the `x,y` header and 0.1 m steps. |
| 3 | **Replay the recording** (task 3) | 40 s | A: `ros2 launch mppi_nav mppi.launch.py path:=demo_rec.csv` | "One launch file with the CSV as an argument; the robot starts tracking by itself." The red line lies on the green one. End on `Goal reached`. |
| 4 | RViz legend | 30 s | (during shot 3 or 5) | Point at each element: green = recorded, red = driven, blue = lidar obstacles, grey = sampled futures, yellow = chosen plan, magenta = progress point. One sentence on MPPI: "1000 futures every 50 ms, weighted by cost." |
| 5 | **Case 1**: thin wall | 50 s | `ros2 launch mppi_nav mppi.launch.py case:=case1` | The robot leaves the path before the wall, goes round the short end, and comes back onto the green line. |
| 6 | **Case 2**: symmetric T | 50 s | `case:=case2` | It leaves the path before the T's leg, runs alongside it, rounds the bar's end and rejoins. (Either side is fine; it picks one.) |
| 7 | **Case 3**: long box | 50 s | `case:=case3` | A long detour **without stopping** ("the progress term stops it freezing"), then rejoins. |
| 8 | **Case 4**: self-crossing path | 60 s | `case:=case4` | Zoom on the crossing: it goes *through* the crossing **in order**, around the loop, then out. "A monotonic progress index means it never jumps to the other pass." |
| 9 | **Dynamic obstacles** (task 5) | 60–90 s | A: `ros2 launch mppi_nav mppi.launch.py case:=loop spawn:=manual`  B: `ros2 service call /drop_obstacle std_srvs/srv/Trigger` | Call B while the robot is on a **straight**: a box appears about 1 m ahead in Gazebo, the blue points appear in RViz, and the robot swerves and rejoins. Drop 2–3 boxes over the run. (Or `case:=dynamic` for automatic drops.) |
| 10 | Results | 30 s | show `README.md` / `results/gazebo_summary.md`, `results/gz_case*.png` | "Every case 3/3 runs, 0 collisions, measured with Gazebo ground truth." Optionally the terminal output of `python3 -m pytest -q test` (32 passed). |
| 11 | Close | 10 s | – | Repo link. "Code, design decisions and the problems I hit are in the README." |

## Tips
- Avoid dropping boxes right at the loop crossing or on the loop curve (see Known limitations in the README). Straights look best.
- If the grey sample lines look too busy for the video, set `num_vis_samples: 5` in `config/mppi_params.yaml` (display only).
- If the robot ever stutters with GUI + RViz + OBS all running, close other apps. MPPI needs about 30–50 ms per step.
- Keep one terminal visible during each case. The start countdown and `Goal reached in … s` lines are good on-screen proof.
