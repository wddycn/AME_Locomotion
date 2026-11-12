"""Record XLL joint positions during a fixed-command Isaac Sim flat-ground rollout."""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

# AME ships its own RSL-RL fork; other workspace projects can shadow it.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "rsl_rl"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, default=Path("/home/mglf/rc/AME_Locomotion/logs/rsl_rl/xll_ame/2026-10-20_12-20-19/model_14400.pt"))
parser.add_argument("--duration", type=float, default=10.0, help="Simulation seconds")
parser.add_argument("--forward-velocity", type=float, default=0.8, help="Forward command in m/s")
parser.add_argument("--output-dir", type=Path, default=Path("xll_flat_isaac"))
parser.add_argument("--task", default="AME-XLL-Stage2-Play-v0")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.duration <= 0:
    parser.error("--duration must be positive")
if not args.checkpoint.is_file():
    parser.error(f"checkpoint does not exist: {args.checkpoint}")
app = AppLauncher(args).app

import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg
import ame_locomotion.tasks  # noqa: F401
from ame_locomotion.tasks.manager_based.ame_locomotion.assets.robots.xll import XLL_JOINT_NAMES
from ame_locomotion.tasks.manager_based.ame_locomotion.agents.xll_ame_rsl_rl_ppo_cfg import XLLAMEPPORunnerCfg


def main():
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[record] Output directory: {output_dir}", flush=True)
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    cfg.scene.terrain.max_init_terrain_level = None
    cfg.curriculum.terrain_levels = None
    cfg.commands.base_velocity.heading_command = False
    cfg.commands.base_velocity.rel_heading_envs = 0.0
    cfg.commands.base_velocity.rel_standing_envs = 0.0
    cfg.commands.base_velocity.ranges.lin_vel_x = (args.forward_velocity, args.forward_velocity)
    cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.heading = (0.0, 0.0)
    cfg.episode_length_s = args.duration + 2.0
    cfg.observations.policy.enable_corruption = False
    cfg.observations.policy.height_scan.params["noise"] = False
    cfg.events.physics_material = None
    cfg.events.add_base_mass = None
    cfg.events.base_com = None
    cfg.events.base_external_force_torque = None
    cfg.events.push_robot = None
    cfg.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
    cfg.events.reset_base.params["velocity_range"] = {k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")}
    cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
    cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)

    env = RslRlVecEnvWrapper(gym.make(args.task, cfg=cfg), clip_actions=100.0)
    try:
        print(f"[record] Loading checkpoint: {args.checkpoint.resolve()}", flush=True)
        agent_cfg = XLLAMEPPORunnerCfg()
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=env.unwrapped.device)
        runner.load(str(args.checkpoint.resolve()))
        print("[record] Checkpoint loaded", flush=True)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        robot = env.unwrapped.scene["robot"]
        joint_ids, found_names = robot.find_joints(XLL_JOINT_NAMES, preserve_order=True)
        if found_names != XLL_JOINT_NAMES:
            raise RuntimeError(f"Unexpected joint order: {found_names}")
        dt = env.unwrapped.step_dt
        steps = math.ceil(args.duration / dt)
        rows = []
        obs = env.get_observations()
        rows.append([0.0, *robot.data.joint_pos[0, joint_ids].cpu().tolist()])
        stop_reason = "duration reached"
        sample_error = None
        with (output_dir / "joint_positions.csv").open("w", newline="", buffering=1) as file:
            writer = csv.writer(file)
            writer.writerow(["time_s", *XLL_JOINT_NAMES])
            writer.writerow(rows[0])
            print(f"[record] Sampling {steps} steps at dt={dt:.4f}s", flush=True)
            try:
                with torch.inference_mode():
                    for step in range(steps):
                        policy_output = policy(obs)
                        actions = policy_output[0] if isinstance(policy_output, tuple) else policy_output
                        if not isinstance(actions, torch.Tensor) or actions.shape != (1, len(XLL_JOINT_NAMES)):
                            raise RuntimeError(f"Unexpected policy action shape: {getattr(actions, 'shape', type(actions))}")
                        obs, _, dones, _ = env.step(actions)
                        if bool(dones[0]):
                            stop_reason = f"environment terminated at t={(step + 1) * dt:.3f}s"
                            print(f"[record] {stop_reason}; saving collected samples", flush=True)
                            break
                        row = [(step + 1) * dt, *robot.data.joint_pos[0, joint_ids].cpu().tolist()]
                        rows.append(row)
                        writer.writerow(row)
                        if (step + 1) % max(1, round(1.0 / dt)) == 0:
                            print(f"[record] t={(step + 1) * dt:.1f}s / {args.duration:.1f}s", flush=True)
            except Exception as exc:
                sample_error = exc
                stop_reason = f"sampling failed: {exc}"
            finally:
                file.flush()
                print(f"[record] CSV contains {len(rows)} samples: {output_dir / 'joint_positions.csv'}", flush=True)
        metadata = {"engine": "isaacsim", "task": args.task, "policy": str(args.checkpoint.resolve()),
                    "scene": "plane", "duration_s": args.duration, "sample_dt_s": dt,
                    "command": [args.forward_velocity, 0.0, 0.0], "joint_units": "rad", "joint_names": XLL_JOINT_NAMES,
                    "samples": len(rows), "stop_reason": stop_reason}
        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        fig, axes = plt.subplots(4, 3, figsize=(15, 10), sharex=True)
        for index, (axis, name) in enumerate(zip(axes.flat, XLL_JOINT_NAMES)):
            axis.plot([r[0] for r in rows], [r[index + 1] for r in rows], linewidth=1)
            axis.set_title(name)
            axis.set_ylabel("rad")
            axis.grid(True, alpha=0.3)
        for axis in axes[-1]:
            axis.set_xlabel("Time (s)")
        fig.tight_layout()
        fig.savefig(output_dir / "joint_positions.png", dpi=150)
        plt.close(fig)
        print(f"[record] Saved {len(rows)} samples to {output_dir}", flush=True)
        if sample_error is not None:
            raise sample_error
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
