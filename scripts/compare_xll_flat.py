"""Compare Isaac Sim and MuJoCo XLL joint-position recordings."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


WORKSPACE = Path(__file__).resolve().parents[2]
DEFAULT_ISAAC = WORKSPACE / "AME_Locomotion/outputs/xll_flat_isaac/joint_positions.csv"
DEFAULT_MUJOCO = WORKSPACE / "Mglf_sar/outputs/xll_flat_mujoco/joint_positions.csv"


def read_recording(path: Path):
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or reader.fieldnames[0] != "time_s":
            raise ValueError(f"Expected time_s as the first column: {path}")
        joints = reader.fieldnames[1:]
        if not joints or len(joints) != len(set(joints)):
            raise ValueError(f"Missing or duplicate joint columns: {path}")
        rows = list(reader)
    if len(rows) < 2:
        raise ValueError(f"Need at least two samples: {path}")
    try:
        times = np.array([float(row["time_s"]) for row in rows], dtype=np.float64)
        positions = np.array([[float(row[name]) for name in joints] for row in rows], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid numeric value in {path}: {exc}") from exc
    if not np.isfinite(times).all() or not np.isfinite(positions).all():
        raise ValueError(f"Non-finite value in {path}")
    if np.any(np.diff(times) <= 0):
        raise ValueError(f"Time values must be strictly increasing: {path}")
    return times, joints, positions


def plot_joints(path, times, joints, first, second=None, *, error=False, rmse=None):
    fig, axes = plt.subplots(4, 3, figsize=(16, 11), sharex=True)
    for index, (axis, joint) in enumerate(zip(axes.flat, joints)):
        axis.plot(times, first[:, index], label="Isaac Sim" if not error else "Isaac - MuJoCo", linewidth=1)
        if second is not None:
            axis.plot(times, second[:, index], label="MuJoCo", linewidth=1, alpha=0.85)
        if error:
            axis.axhline(0, color="black", linewidth=0.6, alpha=0.5)
        axis.set_title(f"{joint} | RMSE {rmse[index]:.4f} rad" if rmse is not None else joint)
        axis.set_ylabel("Error (rad)" if error else "Position (rad)")
        axis.grid(True, alpha=0.25)
    for axis in axes[-1]:
        axis.set_xlabel("Time (s)")
    axes[0, 0].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaac", type=Path, default=DEFAULT_ISAAC, help="Isaac joint_positions.csv")
    parser.add_argument("--mujoco", type=Path, default=DEFAULT_MUJOCO, help="MuJoCo joint_positions.csv")
    parser.add_argument("--output-dir", type=Path, default=WORKSPACE / "AME_Locomotion/outputs/xll_flat_comparison")
    args = parser.parse_args()

    isaac_time, joints, isaac_pos = read_recording(args.isaac)
    mujoco_time, mujoco_joints, mujoco_pos = read_recording(args.mujoco)
    if set(joints) != set(mujoco_joints):
        raise ValueError("Joint names differ between recordings")
    mujoco_pos = mujoco_pos[:, [mujoco_joints.index(joint) for joint in joints]]
    tolerance = 1e-3
    valid = (isaac_time >= mujoco_time[0] - tolerance) & (isaac_time <= mujoco_time[-1] + tolerance)
    times = isaac_time[valid]
    isaac_pos = isaac_pos[valid]
    if len(times) < 2:
        raise ValueError("Recordings have no overlapping time interval")
    aligned_mujoco = np.column_stack(
        [np.interp(times, mujoco_time, mujoco_pos[:, index]) for index in range(len(joints))]
    )
    errors = isaac_pos - aligned_mujoco
    rmse = np.sqrt(np.mean(errors**2, axis=0))
    overall_rmse = float(np.sqrt(np.mean(errors**2)))
    rmse_over_time = np.sqrt(np.mean(errors**2, axis=1))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "joint_errors.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["time_s", *joints, "rmse_all_joints_rad"])
        writer.writerows([time, *row, total] for time, row, total in zip(times, errors, rmse_over_time))
    with (args.output_dir / "rmse_by_joint.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["joint_name", "rmse_rad"])
        writer.writerows(zip(joints, rmse))
        writer.writerow(["all_joints", overall_rmse])
    summary = {
        "isaac_csv": str(args.isaac.resolve()),
        "mujoco_csv": str(args.mujoco.resolve()),
        "error_definition": "Isaac Sim minus MuJoCo, radians",
        "alignment": "MuJoCo linearly interpolated onto Isaac Sim timestamps",
        "samples": len(times),
        "time_start_s": float(times[0]),
        "time_end_s": float(times[-1]),
        "overall_rmse_rad": overall_rmse,
        "rmse_by_joint_rad": dict(zip(joints, map(float, rmse))),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_joints(args.output_dir / "joint_comparison.png", times, joints, isaac_pos, aligned_mujoco)
    plot_joints(args.output_dir / "joint_errors.png", times, joints, errors, error=True, rmse=rmse)
    fig, axis = plt.subplots(figsize=(12, 5))
    bars = axis.bar(joints, rmse)
    axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    axis.set_ylabel("RMSE (rad)")
    axis.set_ylim(0, float(rmse.max()) * 1.18)
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    axis.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(args.output_dir / "rmse_by_joint.png", dpi=150)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(12, 4))
    axis.plot(times, rmse_over_time, linewidth=1)
    axis.set(xlabel="Time (s)", ylabel="RMSE across 12 joints (rad)")
    axis.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output_dir / "rmse_over_time.png", dpi=150)
    plt.close(fig)

    print(f"Compared {len(times)} samples from {times[0]:.3f} to {times[-1]:.3f} s")
    for joint, value in zip(joints, rmse):
        print(f"{joint:20s} RMSE = {value:.6f} rad")
    print(f"All joints            RMSE = {overall_rmse:.6f} rad")
    print(f"Saved comparison to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
