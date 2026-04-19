import argparse
import os
import re
import numpy as np
import matplotlib.pyplot as plt

ROLLOUT_RE = re.compile(r"rollout\s+(\d+):\s+([-+eE0-9.]+)")
AVG_RE = re.compile(r"average return:\s+([-+eE0-9.]+)")


def parse_log(path):
    rollouts = []
    avg = None
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            m = ROLLOUT_RE.match(line)
            if m:
                rollouts.append(float(m.group(2)))
                continue
            m = AVG_RE.match(line)
            if m:
                avg = float(m.group(1))
    if avg is None and rollouts:
        avg = float(np.mean(rollouts))
    return rollouts, avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rlhf-root", required=True)
    parser.add_argument("--synthetic-log", required=True)
    parser.add_argument("--plot-out", required=True)
    args = parser.parse_args()

    syn_rollouts, syn_avg = parse_log(args.synthetic_log)
    if syn_avg is None:
        raise ValueError(f"Could not parse synthetic baseline log: {args.synthetic_log}")

    checkpoints = []
    rlhf_avgs = []
    rlhf_rollouts = {}

    for ckpt in range(100):
        log_path = os.path.join(args.rlhf_root, f"checkpoint{ckpt}", "log.txt")
        if not os.path.exists(log_path):
            continue
        rollouts, avg = parse_log(log_path)
        if avg is None:
            continue
        checkpoints.append(ckpt)
        rlhf_avgs.append(avg)
        rlhf_rollouts[ckpt] = rollouts

    if not checkpoints:
        raise ValueError("No RLHF checkpoint logs found.")

    exceed_ckpt = None
    exceed_rollouts = None
    for ckpt, avg in zip(checkpoints, rlhf_avgs):
        if avg > syn_avg:
            exceed_ckpt = ckpt
            exceed_rollouts = rlhf_rollouts[ckpt]
            break

    plt.figure(figsize=(8, 5))
    plt.plot(checkpoints, rlhf_avgs, marker="o", markersize=3, label="RLHF average return")
    plt.axhline(syn_avg, linestyle="--", label="synthetic_0 checkpoint9 baseline")
    if exceed_ckpt is not None:
        plt.scatter([exceed_ckpt], [np.mean(exceed_rollouts)], s=60)
        plt.annotate(
            f"first exceed: ckpt {exceed_ckpt}",
            (exceed_ckpt, np.mean(exceed_rollouts)),
            textcoords="offset points",
            xytext=(6, 6),
        )
    plt.xlabel("Checkpoint")
    plt.ylabel("Average return over 5 rollouts")
    plt.title("RLHF checkpoints vs synthetic baseline")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.plot_out, dpi=200)

    print(f"Synthetic baseline average return (checkpoint 9): {syn_avg:.4f}")
    print(f"Synthetic baseline rollout returns: {syn_rollouts}")
    print()

    if exceed_ckpt is None:
        print("No RLHF checkpoint exceeded the synthetic baseline.")
    else:
        print(f"First RLHF checkpoint exceeding synthetic baseline: {exceed_ckpt}")
        print(f"RLHF checkpoint {exceed_ckpt} rollout returns: {exceed_rollouts}")
        print(f"RLHF checkpoint {exceed_ckpt} average return: {np.mean(exceed_rollouts):.4f}")


if __name__ == "__main__":
    main()
