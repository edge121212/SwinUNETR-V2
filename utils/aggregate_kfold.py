import argparse
import glob
import re
import statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--log_base", required=True)
    args = parser.parse_args()

    fold_files = sorted(glob.glob(f"runs/{args.log_base}_fold*_test.log"))
    if not fold_files:
        print(f"No fold logs found matching runs/{args.log_base}_fold*_test.log")
        return

    metrics: dict[str, list[float]] = {
        "Overall Mean Dice": [],
        "Overall PZ Dice": [],
        "Overall TZ Dice": [],
    }

    for f in fold_files:
        with open(f) as fh:
            txt = fh.read()
        for key in list(metrics.keys()):
            m = re.search(rf"{re.escape(key)}: ([0-9.]+)", txt)
            if m:
                metrics[key].append(float(m.group(1)))
        for m in re.finditer(r"Overall Class (\d+) Dice: ([0-9.]+)", txt):
            metrics.setdefault(f"Overall Class {m.group(1)} Dice", []).append(float(m.group(2)))

    print(f"Task: {args.task}")
    print(f"Folds aggregated: {len(fold_files)}")
    for f in fold_files:
        print(f"  - {f}")
    print()
    print(f"{'Metric':<26} {'Mean':>8}  {'Std':>10}")
    print("-" * 48)
    for key, vals in metrics.items():
        if not vals:
            continue
        m = statistics.mean(vals)
        s = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        print(f"{key:<26} {m:>8.4f}  (±{s:.4f})")


if __name__ == "__main__":
    main()
