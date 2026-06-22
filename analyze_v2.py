"""
v2 实验结果分析脚本。

用法：
  python analyze_v2.py --config configs/v2.yaml
"""
import argparse, json, sys
from pathlib import Path
from collections import defaultdict
import yaml


def cfg_path(v):
    import os
    return os.path.expanduser(os.path.expandvars(v))


def load_results(d):
    data = []
    for f in sorted(Path(d).glob("injection_k*_trial*.json")):
        try:
            data.append(json.loads(f.read_text()))
        except Exception as e:
            print(f"[WARN] skip {f}: {e}", file=sys.stderr)
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    results = load_results(cfg_path(cfg["output"]["results_dir"]))
    print(f"Loaded {len(results)} result files")
    if not results:
        return

    by_k = defaultdict(list)
    for r in results:
        by_k[r["k"]].append(r)

    timing_keys = [
        "detection_sec", "pg_destroy_sec", "ckpt_load_sec",
        "comm_rebuild_sec", "relaunch_sec",
        "activation_resend_sec", "recovery_compute_sec",
        "overlap_sec", "total_recovery_sec",
    ]

    print("\n=== By k — timing (mean ± std) ===")
    header = ["k", "n", "graph_used", "async_used"] + timing_keys + ["pre_loss", "loss_gap"]
    print(",".join(header))

    for k in sorted(by_k):
        trials = by_k[k]
        row = [
            str(k), str(len(trials)),
            f"{sum(1 for t in trials if t.get('used_retained_graph')) / len(trials):.2f}",
            f"{sum(1 for t in trials if t.get('used_async_pipeline')) / len(trials):.2f}",
        ]
        for tk in timing_keys:
            vals = [t["wall_clock_breakdown"].get(tk, float("nan"))
                    for t in trials if t["wall_clock_breakdown"].get(tk) is not None]
            if vals:
                mean = sum(vals) / len(vals)
                std  = (sum((v - mean)**2 for v in vals) / len(vals)) ** 0.5
                row.append(f"{mean:.4f}±{std:.4f}")
            else:
                row.append("n/a")
        pre_losses = [t["convergence"]["pre_preemption_loss"] for t in trials]
        loss_gaps  = [t["convergence"]["loss_gap"] for t in trials
                      if t["convergence"]["loss_gap"] is not None]
        row.append(f"{sum(pre_losses)/len(pre_losses):.4f}")
        row.append(f"{sum(loss_gaps)/len(loss_gaps):.4f}" if loss_gaps else "n/a")
        print(",".join(row))

    print("\n=== Per-trial ===")
    hdr2 = ["k","trial","graph","async","detect","destroy","ckpt","reinit",
            "relaunch","resend","compute","overlap","total","pre_loss","rec_loss","gap"]
    print(",".join(hdr2))
    for r in sorted(results, key=lambda x: (x["k"], x["trial"])):
        wb = r["wall_clock_breakdown"]
        cv = r["convergence"]
        row = [
            str(r["k"]), str(r["trial"]),
            str(r.get("used_retained_graph")),
            str(r.get("used_async_pipeline")),
            f"{wb.get('detection_sec',0):.3f}",
            f"{wb.get('pg_destroy_sec',0):.3f}",
            f"{wb.get('ckpt_load_sec',0):.3f}",
            f"{wb.get('comm_rebuild_sec',0):.3f}",
            f"{wb.get('relaunch_sec',0):.3f}",
            f"{wb.get('activation_resend_sec',0):.3f}",
            f"{wb.get('recovery_compute_sec',0):.3f}",
            f"{wb.get('overlap_sec',0):.3f}",
            f"{wb.get('total_recovery_sec',0):.3f}",
            f"{cv.get('pre_preemption_loss',0):.4f}",
            f"{cv.get('recovery_loss') or 0:.4f}",
            f"{cv.get('loss_gap') or 0:.4f}",
        ]
        print(",".join(row))

    # 尝试保存 CSV
    try:
        import pandas as pd
        rows = []
        for r in results:
            wb = r["wall_clock_breakdown"]
            cv = r["convergence"]
            rows.append({"k": r["k"], "trial": r["trial"],
                         "graph_used": r.get("used_retained_graph"),
                         "async_used": r.get("used_async_pipeline"),
                         **{k: wb.get(k) for k in timing_keys},
                         "pre_preemption_loss": cv.get("pre_preemption_loss"),
                         "recovery_loss": cv.get("recovery_loss"),
                         "loss_gap": cv.get("loss_gap"),
                         "converge_step": cv.get("converge_step")})
        out = Path(cfg_path(cfg["output"]["results_dir"])) / "summary_v2.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"\n[saved] {out}")
    except ImportError:
        print("\n[INFO] pandas not available; skipping CSV export")


if __name__ == "__main__":
    main()
