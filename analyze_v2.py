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

    # 三种视角(v3 后 result 里可能有 per_rank_breakdown / wall_clock_breakdown_max):
    #   target : wall_clock_breakdown         (target_rank 实测,含真实 ckpt_load)
    #   max    : wall_clock_breakdown_max      (跨 rank 每项取 max,端到端 wall-clock 上界)
    #   (per-rank 明细单独输出到 per_rank CSV)
    def _pick(t, view):
        if view == "target":
            return t.get("wall_clock_breakdown", {})
        if view == "max":
            return t.get("wall_clock_breakdown_max",
                          t.get("wall_clock_breakdown", {}))
        raise ValueError(view)

    def _print_by_k(view_name: str):
        print(f"\n=== By k — timing (mean ± std)  [view={view_name}] ===")
        header = ["k", "n", "graph_used", "async_used"] + timing_keys \
                 + ["gpu_peak_mib_max", "pre_loss", "loss_gap"]
        print(",".join(header))

        for k in sorted(by_k):
            trials = by_k[k]
            row = [
                str(k), str(len(trials)),
                f"{sum(1 for t in trials if t.get('used_retained_graph')) / len(trials):.2f}",
                f"{sum(1 for t in trials if t.get('used_async_pipeline')) / len(trials):.2f}",
            ]
            for tk in timing_keys:
                vals = [_pick(t, view_name).get(tk, float("nan"))
                        for t in trials
                        if _pick(t, view_name).get(tk) is not None]
                if vals:
                    mean = sum(vals) / len(vals)
                    std  = (sum((v - mean)**2 for v in vals) / len(vals)) ** 0.5
                    row.append(f"{mean:.4f}±{std:.4f}")
                else:
                    row.append("n/a")
            # GPU peak: the highest-MiB rank per trial, meaned across trials.
            # Older runs without gpu_memory key skip cleanly (n/a).
            peaks = [t.get("gpu_memory", {}).get("peak_max_mib")
                     for t in trials if t.get("gpu_memory") is not None]
            peaks = [p for p in peaks if p is not None]
            if peaks:
                m = sum(peaks) / len(peaks)
                s = (sum((v - m)**2 for v in peaks) / len(peaks)) ** 0.5
                row.append(f"{m:.1f}±{s:.1f}")
            else:
                row.append("n/a")
            pre_losses = [t["convergence"]["pre_preemption_loss"] for t in trials]
            loss_gaps  = [t["convergence"]["loss_gap"] for t in trials
                          if t["convergence"]["loss_gap"] is not None]
            row.append(f"{sum(pre_losses)/len(pre_losses):.4f}")
            row.append(f"{sum(loss_gaps)/len(loss_gaps):.4f}" if loss_gaps else "n/a")
            print(",".join(row))

    _print_by_k("target")   # target_rank 视角(默认;ckpt_load 真实)
    _print_by_k("max")      # max-over-ranks 视角(端到端 wall-clock)

    # ── Phase-level 端到端时序 ───────────────────────────────────────────────
    # 一次 trial 从头到尾:Phase A warmup → A.6 inject → A.7 save ckpt →
    #                     B relaunch → C recovery → D catchup
    # 这里报告 max-over-ranks(该 phase 结束的 wall-clock)。
    has_phase = any(r.get("phase_timings") for r in results)
    if has_phase:
        print("\n=== Phase-level timings (max over ranks, seconds) ===")
        hdr_pt = ["k", "trial",
                  "A_warmup", "A6_inject", "A7_ckpt_save",
                  "B_relaunch", "C_recovery", "D_catchup",
                  "trial_end_to_end"]
        print(",".join(hdr_pt))
        for r in sorted(results, key=lambda x: (x["k"], x["trial"])):
            pt = r.get("phase_timings") or {}
            print(",".join([
                str(r["k"]), str(r["trial"]),
                f"{pt.get('phase_a_warmup_total_sec', 0):.3f}",
                f"{pt.get('phase_a6_inject_step_sec', 0):.3f}",
                f"{pt.get('phase_a7_ckpt_save_sec_max', 0):.3f}",
                f"{pt.get('phase_b_relaunch_sec_max', 0):.3f}",
                f"{pt.get('phase_c_recovery_sec_max', 0):.3f}",
                f"{pt.get('phase_d_catchup_total_sec', 0):.3f}",
                f"{pt.get('trial_end_to_end_sec', 0):.3f}",
            ]))

    # ── Per-step loss & wall-clock 时序(warmup + catchup)────────────────────
    # 每 trial 一段完整的曲线:warmup 20 步 + catchup 5 步。
    # 输出示例给控制台;完整数据在 summary_v2_steps.csv 里。
    has_steps = any(r.get("per_step_timings") for r in results)
    if has_steps:
        print("\n=== Per-step timings (first trial per k, warmup + catchup) ===")
        seen_k = set()
        for r in sorted(results, key=lambda x: (x["k"], x["trial"])):
            if r["k"] in seen_k:
                continue
            seen_k.add(r["k"])
            pst = r.get("per_step_timings") or {}
            print(f"--- k={r['k']} trial={r['trial']} target_rank={r.get('target_rank')} ---")
            print(f"  warmup ({len(pst.get('warmup', []))} steps):")
            for e in pst.get("warmup", []):
                loss = f"{e['loss']:.4f}" if e.get("loss") is not None else "n/a"
                print(f"    step {e['step']:3d}  {e['sec']*1000:7.1f} ms  loss={loss}")
            print(f"  catchup ({len(pst.get('catchup', []))} steps):")
            for e in pst.get("catchup", []):
                loss = f"{e['loss']:.4f}" if e.get("loss") is not None else "n/a"
                print(f"    step {e['step']:3d}  {e['sec']*1000:7.1f} ms  loss={loss}")

    # ── Per-rank 明细 ────────────────────────────────────────────────────────
    # 每 rank 各写一行:能直接看到 target_rank 上 ckpt_load ≠ 0,
    # 其他 rank 上 ckpt_load == 0(它们是 helper/victim,权重没丢)。
    has_per_rank = any(r.get("per_rank_breakdown") for r in results)
    if has_per_rank:
        print("\n=== Per-rank breakdown (每 rank 独立时序) ===")
        hdr_pr = ["k", "trial", "rank", "role",
                  "detect", "destroy", "ckpt", "reinit",
                  "resend", "compute", "overlap"]
        print(",".join(hdr_pr))
        for r in sorted(results, key=lambda x: (x["k"], x["trial"])):
            prb = r.get("per_rank_breakdown") or {}
            for rank_id in sorted(prb, key=lambda x: int(x)):
                pr = prb[rank_id]
                print(",".join([
                    str(r["k"]), str(r["trial"]), str(rank_id),
                    pr.get("role", "?"),
                    f"{pr.get('detection_sec', 0):.3f}",
                    f"{pr.get('pg_destroy_sec', 0):.3f}",
                    f"{pr.get('ckpt_load_sec', 0):.3f}",
                    f"{pr.get('comm_rebuild_sec', 0):.3f}",
                    f"{pr.get('activation_resend_sec', 0):.3f}",
                    f"{pr.get('recovery_compute_sec', 0):.3f}",
                    f"{pr.get('overlap_sec', 0):.3f}",
                ]))

    print("\n=== Per-trial ===")
    hdr2 = ["k","trial","graph","async","detect","destroy","ckpt","reinit",
            "relaunch","resend","compute","overlap","total",
            "gpu_peak_mib","gpu_peak_rank",
            "pre_loss","rec_loss","gap"]
    print(",".join(hdr2))
    for r in sorted(results, key=lambda x: (x["k"], x["trial"])):
        wb = r["wall_clock_breakdown"]
        cv = r["convergence"]
        gm = r.get("gpu_memory") or {}
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
            f"{gm.get('peak_max_mib', 0):.1f}",
            str(gm.get('peak_max_rank', '')),
            f"{cv.get('pre_preemption_loss',0):.4f}",
            f"{cv.get('recovery_loss') or 0:.4f}",
            f"{cv.get('loss_gap') or 0:.4f}",
        ]
        print(",".join(row))

    # 尝试保存 CSV
    try:
        import pandas as pd
        rows = []
        rows_max = []
        rows_per_rank = []
        for r in results:
            wb = r.get("wall_clock_breakdown") or {}
            wb_max = r.get("wall_clock_breakdown_max") or wb
            cv = r["convergence"]
            gm = r.get("gpu_memory") or {}
            common = {
                "k": r["k"], "trial": r["trial"],
                "target_rank": r.get("target_rank"),
                "graph_used": r.get("used_retained_graph"),
                "async_used": r.get("used_async_pipeline"),
                "gpu_peak_mib_max":  gm.get("peak_max_mib"),
                "gpu_peak_rank_max": gm.get("peak_max_rank"),
                "pre_preemption_loss": cv.get("pre_preemption_loss"),
                "recovery_loss": cv.get("recovery_loss"),
                "loss_gap": cv.get("loss_gap"),
                "converge_step": cv.get("converge_step"),
            }
            rows.append({**common, **{k: wb.get(k) for k in timing_keys}})
            rows_max.append({**common,
                             **{k: wb_max.get(k) for k in timing_keys}})

            prb = r.get("per_rank_breakdown") or {}
            for rank_id in sorted(prb, key=lambda x: int(x)):
                pr = prb[rank_id]
                rows_per_rank.append({
                    "k": r["k"], "trial": r["trial"],
                    "target_rank": r.get("target_rank"),
                    "rank": int(rank_id), "role": pr.get("role"),
                    "detection_sec":         pr.get("detection_sec"),
                    "pg_destroy_sec":        pr.get("pg_destroy_sec"),
                    "ckpt_load_sec":         pr.get("ckpt_load_sec"),
                    "comm_rebuild_sec":      pr.get("comm_rebuild_sec"),
                    "relaunch_sec":          pr.get("relaunch_sec"),
                    "activation_resend_sec": pr.get("activation_resend_sec"),
                    "recovery_compute_sec":  pr.get("recovery_compute_sec"),
                    "overlap_sec":           pr.get("overlap_sec"),
                })

        out_dir = Path(cfg_path(cfg["output"]["results_dir"]))
        out_dir.mkdir(parents=True, exist_ok=True)

        out = out_dir / "summary_v2.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"\n[saved] {out}  (view=target: target_rank 实测,含真实 ckpt_load)")

        out2 = out_dir / "summary_v2_max.csv"
        pd.DataFrame(rows_max).to_csv(out2, index=False)
        print(f"[saved] {out2}  (view=max: 每项跨 rank 取 max)")

        if rows_per_rank:
            out3 = out_dir / "summary_v2_per_rank.csv"
            pd.DataFrame(rows_per_rank).to_csv(out3, index=False)
            print(f"[saved] {out3}  (每 rank 一行,含 role 标签)")

        # ── Phase-level CSV(一行一 trial,含端到端 wall-clock)────────────
        rows_phase = []
        for r in results:
            pt = r.get("phase_timings") or {}
            if not pt:
                continue
            rows_phase.append({
                "k": r["k"], "trial": r["trial"],
                "target_rank": r.get("target_rank"),
                **pt,
            })
        if rows_phase:
            out4 = out_dir / "summary_v2_phases.csv"
            pd.DataFrame(rows_phase).to_csv(out4, index=False)
            print(f"[saved] {out4}  (Phase A/A.6/A.7/B/C/D 完整时序 + trial 端到端)")

        # ── Per-step CSV(warmup + catchup 每步一行,画 loss 曲线用)──────
        rows_step = []
        for r in results:
            pst = r.get("per_step_timings") or {}
            for phase, entries in (("warmup", pst.get("warmup", [])),
                                    ("catchup", pst.get("catchup", []))):
                for e in entries:
                    rows_step.append({
                        "k": r["k"], "trial": r["trial"],
                        "target_rank": r.get("target_rank"),
                        "phase": phase,
                        "step":  e.get("step"),
                        "sec":   e.get("sec"),
                        "loss":  e.get("loss"),
                    })
        if rows_step:
            out5 = out_dir / "summary_v2_steps.csv"
            pd.DataFrame(rows_step).to_csv(out5, index=False)
            print(f"[saved] {out5}  (每步 wall-clock + loss,画曲线用)")
    except ImportError:
        print("\n[INFO] pandas not available; skipping CSV export")


if __name__ == "__main__":
    main()
