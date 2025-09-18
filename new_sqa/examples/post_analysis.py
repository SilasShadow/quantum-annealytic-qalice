#!/usr/bin/env python3
"""
Post-analysis for execution-optimizer sweeps.

Reads results.csv produced by examples/sweep.py and generates:
- Heatmaps of (AC_cost_SA - AC_cost_SQA) over (lambda_risk, eta) for each (bin_size, pov_cap).
- A summary table of SQA win rate and average improvement.
- A scatter plot of SA vs SQA AC_cost with a 45° line.

Outputs:
  heatmap_bin{bin}_pov{pov}.png
  sa_vs_sqa_scatter.png
  summary.txt (plain-text report)
"""

import csv
import math
from collections import defaultdict, Counter
import numpy as np
import matplotlib.pyplot as plt

def load_results(path="results.csv"):
    rows = []
    with open(path) as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            try:
                rows.append({
                    "bin": int(float(r["bin"])),
                    "pov": float(r["pov"]),
                    "eta": float(r["eta"]),
                    "lam": float(r["lambda"]),
                    "reads": int(float(r["reads"])),
                    "sweeps": int(float(r["sweeps"])),
                    "baseline_ac": float(r["baseline_ac"]),
                    "sa_ac": float(r["sa_ac"]),
                    "sqa_ac": float(r["sqa_ac"]),
                    "sa_energy": float(r["sa_energy"]) if r["sa_energy"] not in ("", "-") else math.nan,
                    "sqa_energy": float(r["sqa_energy"]) if r["sqa_energy"] not in ("", "-") else math.nan,
                    "sa_total": float(r["sa_total"]),
                    "sqa_total": float(r["sqa_total"]),
                    "runtime_s": float(r["runtime_s"]),
                })
            except Exception:
                # Skip malformed rows
                pass
    if not rows:
        raise SystemExit("No rows parsed from results.csv. Did you run examples/sweep.py?")
    return rows

def unique_sorted(vals, tol=1e-12):
    """Stable unique with tolerance; returns sorted list of floats."""
    s = sorted(vals)
    out = []
    for v in s:
        if not out or abs(v - out[-1]) > tol:
            out.append(v)
    return out

def heatmap_group(rows, title_prefix=""):
    """
    Build a heatmap of (sa_ac - sqa_ac) with axes: lambda (y) × eta (x).
    Positive values (red) => SQA better than SA (lower cost).
    """
    etas = unique_sorted([r["eta"] for r in rows])
    lams = unique_sorted([r["lam"] for r in rows])
    ix_eta = {v:i for i,v in enumerate(etas)}
    ix_lam = {v:i for i,v in enumerate(lams)}
    Z = np.full((len(lams), len(etas)), np.nan)

    # If multiple points land in same cell (e.g., repeated reads/sweeps), take best improvement
    agg = defaultdict(list)
    for r in rows:
        key = (r["lam"], r["eta"])
        agg[key].append(r["sa_ac"] - r["sqa_ac"])

    for (lam, eta), diffs in agg.items():
        Z[ix_lam[lam], ix_eta[eta]] = np.nanmean(diffs)

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(Z, origin="lower", aspect="auto", cmap="RdYlGn", vmin=-np.nanmax(np.abs(Z)), vmax=np.nanmax(np.abs(Z)))
    # Axes ticks & labels
    ax.set_xticks(range(len(etas)))
    ax.set_yticks(range(len(lams)))
    ax.set_xticklabels([f"{e:.1e}" for e in etas], rotation=45, ha="right")
    ax.set_yticklabels([f"{l:g}" for l in lams])
    ax.set_xlabel("eta (impact coefficient)")
    ax.set_ylabel("lambda (risk aversion)")
    ax.set_title(f"{title_prefix}  ΔAC = SA − SQA  (red ⇒ SQA better)")
    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label("AC_cost(SA) − AC_cost(SQA)")

    # Optional: annotate cells with small grids
    for i in range(len(lams)):
        for j in range(len(etas)):
            val = Z[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2e}", ha="center", va="center", fontsize=7)

    return fig, ax, Z, etas, lams

def main():
    rows = load_results("results.csv")

    # Basic sanity: SA/SQA totals should match target shares (allow small tolerance)
    bad_totals = [r for r in rows if abs(r["sa_total"] - r["sqa_total"]) > 1e-6]
    if bad_totals:
        print(f"[warn] {len(bad_totals)} rows have mismatched SA vs SQA totals (check inputs).")

    # Global scatter: SA vs SQA AC cost
    sa = np.array([r["sa_ac"] for r in rows], float)
    sqa = np.array([r["sqa_ac"] for r in rows], float)
    fig = plt.figure(figsize=(5.5, 5))
    ax = plt.gca()
    ax.scatter(sa, sqa, s=16, alpha=0.7)
    m = np.nanmax([sa, sqa])
    ax.plot([0, m], [0, m], linestyle="--", linewidth=1)
    ax.set_xlabel("AC_cost (SA)")
    ax.set_ylabel("AC_cost (SQA)")
    ax.set_title("SA vs SQA AC cost (lower is better)")
    plt.tight_layout()
    plt.savefig("sa_vs_sqa_scatter.png", dpi=150)
    plt.close(fig)

    # Group by (bin, pov); build heatmaps over (lam, eta)
    by_group = defaultdict(list)
    for r in rows:
        by_group[(r["bin"], r["pov"])].append(r)

    summary_lines = []
    summary_lines.append("Group summary (by bin_size, pov_cap)")
    summary_lines.append("group, n_points, SQA_wins, SA_wins, ties, win_rate_SQA, mean_delta(SA-SQA)")

    for (bin_sz, pov) in sorted(by_group.keys()):
        grp = by_group[(bin_sz, pov)]
        # Heatmap
        title = f"bin={bin_sz}, pov={pov:.2f}"
        fig, ax, Z, etas, lams = heatmap_group(grp, title_prefix=title)
        out_name = f"heatmap_bin{bin_sz}_pov{str(pov).replace('.','_')}.png"
        plt.tight_layout(); plt.savefig(out_name, dpi=150); plt.close(fig)

        # Wins / losses
        wins = Counter()
        deltas = []
        for r in grp:
            d = r["sa_ac"] - r["sqa_ac"]
            deltas.append(d)
            if abs(d) < 1e-12:
                wins["tie"] += 1
            elif d > 0:
                wins["sqa"] += 1
            else:
                wins["sa"] += 1
        n = len(grp)
        sqa_w = wins["sqa"]
        sa_w = wins["sa"]
        ties = wins["tie"]
        win_rate = sqa_w / n if n else float("nan")
        mean_delta = float(np.nanmean(deltas)) if deltas else float("nan")
        summary_lines.append(f"(bin {bin_sz}, pov {pov:.2f}), {n}, {sqa_w}, {sa_w}, {ties}, {win_rate:.3f}, {mean_delta:.3e}")

    # Global summary as well
    global_delta = float(np.nanmean([r["sa_ac"] - r["sqa_ac"] for r in rows]))
    global_wr = sum(1 for r in rows if r["sa_ac"] > r["sqa_ac"]) / len(rows)
    summary_lines.append("")
    summary_lines.append(f"GLOBAL: SQA wins in {global_wr:.1%} of cases; mean ΔAC (SA−SQA) = {global_delta:.3e}")

    with open("summary.txt", "w") as f:
        f.write("\n".join(summary_lines))
    print("\n".join(summary_lines))
    print("\nWrote: sa_vs_sqa_scatter.png, heatmap_bin*_pov*.png, summary.txt")

if __name__ == "__main__":
    main()
