#!/usr/bin/env python3
import csv, itertools, subprocess, sys, time, pathlib

CSV = "examples/big_volume_curve.csv"  # or bigger_curve.csv
OUT = "results.csv"

# grids
LAMBDAS = [0.05, 0.1, 0.2]
ETAS    = [1e-5, 1e-4, 3e-3]
POVS    = [0.1, 0.2, 0.3]
BINS    = [500, 250]
HZS     = [24]  # or [24, 48]
READS   = [100, 500]
SWEEPS  = [1000, 2000]

TARGET  = 20000

def run_one(curve, target, bin_size, pov, eta, lam, reads, sweeps):
    cmd = [sys.executable, "run_my_code.py",
        "--csv", curve, "--target-shares", str(target),
        "--bin-size", str(bin_size), "--pov-cap", str(pov),
        "--eta", str(eta), "--lambda-risk", str(lam), "--sigma", "1e-3",
        "--num-reads", str(reads), "--sweeps", str(sweeps), "--fast"
    ]
    t0 = time.time()
    out = subprocess.check_output(cmd, text=True)
    dt = time.time() - t0
    # parse the summary table from stdout (last block)
    lines = [l for l in out.strip().splitlines() if l and not l.startswith("[")]
    # last 3 lines correspond to baseline, SA, SQA rows
    rows = lines[-3:]
    parsed = [r.split(",") for r in rows[1:]]  # skip header
    # (method, AC_cost, energy, total)
    return dt, parsed

def main():
    path = pathlib.Path(OUT)
    new = not path.exists()
    with open(OUT, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["bin","pov","eta","lambda","reads","sweeps",
                        "baseline_ac","sa_ac","sqa_ac","sa_energy","sqa_energy",
                        "sa_total","sqa_total","runtime_s"])
        for (bin_size,pov,eta,lam,reads,sweeps) in itertools.product(BINS, POVS, ETAS, LAMBDAS, READS, SWEEPS):
            try:
                rt, rows = run_one(CSV, TARGET, bin_size, pov, eta, lam, reads, sweeps)
                # rows[0]=baseline, rows[1]=SA, rows[2]=SQA
                _, base_ac, _, _ = rows[0]
                _, sa_ac, sa_en, sa_tot = rows[1]
                _, sqa_ac, sqa_en, sqa_tot = rows[2]
                w.writerow([bin_size,pov,eta,lam,reads,sweeps,
                            base_ac, sa_ac, sqa_ac, sa_en, sqa_en, sa_tot, sqa_tot, f"{rt:.3f}"])
                f.flush()
                print("OK", bin_size,pov,eta,lam,reads,sweeps)
            except subprocess.CalledProcessError as e:
                print("FAIL", e)
                continue

if __name__ == "__main__":
    main()
