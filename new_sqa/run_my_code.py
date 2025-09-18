#!/usr/bin/env python3
import json, csv, sys, subprocess
import numpy as np
import dimod
from perf import timed

# ---------------------------------------------------------
# Smoothness Penalty
# ---------------------------------------------------------
def smoothness_penalty(fills, gamma=1e-3):
    """
    Penalize large jumps between consecutive fills.
    Args:
        fills : list or np.array of trade sizes per bin
        gamma : float, weight of smoothness term
    Returns:
        float, smoothness penalty value
    """
    q = np.asarray(fills, float)
    diffs = np.diff(q)               # q[t+1] - q[t]
    return float(gamma * np.sum(diffs**2))


# ---------------------------------------------------------
# AC cost
# ---------------------------------------------------------

def ac_cost(q, V, eta, lam, sigma):
    q, V = np.asarray(q, float), np.asarray(V, float) + 1e-12
    impact = np.sum(eta * (q / V) * q)
    inv = np.cumsum(q[::-1])[::-1]
    if np.isscalar(sigma):
        sigma = np.full_like(q, float(sigma))
    risk = lam * np.sum((np.asarray(sigma, float) ** 2) * inv**2)
    return float(impact + risk)

# ---------------------------------------------------------
# Data
# ---------------------------------------------------------

def load_curve(csv_path):
    V, blackout = [], []
    with open(csv_path) as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            V.append(float(r["exp_vol"]))
            blackout.append(r["blackout"].lower() == "true")
    return np.array(V, float), np.array(blackout, bool)

# ---------------------------------------------------------
# Baseline schedule (PoV)
# ---------------------------------------------------------
def baseline_schedule(X, V, pov_cap, blackout):
    q = np.minimum(pov_cap * V, 1e18)
    q[blackout] = 0.0
    q = q / (q.sum() + 1e-12) * X
    return np.round(q, 6)

# ---------------------------------------------------------
# QUBO builders (same signature)
# ---------------------------------------------------------

@timed("build_qubo")
def build_qubo(V, blackout, X, bin_size, pov_cap, eta, lam_buy=500.0):
    N = len(V)
    B = int(X // bin_size) + 2
    linear, quad = {}, {}

    # linear terms: impact + pov + blackout
    for t in range(N):
        for k in range(B):
            v = f"x_{t}_{k}"
            qty = k * bin_size
            coeff = eta * (qty**2) / (V[t] + 1e-12)
            cap = pov_cap * V[t]
            if qty > cap:
                coeff += 10.0 * (qty - cap)
            if blackout[t]:
                coeff += 1e6
            if coeff:
                linear[v] = linear.get(v, 0.0) + coeff

    # buy-all penalty (Σ qty_i x_i - X)^2
    vars_list, coeffs = [], []
    for t in range(N):
        for k in range(B):
            vars_list.append(f"x_{t}_{k}")
            coeffs.append(k * bin_size)

    for i, v in enumerate(vars_list):
        linear[v] = linear.get(v, 0.0) + lam_buy * (coeffs[i]**2 - 2 * coeffs[i] * X)

    for i in range(len(vars_list)):
        ci, vi = coeffs[i], vars_list[i]
        if ci == 0.0:
            continue
        for j in range(i + 1, len(vars_list)):
            cj = coeffs[j]
            if cj == 0.0:
                continue
            vj = vars_list[j]
            quad[(vi, vj)] = quad.get((vi, vj), 0.0) + 2.0 * lam_buy * ci * cj

    return dimod.BinaryQuadraticModel(linear, quad, 0.0, vartype=dimod.BINARY)

# ---- Numba-accelerated pieces (optional) ----
def _have_numba():
    try:
        import numba as nb  # noqa
        return True
    except Exception:
        return False

if _have_numba():
    import numba as nb


    @nb.njit(cache=True)
    def _linear_and_coeffs(N, B, bin_size, X, eta, pov_cap, V, blackout):
        total = N * B
        linear = np.zeros(total)
        coeffs = np.zeros(total)
        for t in range(N):
            vt = V[t]
            cap = pov_cap * vt
            blk = blackout[t]
            for k in range(B):
                idx = t * B + k
                qty = k * bin_size
                val = eta * (qty * qty) / (vt + 1e-12)
                if qty > cap:
                    val += 10.0 * (qty - cap)
                if blk:
                    val += 1e6
                linear[idx] = val
                coeffs[idx] = qty
        return linear, coeffs

@timed("build_qubo_fast")
def build_qubo_fast(V, blackout, X, bin_size, pov_cap, eta, lam_buy=500.0):
    """
    Fast builder: JIT only the per-variable linear terms + coeffs,
    then stream the rank-1 quadratic penalty without dense matrices.
    """
    if not _have_numba():
        # Fallback cleanly if numba isn't installed
        return build_qubo(V, blackout, X, bin_size, pov_cap, eta, lam_buy)

    N = len(V)
    B = int(X // bin_size) + 2
    V_arr = np.asarray(V, np.float64)
    blk = np.asarray(blackout, np.bool_)
    lin_arr, coeffs = _linear_and_coeffs(N, B, bin_size, X, eta, pov_cap, V_arr, blk)

    linear = {}
    for idx, base in enumerate(lin_arr):
        # add buy-all linear piece
        c = coeffs[idx]
        val = base + lam_buy * (c * c - 2.0 * c * X)
        if val != 0.0:
            t, k = divmod(idx, B)
            linear[f"x_{t}_{k}"] = float(val)

    quad = {}
    # stream the upper triangle of the rank-1 term
    for i in range(N * B):
        ci = coeffs[i]
        if ci == 0.0:
            continue
        ti, ki = divmod(i, B)
        vi = f"x_{ti}_{ki}"
        for j in range(i + 1, N * B):
            cj = coeffs[j]
            if cj == 0.0:
                continue
            tj, kj = divmod(j, B)
            vj = f"x_{tj}_{kj}"
            quad[(vi, vj)] = quad.get((vi, vj), 0.0) + 2.0 * lam_buy * ci * cj

    return dimod.BinaryQuadraticModel(linear, quad, 0.0, vartype=dimod.BINARY)

# ---------------------------------------------------------
# Decode
# ---------------------------------------------------------
@timed("decode_best")
def decode_best(sample, N, X, bin_size):
    B = int(X // bin_size) + 2
    fills = []
    for t in range(N):
        q = 0
        for k in range(B):
            if sample.get(f"x_{t}_{k}", 0) == 1:
                q += k * bin_size
        fills.append(q)
    return fills

# ---------------------------------------------------------
# OpenJij runner (SA or SQA)
# ---------------------------------------------------------

@timed("run_openjij")
def run_openjij(bqm, sampler_kind="sa", num_reads=1000, sweeps=10000):
    import openjij as oj
    if sampler_kind == "sqa":
        return oj.SQASampler().sample(bqm, num_reads=num_reads, num_sweeps=sweeps)
    else:
        return oj.SASampler().sample(bqm, num_reads=num_reads, num_sweeps=sweeps)

# ---------------------------------------------------------
# Case Runners (Almgren-Chriss, SA or SQA)
# ---------------------------------------------------------

def ac_baseline(X, V, pov_cap, blackout, eta, lambda_risk, sigma, results):
    q_base = baseline_schedule(X, V, pov_cap, blackout)
    cost_base = ac_cost(q_base, V, eta, lambda_risk, sigma) + smoothness_penalty(fills, gamma=1e-3)

    res_base = {"method": "baseline", "fills": q_base.tolist(), "ac_cost": cost_base}
    results.append(res_base)
    with open("baseline_result.json", "w") as f:
        json.dump(res_base, f)
    return results

def qubo_sa(N, X, V, pov_cap, blackout, bin_size, fast, eta, lambda_risk, sigma, num_reads, sweeps, results):
    builder = build_qubo_fast if fast else build_qubo
    bqm = builder(V, blackout, X, bin_size, pov_cap, eta)
    sa_samples = run_openjij(bqm, "sa", num_reads, sweeps)
    best_sa = sa_samples.first
    q_sa = decode_best(best_sa.sample, N, X, bin_size)
    cost_sa = ac_cost(q_sa, V, eta,lambda_risk, sigma)  + smoothness_penalty(fills, gamma=1e-3)
    res_sa = {"method": "qubo_sa", "fills": q_sa, "energy": float(best_sa.energy), "ac_cost": cost_sa}
    results.append(res_sa)
    with open("qubo_sa_result.json", "w") as f:
        json.dump(res_sa, f)
    return bqm, results

def qubo_sqa(N, X, V, bqm,num_reads, sweeps, bin_size, eta, lambda_risk, sigma, results):
    sqa_samples = run_openjij(bqm, "sqa", num_reads, sweeps)
    best_sqa = sqa_samples.first
    q_sqa = decode_best(best_sqa.sample, N, X, bin_size)
    cost_sqa = ac_cost(q_sqa, V, eta, lambda_risk, sigma)  + smoothness_penalty(fills, gamma=1e-3)
    res_sqa = {"method": "qubo_sqa", "fills": q_sqa, "energy": float(best_sqa.energy), "ac_cost": cost_sqa}
    results.append(res_sqa)
    with open("qubo_qa_result.json", "w") as f:  # filename aligned to compare script
        json.dump(res_sqa, f)
    return results
# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    import argparse, pathlib
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV file with t,exp_vol,blackout")
    ap.add_argument("--target-shares", type=int, default=600)
    ap.add_argument("--bin-size", type=int, default=100)
    ap.add_argument("--pov-cap", type=float, default=0.2)
    ap.add_argument("--eta", type=float, default=1e-7)
    ap.add_argument("--lambda-risk", type=float, default=0.05)
    ap.add_argument("--sigma", type=float, default=1e-3)
    ap.add_argument("--num-reads", type=int, default=500)
    ap.add_argument("--sweeps", type=int, default=5000)
    ap.add_argument("--fast", action="store_true", help="Use Numba-accelerated QUBO builder if available")
    ap.add_argument("--auto-plot", action="store_true", help="Run compare_exec_methods_2.py afterwards")
    ap.add_argument("--compare-script", default="examples/compare_exec_methods_2.py")
    args = ap.parse_args()

    V, blackout = load_curve(args.csv)
    X, N = args.target_shares, len(V)

    results = []

    # 1) Baseline
    results = ac_baseline(X, V, args.pov_cap, blackout, args.eta, args.lambda_risk, args.sigma, results)

    # 2) QUBO + SA (OpenJij)
    bqm, results = qubo_sa(N, X, V, args.pov_cap, blackout, args.bin_size, args.fast, args.eta, args.lambda_risk, args.sigma, args.num_reads, args.sweeps, results)

    # 3) QUBO + SQA (OpenJij)
    results = qubo_sqa(N, X, V, bqm, args.num_reads, args.sweeps, args.bin_size, args.eta, args.lambda_risk, args.sigma, results)

    # Summary
    print("Method,AC_cost,Energy,Total_shares")
    for r in results:
        print(f"{r['method']},{r['ac_cost']:.6g},{r.get('energy','-')},{sum(r['fills'])}")

    # Auto-plot
    if args.auto_plot:
        compare_path = pathlib.Path(args.compare_script)
        if not compare_path.exists() and pathlib.Path("compare_exec_methods_2.py").exists():
            compare_path = pathlib.Path("compare_exec_methods_2.py")
        if compare_path.exists():
            try:
                out = subprocess.check_output([sys.executable, str(compare_path), "--csv", args.csv], text=True)
                print(out)
                print("[info] Plots saved by compare script (e.g., comparison.png, schedules.png).")
            except subprocess.CalledProcessError as e:
                print("[warn] compare script failed:", e.output)
        else:
            print(f"[note] Compare script not found at {args.compare_script}. Skipping plots.")


if __name__ == "__main__":
    main()
