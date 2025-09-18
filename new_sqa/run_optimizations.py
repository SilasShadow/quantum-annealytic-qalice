#!/usr/bin/env python3
import json, csv, sys, os
import numpy as np
path = os.getcwd()
# -----------------------------
# Shared AC cost function
# -----------------------------
def ac_cost(q, V, eta, lam, sigma):
    q, V = np.asarray(q, float), np.asarray(V, float) + 1e-12
    impact = np.sum(eta * (q / V) * q)
    inv = np.cumsum(q[::-1])[::-1]
    if np.isscalar(sigma):
        sigma = np.full_like(q, float(sigma))
    risk = lam * np.sum((np.asarray(sigma, float) ** 2) * inv**2)
    return float(impact + risk)

# -----------------------------
# Load volume curve CSV
# -----------------------------
def load_curve(csv_path):
    V, blackout = [], []
    with open(path+"/"+csv_path) as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            V.append(float(r["exp_vol"]))
            blackout.append(r["blackout"].lower() == "true")
    return np.array(V, float), np.array(blackout, bool)

# -----------------------------
# Baseline schedule (simple proportional-to-volume)
# -----------------------------
def baseline_schedule(X, V, pov_cap, blackout):
    q = np.minimum(pov_cap * V, 1e18)
    q[blackout] = 0.0
    q = q / q.sum() * X
    return np.round(q, 6)

# -----------------------------
# QUBO builder
# -----------------------------
import dimod
def build_qubo(V, blackout, X, bin_size, pov_cap, eta, lam_buy=500.0):
    N = len(V)
    B = int(X // bin_size) + 2
    linear, quad = {}, {}
    for t in range(N):
        for k in range(B):
            v = f"x_{t}_{k}"
            qty = k * bin_size
            linear[v] = linear.get(v, 0.0) + eta * (qty**2) / (V[t] + 1e-12)
            cap = pov_cap * V[t]
            if qty > cap: linear[v] += 10.0 * (qty - cap)
            if blackout[t]: linear[v] += 1e6
    vars_list, coeffs = [], []
    for t in range(N):
        for k in range(B):
            vars_list.append(f"x_{t}_{k}")
            coeffs.append(k * bin_size)
    for i, v in enumerate(vars_list):
        linear[v] = linear.get(v, 0.0) + lam_buy * (coeffs[i]**2 - 2*coeffs[i]*X)
    for i in range(len(vars_list)):
        ci, vi = coeffs[i], vars_list[i]
        for j in range(i+1, len(vars_list)):
            quad[(vi, vars_list[j])] = quad.get((vi, vars_list[j]), 0.0) + 2 * lam_buy * ci * coeffs[j]
    return dimod.BinaryQuadraticModel(linear, quad, 0.0, vartype=dimod.BINARY)

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

# -----------------------------
# OpenJij runner
# -----------------------------
def run_openjij(bqm, sampler_kind="sa", num_reads=1000, sweeps=10000):
    import openjij as oj
    if sampler_kind == "sqa":
        sampler = oj.SQASampler()
        return sampler.sample_bqm(bqm, num_reads=num_reads, num_sweeps=sweeps)
    else:
        sampler = oj.SASampler()
        return sampler.sample_bqm(bqm, num_reads=num_reads, num_sweeps=sweeps)

# -----------------------------
# Main driver
# -----------------------------
def main():
    import argparse
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
    args = ap.parse_args()

    V, blackout = load_curve(args.csv)
    X, N = args.target_shares, len(V)

    results = []

    # 1. Baseline
    q_base = baseline_schedule(X, V, args.pov_cap, blackout)
    cost_base = ac_cost(q_base, V, args.eta, args.lambda_risk, args.sigma)
    res_base = {"method":"baseline","fills":q_base.tolist(),"ac_cost":cost_base}
    results.append(res_base)
    json.dump(res_base, open("baseline_result.json","w"))

    # 2. QUBO + OpenJij SA
    bqm = build_qubo(V, blackout, X, args.bin_size, args.pov_cap, args.eta)
    sa_samples = run_openjij(bqm, "sa", args.num_reads, args.sweeps)
    best_sa = sa_samples.first
    q_sa = decode_best(best_sa.sample, N, X, args.bin_size)
    cost_sa = ac_cost(q_sa, V, args.eta, args.lambda_risk, args.sigma)
    res_sa = {"method":"qubo_sa","fills":q_sa,"energy":float(best_sa.energy),"ac_cost":cost_sa}
    results.append(res_sa)
    json.dump(res_sa, open("qubo_sa_result.json","w"))

    # 3. QUBO + OpenJij SQA
    sqa_samples = run_openjij(bqm, "sqa", args.num_reads, args.sweeps)
    best_sqa = sqa_samples.first
    q_sqa = decode_best(best_sqa.sample, N, X, args.bin_size)
    cost_sqa = ac_cost(q_sqa, V, args.eta, args.lambda_risk, args.sigma)
    res_sqa = {"method":"qubo_sqa","fills":q_sqa,"energy":float(best_sqa.energy),"ac_cost":cost_sqa}
    results.append(res_sqa)
    json.dump(res_sqa, open("qubo_sqa_result.json","w"))

    # Print summary table
    print("Method,AC_cost,Energy,Total_shares")
    for r in results:
        print(f"{r['method']},{r['ac_cost']:.6g},{r.get('energy','-')},{sum(r['fills'])}")

if __name__ == "__main__":
    main()
