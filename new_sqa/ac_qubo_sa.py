#!/usr/bin/env python3
import argparse, json
import numpy as np, dimod

# Prefer modern sampler; else use dimod reference SA as fallback
# sampler selection (OpenJij first, then fallbacks)
def pick_sampler(kind="sa", sweeps=10_000):
    """
    kind: 'sa' (Simulated Annealing), 'sqa' (Simulated Quantum Annealing), or 'ref' (dimod reference SA)
    """
    try:
        import openjij as oj
        if kind == "sqa":
            return lambda bqm, num_reads: oj.SQASampler(num_reads=num_reads, sweeps=sweeps).sample_bqm(bqm)
        else:
            return lambda bqm, num_reads: oj.SASampler(num_reads=num_reads, sweeps=sweeps).sample_bqm(bqm)
    except Exception:
        # fall back to dwave-samplers SA if present, else dimod reference
        try:
            from dwave.samplers import SimulatedAnnealingSampler as SA
            return lambda bqm, num_reads: SA().sample(bqm, num_reads=num_reads)
        except Exception:
            from dimod.reference.samplers import SimulatedAnnealingSampler as RefSA
            return lambda bqm, num_reads: RefSA().sample(bqm, num_reads=num_reads)

def ac_cost(q, V, eta, lam, sigma):
    q, V = np.asarray(q, float), np.asarray(V, float) + 1e-12
    impact = np.sum(eta * (q / V) * q)
    inv = np.cumsum(q[::-1])[::-1]
    if np.isscalar(sigma): sigma = np.full_like(q, float(sigma))
    risk = lam * np.sum((np.asarray(sigma, float) ** 2) * inv**2)
    return float(impact + risk)

def load_curve(csv_path):
    import csv
    V, blackout = [], []
    with open(csv_path) as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            V.append(float(r["exp_vol"]))
            blackout.append(r["blackout"].lower() == "true")
    return np.array(V, float), np.array(blackout, bool)

def build_qubo(V, blackout, X, bin_size, pov_cap, eta, lam_buy=500.0):
    """
    Constraint→penalty QUBO: (Σ b_k x_k − X)^2 + linear impact + POV/blackout penalties.
    Matches the standard transformation from constrained quadratic to QUBO used in the papers.
    """
    N = len(V)
    B = int(X // bin_size) + 2
    linear, quad = {}, {}

    # linear terms: impact proxy (eta * (q^2 / V)), POV soft penalty, blackout hard penalty
    for t in range(N):
        for k in range(B):
            var = f"x_{t}_{k}"
            qty = k * bin_size
            # impact ~ eta * (qty^2 / V_t) → linear in x_{t,k} with coefficient proportional to qty^2
            linear[var] = linear.get(var, 0.0) + eta * (qty**2) / (V[t] + 1e-12)
            # pov soft penalty
            cap = pov_cap * V[t]
            if qty > cap:
                linear[var] += 10.0 * (qty - cap)
            # blackout hard penalty
            if blackout[t]:
                linear[var] += 1e6

    # buy-all: (Σ qty_i x_i − X)^2
    # expand: adds linear & quadratic couplings
    vars_list, coeffs = [], []
    for t in range(N):
        for k in range(B):
            vars_list.append(f"x_{t}_{k}")
            coeffs.append(k * bin_size)

    for i, v in enumerate(vars_list):
        linear[v] = linear.get(v, 0.0) + lam_buy * (coeffs[i]**2 - 2*coeffs[i]*X)

    for i in range(len(vars_list)):
        ci = coeffs[i]
        vi = vars_list[i]
        for j in range(i+1, len(vars_list)):
            vj = vars_list[j]
            quad[(vi, vj)] = quad.get((vi, vj), 0.0) + 2 * lam_buy * ci * coeffs[j]

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--target-shares", type=int, default=10000)
    ap.add_argument("--bin-size", type=int, default=100)
    ap.add_argument("--pov-cap", type=float, default=0.2)
    ap.add_argument("--eta", type=float, default=1e-7)
    ap.add_argument("--lambda-risk", type=float, default=0.05)  # not in QUBO v1, kept for AC eval
    ap.add_argument("--sigma", type=float, default=1e-3)
    ap.add_argument("--num-reads", type=int, default=2000)
    args = ap.parse_args()

    V, blackout = load_curve(args.csv)
    N, X = len(V), args.target_shares
    bqm = build_qubo(V, blackout, X, args.bin_size, args.pov_cap, args.eta)
    best = SA().sample(bqm, num_reads=args.num_reads).first
    fills = decode_best(best.sample, N, X, args.bin_size)
    result = {
        "fills": fills,
        "energy": float(best.energy),
        "ac_cost": ac_cost(fills, V, args.eta, args.lambda_risk, args.sigma)
    }
    print(json.dumps(result))
    with open("qubo_sa_result.json", "w") as f:
        json.dump(result, f)

if __name__ == "__main__":
    main()
