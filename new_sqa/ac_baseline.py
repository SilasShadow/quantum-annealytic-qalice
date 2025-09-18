#!/usr/bin/env python3
import argparse, json, math
import numpy as np

def ac_cost(q, V, eta, lam, sigma):
    q, V = np.asarray(q, float), np.asarray(V, float) + 1e-12
    impact = np.sum(eta * (q / V) * q)
    inv = np.cumsum(q[::-1])[::-1]
    if np.isscalar(sigma): sigma = np.full_like(q, float(sigma))
    risk = lam * np.sum((np.asarray(sigma, float) ** 2) * inv**2)
    return float(impact + risk)

def pov_schedule(X, V, pov_cap, blackout):
    q = np.minimum(pov_cap * V, 1e18)
    q[blackout] = 0.0
    if q.sum() <= 0: raise ValueError("All slices blacked out or pov 0.")
    q = q / q.sum() * X
    return np.round(q, 6)

def uniform_schedule(X, V, pov_cap, blackout):
    m = (~blackout).sum()
    if m == 0: raise ValueError("All slices blacked out.")
    q = np.zeros_like(V, float)
    q[~blackout] = X / m
    # soft clamp to POV (renormalize)
    over = q > pov_cap * V
    if np.any(over):
        extra = float(np.sum(q[over] - pov_cap * V[over]))
        q[over] = pov_cap * V[over]
        room = (~blackout) & (~over)
        q[room] += extra * (V[room] / V[room].sum())
    return np.round(q, 6)

def smooth_schedule(X, V, pov_cap, blackout, eta, lam, sigma, iters=500, lr=0.1):
    # Projected gradient on AC objective with equality + box [0, pov*V] on q_t
    n = len(V)
    q = pov_schedule(X, V, min(pov_cap, 1.0), blackout)  # good init
    if np.isscalar(sigma): sigma = np.full(n, float(sigma))
    cap = pov_cap * V
    cap[blackout] = 0.0
    for _ in range(iters):
        inv = np.cumsum(q[::-1])[::-1]
        # d/dq_t impact: eta*(2*q_t/V_t)
        g = 2 * eta * q / (V + 1e-12)
        # d/dq_t risk: lam * 2*sigma^2 * sum_{u<=t} inv_u  (each q_t affects all inv_u with u<=t)
        # Efficiently compute cumulative factor:
        c = 2 * lam * (sigma**2)
        g += np.cumsum(c[::-1])[::-1] * inv  # Hadamard with inv then cum-sum
        # gradient step
        q = q - lr * g
        # project to [0, cap]
        q = np.clip(q, 0.0, cap)
        # project to equality sum(q)=X (on non-blackout mass)
        active = cap > 0
        if active.any():
            # simple shift proportional to capacity weights
            w = np.where(active, cap, 0.0)
            w = np.where(active, w / (w.sum() + 1e-12), 0.0)
            delta = X - q.sum()
            q = q + delta * w
            q = np.clip(q, 0.0, cap)
    return np.round(q, 6)

def load_curve(csv_path):
    import csv
    t, V, blackout = [], [], []
    with open(csv_path) as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            t.append(int(r["t"]))
            V.append(float(r["exp_vol"]))
            blackout.append(r["blackout"].lower() == "true")
    return np.array(V, float), np.array(blackout, bool)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV with t,exp_vol,blackout")
    ap.add_argument("--target-shares", type=int, default=10000)
    ap.add_argument("--sigma", type=float, default=1e-3)
    ap.add_argument("--eta", type=float, default=1e-7)
    ap.add_argument("--lambda-risk", type=float, default=0.05)
    ap.add_argument("--pov-cap", type=float, default=0.2)
    args = ap.parse_args()

    V, blackout = load_curve(args.csv)
    X = args.target_shares

    out = []
    for name, fn in [
        ("pov", pov_schedule),
        ("uniform", uniform_schedule),
        ("smooth", lambda X,V,p,b: smooth_schedule(X,V,p,b,args.eta,args.lambda_risk,args.sigma))
    ]:
        q = fn(X, V.copy(), args.pov_cap, blackout.copy())
        cost = ac_cost(q, V, args.eta, args.lambda_risk, args.sigma)
        pov_viol = int(np.sum(q > args.pov_cap * V + 1e-6))
        blk_viol = int(np.sum(blackout & (q > 1e-12)))
        out.append({"policy": name, "fills": q.tolist(), "ac_cost": cost,
                    "constraints": {"pov_violations": pov_viol, "blackout_violations": blk_viol}})

    best = min(out, key=lambda r: r["ac_cost"])
    print(json.dumps(best))
    with open("baseline_result.json", "w") as f:
        json.dump(best, f)

if __name__ == "__main__":
    main()
