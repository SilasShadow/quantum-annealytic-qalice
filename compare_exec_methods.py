#!/usr/bin/env python3
import argparse, json, csv, numpy as np, matplotlib.pyplot as plt

def ac_cost(q, V, eta, lam, sigma):
    q, V = np.asarray(q, float), np.asarray(V, float) + 1e-12
    impact = np.sum(eta * (q / V) * q)
    inv = np.cumsum(q[::-1])[::-1]
    if np.isscalar(sigma):
        sigma = np.full_like(q, float(sigma))
    risk = lam * np.sum((np.asarray(sigma, float) ** 2) * inv**2)
    return float(impact + risk)

def load_json(path):
    with open(path) as f:
        return json.load(f)

def load_volume(csv_path):
    V = []
    with open(csv_path) as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            V.append(float(r["exp_vol"]))
    return np.array(V, float)

def alpha_like(fills, V):
    X = float(sum(fills))
    vnorm = V / (V.sum() + 1e-12)
    fnorm = np.asarray(fills, float) / (X + 1e-12)
    return float(np.sum(np.abs(fnorm - vnorm)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="baseline_result.json")
    ap.add_argument("--sa", default="qubo_sa_result.json")
    ap.add_argument("--sqa", default="qubo_sqa_result.json")
    ap.add_argument("--csv", required=True, help="same volume curve used in runs")
    ap.add_argument("--eta", type=float, default=1e-7)
    ap.add_argument("--lambda-risk", type=float, default=0.05)
    ap.add_argument("--sigma", type=float, default=1e-3)
    args = ap.parse_args()

    V = load_volume(args.csv)
    results = []
    for label, path in [("baseline", args.baseline),
                        ("SA-QUBO", args.sa),
                        ("SQA-QUBO", args.sqa)]:
        try:
            r = load_json(path)
            fills = r["fills"]
            results.append((label, fills,
                            r.get("ac_cost",
                                  ac_cost(fills, V, args.eta,
                                          args.lambda_risk, args.sigma))))
        except Exception as e:
            print(f"[warn] Could not load {label} from {path}: {e}")

    if not results:
        raise SystemExit("No result files found.")

    # metrics
    alpha_vals = [alpha_like(f, V) for _, f, _ in results]
    ac_vals = [c for _, _, c in results]
    labels = [lbl for lbl, _, _ in results]

    # comparison bars
    plt.figure(figsize=(9,4))
    plt.subplot(1,2,1)
    plt.title("AC cost (lower=better)")
    plt.bar(labels, ac_vals)
    plt.xticks(rotation=20)
    plt.ylabel("Cost")

    plt.subplot(1,2,2)
    plt.title("Alpha-like L1 to volume profile")
    plt.bar(labels, alpha_vals)
    plt.xticks(rotation=20)
    plt.ylabel("Deviation from curve")

    plt.tight_layout()
    plt.savefig("comparison.png", dpi=150)

    # schedule lines
    plt.figure(figsize=(8,4))
    for lbl, fills, _ in results:
        plt.plot(fills, marker='o', label=lbl)
    plt.plot(list(V / V.max() * max(max(f) for _, f, _ in results)),
             linestyle='--', label="volume shape (scaled)")
    plt.title("Fills per slice")
    plt.xlabel("Time bin")
    plt.ylabel("Shares executed")
    plt.legend()
    plt.tight_layout()
    plt.savefig("schedules.png", dpi=150)

    # stdout table
    print("Method,AC_cost,Alpha_like")
    for (lbl, fills, cost), a in zip(results, alpha_vals):
        print(f"{lbl},{cost:.6g},{a:.6g}")

if __name__ == "__main__":
    main()
