#!/usr/bin/env python3
import csv, argparse, math, random


def u_shape(n, total_vol=5_000_000, noise=0.12):
    # classic intraday U: heavy open/close, lighter midday
    base = []
    for t in range(n):
        x = t/(n-1)
        # U-shape using a simple bowl + taper
        val = 0.35*(1 - (2*x-1)**2) + 0.65*(0.5*(math.cos(2*math.pi*x)+1))
        base.append(max(val, 1e-4))
    s = sum(base)
    vols = [total_vol * b/s for b in base]
    # add noise but keep positive
    out = []
    for v in vols:
        v *= max(0.25, random.gauss(1.0, noise))
        out.append(max(1000.0, v))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=48)
    ap.add_argument("--total-vol", type=float, default=5_000_000)
    ap.add_argument("--noise", type=float, default=0.12)
    ap.add_argument("--blackout-pct", type=float, default=0.0,
                    help="fraction of random slices set to blackout")
    ap.add_argument("--out", default="examples/big_volume_curve.csv")
    args = ap.parse_args()

    vols = u_shape(args.horizon, args.total_vol, args.noise)
    # random sparse blackouts (e.g. auction/halts, venue unavailability)
    import random
    blackout = [False]*args.horizon
    k = int(args.blackout_pct * args.horizon)
    for i in random.sample(range(args.horizon), k):
        blackout[i] = True

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t","exp_vol","blackout"])
        for t,(v,b) in enumerate(zip(vols, blackout)):
            w.writerow([t, int(v), str(b).lower()])
    print("Wrote", args.out)

if __name__ == "__main__":
    main()
