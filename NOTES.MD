Fine-tuning your run (48 slices, 1M volume)

You’ve set:

--target-shares 50000
--bin-size 20
--pov-cap 0.05
--eta 1e-3
--lambda-risk 0.2
--sigma 1e-3
--num-reads 50
--sweeps 500

1. Data size / realism

--horizon 48 → 48 bins. That’s a good balance: not toy-small (10) and not crazy-large (hundreds).

--target-shares 50,000 → with 1M market volume, that’s ~5% participation. Fits the --pov-cap 0.05 nicely.

--bin-size 20 → very fine granularity (up to 2500 possible quanta). This makes the QUBO much larger.

👉 If runs are sluggish, increase bin-size to 50 or 100. That reduces state space.

--pov-cap 0.05 → realistic (max 5% of market per slice).

2. Cost weights

--eta 1e-3 → relatively strong impact cost. This will discourage big trades in thin slices.

--lambda-risk 0.2 → good risk weight; trades should spread out more evenly.

--sigma 1e-3 → small, but consistent with your earlier configs.

3. Annealing parameters

--num-reads 50 and --sweeps 500 → quite light.

For a bigger horizon (48 bins, bin-size 20), this may not explore the space enough.

👉 Try: --num-reads 200 and --sweeps 5000. Still manageable runtime but more exploration.

If too slow, compromise: --num-reads 100 and --sweeps 2000.

⚡ Suggested recipe for “big but not too big”

Keep horizon = 48 (good size).

Use bin-size = 50 or 100 instead of 20 → reduces QUBO size.

Keep target-shares = 50k.

Try num-reads = 100–200, sweeps = 2000–5000 → good annealer coverage without being overnight.

Fine-tune --eta and --lambda-risk:

If schedules still bunch → raise --lambda-risk (0.3–0.5).

If they still crash at the end → also raise --eta (1e-2).
