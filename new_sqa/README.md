How to use it

Install deps:

pip install numpy dimod openjij


Prepare your CSV (like examples/synth_volume_curve.csv):

t,exp_vol,blackout
0,2000,false
1,1500,false
2,1200,false
3,1200,false
4,1500,false
5,2000,false


Run all methods at once:

python run_optimizations.py --csv examples/synth_volume_curve.csv --target-shares 600 --bin-size 100


You’ll see a table like:

Method,AC_cost,Energy,Total_shares
baseline,0.00123,-,600
qubo_sa,0.00101,-12.3,600
qubo_sqa,0.00095,-11.7,600


and you’ll get JSONs:

baseline_result.json

qubo_sa_result.json

qubo_sqa_result.json

These feed straight into your compare_exec_methods.py plotter.

# Experiment
## Generate data
```bash
python examples/gen_big_curve.py --horizon 48 --total-vol 5_000_000 --noise 0.12 --out examples/big_volume_curve.csv

python examples/gen_big_curve.py --horizon 48 --total-vol 5_000_000 --noise 0.12 --out examples/big_volume_curve.csv
```
### Run Primary Demo (Config A)
```bash
# SA + SQA, auto plots; turn on --fast to use Numba builder
python run_my_code.py --csv examples/big_volume_curve.csv \
  --target-shares 60000 --bin-size 500 --pov-cap 0.2 \
  --eta 1e-7 --lambda-risk 0.05 --sigma 1e-3 \
  --num-reads 500 --sweeps 20000 --fast --auto-plot

  python run_my_code.py --csv examples/small_volume_curve.csv \
  --target-shares 1000 --bin-size 50 --pov-cap 0.2 \
  --eta 1e-7 --lambda-risk 0.05 --sigma 1e-3 \
  --num-reads 30 --sweeps 200 --fast --auto-plot
```

### Stretch Demo (Config B)
```bash
python examples/gen_big_curve.py --horizon 78 --total-vol 8_000_000 --out examples/bigger_curve.csv
python run_my_code.py --csv examples/bigger_curve.csv \
  --target-shares 80000 --bin-size 500 --pov-cap 0.2 \
  --eta 1e-7 --lambda-risk 0.05 --sigma 1e-3 \
  --num-reads 300 --sweeps 15000 --fast --auto-plot
```
