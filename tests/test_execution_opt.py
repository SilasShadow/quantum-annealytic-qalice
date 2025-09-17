import csv, json, math
import pytest
from qalice_core.execution_opt.schemas import ExecInput, VolumeSlice
from qalice_core.execution_opt.qubo_builder import build_qubo
from qalice_core.execution_opt.solvers import solve_qubo

def load_curve(path="examples/synth_volume_curve.csv"):
    out = []
    with open(path) as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            out.append(VolumeSlice(t=int(r["t"]), exp_vol=float(r["exp_vol"]), blackout=r["blackout"].lower()=="true"))
    return out

@pytest.mark.skipif(False, reason="basic smoke test")
def test_exec_plan_smoke():
    curve = load_curve()
    inp = ExecInput(
        target_shares=10000, horizon=len(curve), bin_size=100,
        lambda_risk=0.05, impact_eta=1e-7, pov_cap=0.2, volume_curve=curve
    )
    bqm = build_qubo(inp)
    best = solve_qubo(bqm, num_reads=500)
    B = (inp.target_shares // inp.bin_size) + 2
    fills = []
    for t in range(inp.horizon):
        q = 0
        for k in range(B):
            if best.sample.get(f"x_{t}_{k}", 0) == 1:
                q += k * inp.bin_size
        fills.append(q)

    assert len(fills) == inp.horizon
    # allow some slack due to penalty approximation
    assert abs(sum(fills) - inp.target_shares) <= 200
    # POV cap soft-check
    for t, q in enumerate(fills):
        cap = (inp.pov_cap or 1.0) * curve[t].exp_vol + 200
        assert q <= cap