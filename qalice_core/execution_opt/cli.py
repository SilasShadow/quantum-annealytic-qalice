import json

import typer

from .qubo_builder import build_qubo
from .schemas import ExecInput
from .solvers import solve_qubo

app = typer.Typer(help="Execution Optimizer (QUBO) for slice scheduling")


@app.command()
def plan(input_json: str):
    """
    input_json: JSON string matching ExecInput schema.
    Prints JSON: {"fills": [...], "objective": float}
    """
    inp = ExecInput.model_validate_json(input_json)
    bqm = build_qubo(inp)
    best = solve_qubo(bqm)

    # decode fills per time slice
    N = inp.horizon
    B = (inp.target_shares // inp.bin_size) + 2
    fills = []
    for t in range(N):
        q = 0
        for k in range(B):
            if best.sample.get(f"x_{t}_{k}", 0) == 1:
                q += k * inp.bin_size
        fills.append(q)

    print(json.dumps({"fills": fills, "objective": float(best.energy)}))
