from typing import Any

import dimod

try:
    from neal import SimulatedAnnealingSampler

    _HAVE_NEAL = True
except Exception:
    _HAVE_NEAL = False


def solve_qubo(bqm: dimod.BinaryQuadraticModel, num_reads: int = 1000) -> Any:
    if _HAVE_NEAL:
        return SimulatedAnnealingSampler().sample(bqm, num_reads=num_reads).first
    # Fallback to exact (small problems) if neal not installed
    return dimod.ExactSolver().sample(bqm).first
