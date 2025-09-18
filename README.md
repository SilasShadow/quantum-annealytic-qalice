# Quantum-Inspired Execution Optimizer

This project implements a **quantum-inspired optimizer** for large equity trades, based on the **Almgren–Chriss optimal execution model**.
It minimizes **implementation shortfall** (negative alpha) by splitting trades across time slices while respecting volume curves, blackout windows, and participation caps.

---

## Features
- **Almgren–Chriss baseline model** (deterministic cost + risk).
- **QUBO formulation** of trade scheduling.
- **Quantum-inspired solvers**:
  - [dimod](https://docs.ocean.dwavesys.com/en/stable/docs_dimod/)
  - [neal](https://docs.ocean.dwavesys.com/en/stable/docs_neal/) (simulated annealing, local)
  - [openjij](https://openjij.github.io/OpenJij/) (alternative annealer)
- **CLI interface** powered by [Typer](https://typer.tiangolo.com/).
- **Examples**: synthetic intraday volume curves.

---

## Installation

Clone the repo and install base dependencies:

```bash
git clone https://github.com/SilasShadow/quantum-annealytic-qalice.git
cd quantum-annealytic-qalice

# base runtime
pip install -r requirements.txt

# dev tools (tests, linting, formatting)
pip install -r requirements-dev.txt

# optional solvers
pip install -r requirements-optional.txt
```

### Example usage:
1. Prepare a volume curve (example provided):
```bash
cat examples/synth_volume_curve.csv
```
2. Run the optimizer:
```bash
python -m qalice_core.execution_opt.cli plan-file examples/synth_volume_curve.csv

python -m qalice_core.execution_opt.cli plan \
  '{"target_shares":10000,"horizon":12,"bin_size":100,"lambda_risk":0.05,"impact_eta":1e-7,"pov_cap":0.2,
    "volume_curve":[{"t":0,"exp_vol":120000,"blackout":false}, ...]}'
```
3. Output:
```json
{"fills": [800, 600, ..., 900], "objective": 12345.67}
```

### Development
Run tests:
```bash
pytest -q
```
Format & Lint:
```bash
black .
flake8 .
```
Pre-commit hooks:
```bash
pre-commit install
```

## Roadmap

- Support live market data ingestion.

- Visualization of trade trajectories (qalice_viz).

- Integration with broker APIs.

- Experiment with hybrid solvers (Qiskit / D-Wave).
