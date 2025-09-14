# Quantum Annealytics — Lattice

Baseline repo for the FinTech Hackathon. This provides a clean Python 3.12 pipeline skeleton you can extend into:
- classical segmentation (k-means, hierarchical, etc.)
- QUBO-ready interfaces (plug in D-Wave/QAOA later)
- feature engineering + evaluation loop

## Quick start

```bash
# 1) Create the conda env
conda env create -f environment.yml
conda activate lattice

# 2) (Optional) Install pre-commit
pre-commit install

# 3) Run tests
pytest -q

# 4) Try the CLI
lattice --help

# 5) Install optional dependencies
conda env update -n lattice -f environment.optional.yml
# OR
pip install -e .[quantum]
# OR
pip install -r requirements-optional.txt
```

# Committing to GitHub

1. Before staging, run:
```bash
pre-commit run --all-files
```

# Build, Execute and Deploy
## Makefile
### How this maps to Maven-ish habits:

- ```make clean``` → ```mvn clean```

- ```make package``` → ```mvn package``` (builds wheel/sdist)

- ```make test / make testv / make test-cov``` → ```mvn test / verbose / with coverage```

- ```make lint / make format``` → quality gates before packaging

- ```make ci``` → mirrors the GitHub Actions steps locally

- ```make env / make extras-conda / make extras-pip``` → quick environment bootstrap like “profiles”

- ```make install``` → like ```mvn install``` (puts your package in the env in editable mode)

```bash
# first time
make env
conda activate lattice
make install
make dev        # installs pre-commit + runs smoke test

# day-to-day
make format
make lint
make test

# add quantum extras (choose one path)
make extras-conda    # or: make extras-pip

# prepare an artifact
make package

# replicate CI locally
make ci
```
