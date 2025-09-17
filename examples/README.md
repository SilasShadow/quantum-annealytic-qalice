# Examples

## Synthetic Volume Curve

The `synth_volume_curve.csv` contains a sample trading volume curve with a U-shape pattern (higher volume at market open/close).

### Converting to ExecInput JSON

```python
import pandas as pd
import json

# Read CSV
df = pd.read_csv('synth_volume_curve.csv')

# Convert to ExecInput format
exec_input = {
    "bin_size": 100,
    "target_shares": 10000,
    "horizon": 12,
    "volume_curve": df['exp_vol'].tolist(),
    "blackout_periods": df['blackout'].tolist()
}

print(json.dumps(exec_input))
```

## Quick Demo

```bash
# One-liner to build input and run optimizer
python -c 'import json,csv;curve=[{"t":i,"exp_vol":[120000,90000,70000,60000,55000,50000,52000,60000,70000,85000,100000,130000][i],"blackout":False} for i in range(12)];print(json.dumps({"target_shares":10000,"horizon":12,"bin_size":100,"lambda_risk":0.05,"impact_eta":1e-7,"pov_cap":0.2,"volume_curve":curve}))' \
| xargs -I{} python -m qalice_core.execution_opt.cli plan "{}"
```