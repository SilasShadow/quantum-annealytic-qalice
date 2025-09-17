import dimod


def build_qubo(inp):
    N = inp.horizon
    B = (inp.target_shares // inp.bin_size) + 2
    linear, quad = {}, {}

    # Linear terms: impact-proxy + POV + blackout penalties
    for t in range(N):
        exp_vol = float(inp.volume_curve[t].exp_vol)
        for k in range(B):
            v = f"x_{t}_{k}"
            base = k * inp.bin_size
            # quadratic impact approximated into linear per-bin cost
            linear[v] = linear.get(v, 0.0) + inp.impact_eta * (base**2)
            # POV soft-penalty
            pov_cap = (inp.pov_cap or 1.0) * exp_vol
            if base > pov_cap:
                linear[v] += 10.0 * (base - pov_cap)
            # blackout hard-penalty
            if inp.volume_curve[t].blackout:
                linear[v] += 1e6

    # Buy-all hard penalty: (sum b_k x - X)^2
    strength = 1000.0
    vars_list, coeffs = [], []
    for t in range(N):
        for k in range(B):
            v = f"x_{t}_{k}"
            vars_list.append(v)
            coeffs.append(k * inp.bin_size)

    # Expand square: linear
    for i, v in enumerate(vars_list):
        linear[v] = linear.get(v, 0.0) + strength * (
            coeffs[i] ** 2 - 2 * coeffs[i] * inp.target_shares
        )

    # Quadratic cross terms
    for i in range(len(vars_list)):
        for j in range(i + 1, len(vars_list)):
            quad[(vars_list[i], vars_list[j])] = (
                quad.get((vars_list[i], vars_list[j]), 0.0)
                + 2 * strength * coeffs[i] * coeffs[j]
            )

    return dimod.BinaryQuadraticModel(linear, quad, 0.0, vartype=dimod.BINARY)
