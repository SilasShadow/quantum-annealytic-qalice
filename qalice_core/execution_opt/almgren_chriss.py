import numpy as np


def ac_cost(q, V, eta, lam, sigma):
    impact = np.sum(eta * (q / (V + 1e-12)) * q)
    inv = np.cumsum(q[::-1])[::-1]
    risk = lam * np.sum((sigma**2) * inv**2)
    return float(impact + risk)
