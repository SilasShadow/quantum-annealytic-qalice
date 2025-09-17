from typing import List, Optional

from pydantic import BaseModel


class VolumeSlice(BaseModel):
    t: int
    exp_vol: float
    blackout: bool = False


class ExecInput(BaseModel):
    target_shares: int
    horizon: int
    bin_size: int = 100
    lambda_risk: float = 0.1
    impact_eta: float = 1e-6
    pov_cap: Optional[float] = 0.2
    max_slice: Optional[int] = None
    volume_curve: List[VolumeSlice]


class ExecPlan(BaseModel):
    fills: List[int]
    objective: float
