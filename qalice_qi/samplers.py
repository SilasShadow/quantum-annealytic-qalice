"""Unified interface for BQM sampling with classical SA and SQA."""

import logging
from typing import TYPE_CHECKING, Literal, TypedDict

if TYPE_CHECKING:
    import dimod

logger = logging.getLogger(__name__)


class SamplerConfig(TypedDict):
    """Configuration for BQM sampling."""
    method: Literal["sa", "sqa"]
    sweeps: int
    num_reads: int
    random_state: int
    # SQA-only (ignored by SA):
    beta_start: float   # inverse temperature start (e.g., 0.1)
    beta_end: float     # inverse temperature end (e.g., 5.0)
    gamma_start: float  # transverse field start (e.g., 2.0)
    gamma_end: float    # transverse field end (e.g., 0.01)
    trotter: int        # number of Trotter slices (e.g., 8)


def sample_bqm(bqm: "dimod.BinaryQuadraticModel", cfg: SamplerConfig) -> "dimod.SampleSet":
    """Sample a BQM using either SA or SQA based on configuration.
    
    Args:
        bqm: Binary quadratic model to sample
        cfg: Sampling configuration
        
    Returns:
        Sample set from the chosen sampler
    """
    num_vars = len(bqm.variables)
    num_couplers = len(bqm.quadratic)
    logger.info(f"Sampling BQM with {num_vars} variables, {num_couplers} couplers using {cfg['method'].upper()}")
    
    if cfg["method"] == "sa":
        return _sample_sa(bqm, cfg)
    elif cfg["method"] == "sqa":
        return _sample_sqa(bqm, cfg)
    else:
        raise ValueError(f"Unknown sampling method: {cfg['method']}")


def _sample_sa(bqm: "dimod.BinaryQuadraticModel", cfg: SamplerConfig) -> "dimod.SampleSet":
    """Sample using classical simulated annealing."""
    try:
        import neal
        sampler = neal.SimulatedAnnealingSampler()
        logger.info("Using neal.SimulatedAnnealingSampler")
    except ImportError:
        import dimod.reference.samplers
        sampler = dimod.reference.samplers.SimulatedAnnealingSampler()
        logger.info("Using dimod.reference.samplers.SimulatedAnnealingSampler")
    
    return sampler.sample(
        bqm,
        num_sweeps=cfg["sweeps"],
        num_reads=cfg["num_reads"],
        seed=cfg["random_state"]
    )


def _sample_sqa(bqm: "dimod.BinaryQuadraticModel", cfg: SamplerConfig) -> "dimod.SampleSet":
    """Sample using simulated quantum annealing."""
    try:
        import openjij as oj
        
        # Build linear schedules
        beta_schedule = [[i, cfg["beta_start"] + (cfg["beta_end"] - cfg["beta_start"]) * i / cfg["sweeps"]] 
                        for i in range(cfg["sweeps"] + 1)]
        gamma_schedule = [[i, cfg["gamma_start"] + (cfg["gamma_end"] - cfg["gamma_start"]) * i / cfg["sweeps"]] 
                         for i in range(cfg["sweeps"] + 1)]
        
        logger.info(f"Using OpenJij SQASampler with beta: {cfg['beta_start']}→{cfg['beta_end']}, "
                   f"gamma: {cfg['gamma_start']}→{cfg['gamma_end']}, trotter: {cfg['trotter']}")
        
        sampler = oj.SQASampler()
        sampleset = sampler.sample(
            bqm,
            beta_schedule=beta_schedule,
            gamma_schedule=gamma_schedule,
            trotter=cfg["trotter"],
            num_reads=cfg["num_reads"],
            seed=cfg["random_state"]
        )
        
        # Convert to dimod.SampleSet if needed
        import dimod
        if not isinstance(sampleset, dimod.SampleSet):
            return dimod.SampleSet.from_samples(
                sampleset.samples(),
                vartype=bqm.vartype,
                energy=sampleset.data_vectors['energy']
            )
        return sampleset
        
    except ImportError:
        logger.warning("OpenJij not available, falling back to classical SA")
        return _sample_sa(bqm, cfg)


def default_sa_config(random_state: int = 42) -> SamplerConfig:
    """Default configuration for simulated annealing.
    
    Args:
        random_state: Random seed
        
    Returns:
        SA configuration with sensible defaults
    """
    return SamplerConfig(
        method="sa",
        sweeps=10_000,
        num_reads=64,
        random_state=random_state,
        beta_start=0.1,
        beta_end=5.0,
        gamma_start=2.0,
        gamma_end=0.01,
        trotter=8
    )


def default_sqa_config(random_state: int = 42) -> SamplerConfig:
    """Default configuration for simulated quantum annealing.
    
    Args:
        random_state: Random seed
        
    Returns:
        SQA configuration with sensible defaults
    """
    return SamplerConfig(
        method="sqa",
        sweeps=10_000,
        num_reads=64,
        random_state=random_state,
        beta_start=0.1,
        beta_end=5.0,
        gamma_start=2.0,
        gamma_end=0.01,
        trotter=8
    )