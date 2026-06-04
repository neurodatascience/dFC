"""Registry shim for dFC validation — delegates discovery to pydfc auto-scan."""

from __future__ import annotations

from typing import Dict, List, Tuple

# CLI shorthand aliases: MEASURE_NAME -> [alias, ...]
_ALIASES: Dict[str, List[str]] = {
    "SlidingWindow": ["sw"],
    "Time-Freq": ["tf", "wtc"],
    "ExponentialWindow": ["ew"],
    "AdaptiveExponentialWindow": ["aew"],
    "MultiscaleWindow": ["msw"],
    "EdgeCoactivation": ["eca"],
    "PhaseLockingWindow": ["plv"],
    "DerivativeWeightedWindow": ["dww"],
    "TemporalDerivativeMultiplication": ["tdm"],
    "ChangepointResetWindow": ["crw"],
    "KalmanCovariance": ["kalman", "kcv"],
    "LaggedMaxCorrelation": ["lmc"],
    "PrecisionShrinkageWindow": ["psw"],
    "RecurrenceKernelDependence": ["rkd"],
    "RandomFourierDependence": ["rfd"],
    "EventSynchronization": ["event"],
    "CopulaTailDependence": ["ctd"],
    "OjaSubspaceConnectivity": ["oja"],
    "GraphDiffusionCoactivation": ["gdc"],
    "CAP": ["cap"],
    "ContinuousHMM": ["chmm"],
    "DiscreteHMM": ["dhmm"],
    "Windowless": ["windowless"],
    "SlidingWindowClustr": ["swc"],
    "PooledKMeansStates": ["pkms"],
    "MiniBatchKMeansStates": ["mbkms"],
    "GaussianMixtureStates": ["gms"],
    "BayesianGaussianMixtureStates": ["bgms"],
    "BirchStates": ["birchstates"],
    "AgglomerativeStates": ["aggstates"],
    "SpectralStates": ["specstates"],
    "LaggedKMeansStates": ["lagkm"],
    "MarkovSmoothedKMeansStates": ["mskms"],
    "MarkovSmoothedGMMStates": ["msgms"],
    "AmplitudeEnvelopeCorrelation": ["aec"],
    "LeadingEigenvectorDynamics": ["led", "leida"],
    "InstantaneousPhaseCoherence": ["ipc"],
    "DynamicPartialCorrelation": ["dypc"],
    "PhaseLagIndexWindow": ["pliw", "pli"],
    "PointProcessConnectivity": ["ppfc"],
    "STFTCoherence": ["stft", "stftcoh"],
    "RobustSlidingWindow": ["rsw"],
    "SynchronyLikelihoodWindow": ["slw"],
    "NMFStates": ["nmf"],
    "DifferentialCoactivationFC": ["diffcoact"],
    "CurvatureCorrelationFC": ["curvcorr"],
    "VolatilityWeightedFC": ["vwfc"],
    "TemporalAsymmetryFC": ["tafc"],
    "PhaseAmplitudeCrossFC": ["pafc"],
    "MutualCompressionFC": ["mcfc"],
    "TangentSpaceFC": ["tsfc"],
    "SpectralSimilarityFC": ["ssfc"],
    "PositiveNegativeAsymmetryFC": ["pnafc"],
    "StateSpaceNeighborhoodFC": ["ssnfc", "knnfc"],
    "DCCConnectivity": ["dcc"],
    "PersistentHomologyFC": ["phfc", "tda"],
    "TimeReversalAsymmetryFC": ["trac"],
    "QuantumMutualInformationFC": ["qmic"],
    "ReservoirEchoStateFC": ["resc"],
    "LocalJacobianCouplingFC": ["ljcc"],
    "SparseCoactivationCodeFC": ["sccc"],
}


def _normalize(name: str) -> str:
    return "".join(c.lower() for c in name if c.isalnum())


def _registry() -> Dict[str, object]:
    from pydfc.multi_analysis_utils import _build_measure_registry

    return _build_measure_registry()


def list_registered_methods() -> List[str]:
    return list(_registry().keys())


def get_method_catalog() -> List[dict]:
    return [
        {
            "id": idx,
            "key": name,
            "aliases": _ALIASES.get(name, []),
            "available": True,
            "reason": "",
        }
        for idx, name in enumerate(_registry(), start=1)
    ]


def get_method_availability() -> Tuple[List[str], Dict[str, str]]:
    """Return (available_measure_names, {}).

    Methods that fail to import are absent from pydfc auto-discovery and will
    simply not appear in the returned list. The empty dict signals no tracked
    unavailability — import failures surface via the registry_instantiation
    sub-check in api_checks.py.
    """
    return list(_registry().keys()), {}


def resolve_method_requests(
    requested: List[str],
) -> Tuple[List[str], List[str], Dict[str, str]]:
    """Resolve CLI names / aliases to MEASURE_NAMEs.

    Returns (found, missing, unavailable). `unavailable` is always empty
    because availability is determined by the conformance check itself.
    """
    alias_map: Dict[str, str] = {}
    for name in _registry():
        alias_map[_normalize(name)] = name
        for alias in _ALIASES.get(name, []):
            alias_map.setdefault(_normalize(alias), name)

    found: List[str] = []
    missing: List[str] = []
    for req in requested:
        canonical = alias_map.get(_normalize(req))
        if canonical:
            found.append(canonical)
        else:
            missing.append(req)
    return found, missing, {}
