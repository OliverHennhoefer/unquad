import importlib
from pathlib import Path

import pytest

import nonconform

CURATED_ROOT_EXPORTS = [
    "ConformalDetector",
    "Split",
    "DerandomizedSplits",
    "CrossValidation",
    "JackknifeBootstrap",
    "Empirical",
    "Probabilistic",
    "logistic_weight_estimator",
    "forest_weight_estimator",
]

REMOVED_ROOT_SYMBOLS = [
    "PYOD_AVAILABLE",
    "Aggregation",
    "AnomalyDetector",
    "BaseConformalDetector",
    "BaseEstimation",
    "BaseStrategy",
    "BaseWeightEstimator",
    "BootstrapBaggedWeightEstimator",
    "ConformalResult",
    "Distribution",
    "IdentityWeightEstimator",
    "Kernel",
    "Pruning",
    "PyODAdapter",
    "SklearnWeightEstimator",
    "adapt",
    "aggregate",
    "false_discovery_rate",
    "statistical_power",
    "weighted_false_discovery_control",
]

PUBLIC_MODULE_EXPORTS = {
    "nonconform.adapters": [
        "PYOD_AVAILABLE",
        "PyODAdapter",
        "ScorePolarityAdapter",
        "adapt",
        "apply_score_polarity",
        "parse_score_polarity",
        "resolve_implicit_score_polarity",
        "resolve_score_polarity",
    ],
    "nonconform.detector": [
        "BaseConformalDetector",
        "ConformalDetector",
    ],
    "nonconform.enums": [
        "ConformalMode",
        "Distribution",
        "Kernel",
        "Pruning",
        "ScorePolarity",
        "TieBreakMode",
    ],
    "nonconform.fdr": [
        "EValueSelectionResult",
        "FDPCertificate",
        "FPRCertificate",
        "Pruning",
        "conformal_e_values",
        "e_value_false_discovery_control",
        "select_conformal_e_values",
        "weighted_false_discovery_control",
        "weighted_false_discovery_control_from_arrays",
    ],
    "nonconform.martingales": [
        "AlarmConfig",
        "BaseMartingale",
        "MartingaleState",
        "MixtureMartingale",
        "PowerMartingale",
        "SimpleJumperMartingale",
        "SimpleMixtureMartingale",
    ],
    "nonconform.monitoring": [
        "ExchangeabilityMonitor",
        "MonitorState",
        "SequentialRankConformalizer",
    ],
    "nonconform.metrics": [
        "aggregate",
        "false_discovery_rate",
        "statistical_power",
    ],
    "nonconform.resampling": [
        "BaseStrategy",
        "CrossValidation",
        "DerandomizedSplits",
        "JackknifeBootstrap",
        "Split",
    ],
    "nonconform.scoring": [
        "BaseEstimation",
        "ConditionalEmpirical",
        "Empirical",
        "Kernel",
        "Probabilistic",
        "calculate_p_val",
        "calculate_weighted_p_val",
    ],
    "nonconform.structures": [
        "AnomalyDetector",
        "ConformalResult",
    ],
    "nonconform.weighting": [
        "EPSILON",
        "BaseWeightEstimator",
        "BootstrapBaggedWeightEstimator",
        "IdentityWeightEstimator",
        "ProbabilisticClassifier",
        "SklearnWeightEstimator",
        "forest_weight_estimator",
        "logistic_weight_estimator",
    ],
}


def test_root_all_is_exact_curated_surface():
    assert nonconform.__all__ == CURATED_ROOT_EXPORTS


def test_package_declares_pep_561_type_marker():
    assert Path(nonconform.__file__).with_name("py.typed").is_file()


def test_star_import_exposes_only_curated_symbols():
    namespace: dict[str, object] = {}
    exec("from nonconform import *", namespace)
    exported = {name for name in namespace if not name.startswith("__")}
    assert exported == set(CURATED_ROOT_EXPORTS)


@pytest.mark.parametrize(
    ("module_name", "symbol"),
    [("nonconform", symbol) for symbol in REMOVED_ROOT_SYMBOLS]
    + [
        ("nonconform.enums", "Aggregation"),
        ("nonconform.fdr", "conformal_e_value_selection"),
        ("nonconform._internal.constants", "Dataset"),
    ],
)
def test_removed_symbol_is_neither_attribute_nor_importable(
    module_name: str, symbol: str
):
    assert not hasattr(importlib.import_module(module_name), symbol)
    with pytest.raises(ImportError, match=symbol):
        exec(f"from {module_name} import {symbol}", {})


@pytest.mark.parametrize(
    ("module_name", "expected_exports"),
    PUBLIC_MODULE_EXPORTS.items(),
)
def test_public_module_exports_match_contract(
    module_name: str, expected_exports: list[str]
):
    module = importlib.import_module(module_name)
    namespace: dict[str, object] = {}
    exec(f"from {module_name} import *", namespace)

    assert module.__all__ == expected_exports
    for symbol in expected_exports:
        assert hasattr(module, symbol)
    exported = {name for name in namespace if not name.startswith("__")}
    assert exported == set(expected_exports)


def test_conditional_empirical_import_contract():
    namespace: dict[str, object] = {}
    exec("from nonconform.scoring import ConditionalEmpirical", namespace)
    assert callable(namespace["ConditionalEmpirical"])
    assert not hasattr(nonconform, "ConditionalEmpirical")

    with pytest.raises(ImportError, match="ConditionalEmpirical"):
        exec("from nonconform import ConditionalEmpirical", {})
