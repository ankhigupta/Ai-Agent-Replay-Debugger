"""Phase 4: failure-driven deterministic localization.

    FailureSignal -> LocalizationAnalyzer -> search Trace.context + spans
    (deterministic order) -> first observed occurrence -> Finding + Claim

Localization is observational, not causal: a Finding here says a value was
first observed at a location, was not observed in the available evidence,
or that the evidence was insufficient to tell — never that a span caused
anything, nor who or what was responsible for a value appearing.

Public API: `LocalizationAnalyzer`, `LocalizationTarget`, `ValueType`
(the contract and target vocabulary — see `base.py`), and
`FirstObservedOccurrenceAnalyzer` (the V1 concrete analyzer — see
`first_observed.py`). Internal helpers (`values_match`, the observation
bucket builders, etc.) are not re-exported here.
"""

from __future__ import annotations

from agentdebug.analysis.localization.base import LocalizationAnalyzer, LocalizationTarget, ValueType
from agentdebug.analysis.localization.first_observed import FirstObservedOccurrenceAnalyzer

__all__ = [
    "FirstObservedOccurrenceAnalyzer",
    "LocalizationAnalyzer",
    "LocalizationTarget",
    "ValueType",
]
