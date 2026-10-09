"""LocalizationAnalyzer contract and the LocalizationTarget/ValueType vocabulary.

Note on package layout: the brief for this phase asked for a single file
`agentdebug/analysis/localization.py` for this abstraction, and separately
`agentdebug/analysis/localization/` (a package) for the concrete analyzer.
A module and a package of the same name cannot coexist at the same import
path, so this is implemented as one package (`agentdebug/analysis/
localization/`) with the abstraction here in `base.py`, re-exported from
`agentdebug/analysis/localization/__init__.py` — so `LocalizationAnalyzer`
is still reachable at the natural `agentdebug.analysis.localization` path,
just as a package attribute rather than a bare module.

LocalizationAnalyzer is deliberately NOT a TraceCheck:

    TraceCheck.run(trace) -> Finding
    LocalizationAnalyzer.run(trace, failure_signal, ...) -> Finding

A TraceCheck answers "what pattern exists in this trace" using only the
trace itself. A LocalizationAnalyzer answers a narrower, externally-posed
question — "where was THIS specific failure target first observed" — and
therefore always requires a FailureSignal (the question), not only a Trace
(the evidence). Making this a separate base class keeps that requirement
visible in the type signature rather than hidden behind a runtime check or
an optional parameter on TraceCheck.

Everything here is observational vocabulary only: a LocalizationTarget
describes *what value, where* to look for; `values_match` describes
*whether two observed values count as the same value*. Neither this module
nor anything built on it may describe *why* a value appeared or assign
responsibility for it.
"""

from __future__ import annotations

import datetime
from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any, Optional

from agentdebug.analysis.check import CheckDefinition
from agentdebug.core.enums import Capability
from agentdebug.core.failure_signal import FailureSignal
from agentdebug.core.finding import Finding
from agentdebug.core.trace import Trace


class ValueType(str, Enum):
    """Conservative, explicitly-opted-into type normalization categories.

    `None` (the default on `LocalizationTarget`) means "no normalization —
    use exact structural equality." A `ValueType` is never inferred by
    sniffing a value's shape or string form; the caller must say so
    explicitly, which is what keeps normalization from silently matching
    unrelated values that merely look similar (e.g. "$100" and 100).
    """

    MONETARY = "MONETARY"
    TEMPORAL = "TEMPORAL"
    IDENTIFIER = "IDENTIFIER"


def _normalize_monetary(value: Any) -> Optional[Decimal]:
    """Conservative monetary normalization: strips common currency symbols
    and thousands separators, then parses as a Decimal. Returns None if
    `value` can't be safely interpreted as a monetary amount at all (never
    guesses)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float, Decimal)):
        return Decimal(str(value))
    if isinstance(value, str):
        cleaned = value.strip()
        for symbol in ("$", "€", "£", "¥"):
            cleaned = cleaned.replace(symbol, "")
        cleaned = cleaned.replace(",", "").strip()
        try:
            return Decimal(cleaned)
        except InvalidOperation:
            return None
    return None


def _normalize_temporal(value: Any) -> Optional[str]:
    """Conservative temporal normalization: accepts `date`/`datetime`
    objects directly, or ISO-8601 strings parseable by the stdlib. No
    fuzzy date parsing, no external dependency. Returns None if `value`
    can't be safely interpreted as a date/datetime."""
    if isinstance(value, (datetime.date, datetime.datetime)):
        return value.isoformat()
    if isinstance(value, str):
        text = value.strip()
        for parser in (datetime.date.fromisoformat, datetime.datetime.fromisoformat):
            try:
                return parser(text).isoformat()
            except ValueError:
                continue
    return None


def _normalize_identifier(value: Any) -> Optional[str]:
    """Conservative identifier normalization: strings and ints are
    coerced to a stripped string for comparison (e.g. so an int id can
    match its string form). Case is preserved — identifiers may be
    case-sensitive in the originating system, and guessing otherwise
    would be exactly the kind of unsafe normalization this is meant to
    avoid."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (str, int)):
        return str(value).strip()
    return None


_NORMALIZERS = {
    ValueType.MONETARY: _normalize_monetary,
    ValueType.TEMPORAL: _normalize_temporal,
    ValueType.IDENTIFIER: _normalize_identifier,
}


def _structural_equal(a: Any, b: Any) -> bool:
    """Exact structural equality: dict key order is irrelevant, list order
    is significant, and no cross-type coincidences (e.g. `True == 1`,
    `1 == 1.0`) count as equal. Never uses `hash()` or object identity.
    """
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(_structural_equal(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        return all(_structural_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    return type(a) is type(b) and a == b


def values_match(candidate: Any, target_value: Any, value_type: Optional[ValueType]) -> bool:
    """Whether `candidate` counts as the same value as `target_value`.

    `value_type=None` -> exact structural equality (see `_structural_equal`).
    Otherwise both values are normalized via the matching `ValueType`
    normalizer; if either fails to normalize, they do NOT match — an
    unsafe/ambiguous normalization never falls back to guessing.
    """
    if value_type is None:
        return _structural_equal(candidate, target_value)
    normalizer = _NORMALIZERS[value_type]
    normalized_candidate = normalizer(candidate)
    normalized_target = normalizer(target_value)
    if normalized_candidate is None or normalized_target is None:
        return False
    return normalized_candidate == normalized_target


@dataclass(frozen=True, kw_only=True)
class LocalizationTarget:
    """What value to look for, and optionally where within a structure.

    Attributes:
        value: The value being searched for.
        field_path: Optional dot-separated path (e.g. "refund.eligible")
            resolved against each candidate structure via nested dict
            lookups only (no list indexing in V1). When None, each
            candidate structure is compared to `value` as a whole.
        value_type: Optional explicit type category enabling conservative
            normalization (see `ValueType`). When None, matching is exact
            structural equality — no guessing.

    This is intentionally new, Phase-4-local vocabulary, not a change to
    `FailureSignal`: `FailureSignal.expected`/`.actual` already hold
    whatever structured value an external evaluator reported, and
    `LocalizationTarget` only adds the *optional* extra addressing
    (`field_path`, `value_type`) a caller may supply on top of that value
    — it never changes what `FailureSignal` means or how it serializes.
    """

    value: Any
    field_path: Optional[str] = None
    value_type: Optional[ValueType] = None

    @classmethod
    def from_failure_signal(cls, failure_signal: FailureSignal) -> "LocalizationTarget":
        """Deterministically derive a target from `failure_signal.actual`.

        Uses only the already-structured `.actual` field — never NLP or
        heuristic parsing of `.detail`, and never an LLM. If the caller
        needs to target a specific field within a structured `.actual` (or
        wants type normalization), they should construct a
        `LocalizationTarget` explicitly instead of relying on this default.
        """
        return cls(value=failure_signal.actual)

    def extract(self, candidate: Any) -> tuple[bool, Any]:
        """Resolve `field_path` against `candidate`.

        Returns `(True, value)` if resolvable, `(True, candidate)` if
        `field_path` is None, or `(False, None)` if any path segment is
        missing or an intermediate value isn't a dict. A failed resolution
        means "not present in this candidate", not an error — the caller
        simply treats it as no match here and keeps searching elsewhere.
        """
        if self.field_path is None:
            return True, candidate
        current = candidate
        for segment in self.field_path.split("."):
            if not isinstance(current, dict) or segment not in current:
                return False, None
            current = current[segment]
        return True, current

    def matches(self, candidate: Any) -> bool:
        """Whether `candidate` (a span input/output, context field, etc.)
        contains an observation of this target."""
        found, value = self.extract(candidate)
        if not found:
            return False
        return values_match(value, self.value, self.value_type)


class LocalizationAnalyzer(ABC):
    """Base class for failure-driven localization analyzers.

    `definition` reuses Phase 3's `CheckDefinition` shape — the same
    static-metadata concept (id, basis, required capabilities, known
    false-positive modes) applies unchanged to a localization analyzer.
    """

    definition: CheckDefinition

    @abstractmethod
    def run(
        self,
        trace: Trace,
        failure_signal: FailureSignal,
        *,
        target: Optional[LocalizationTarget] = None,
        available_capabilities: Optional[frozenset[Capability]] = None,
    ) -> Finding:
        """Localize `target` (or one derived from `failure_signal` via
        `LocalizationTarget.from_failure_signal` if `target` is omitted)
        within `trace`'s evidence.

        Must be deterministic: the same `trace`, `failure_signal.actual`
        (or explicit `target`), and `available_capabilities` must always
        produce a Finding with identical analytical content (`status`,
        `claims`, `capability_gaps`, `diagnosis_confidence`) across
        repeated calls — only `Finding.id` may differ. `triggering_signal_id`
        must always be set to `failure_signal.id`.
        """
        raise NotImplementedError
