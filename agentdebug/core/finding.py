"""Finding: the result of running one deterministic check against a Trace.

Status-conditioned shape (enforced in `__post_init__`):

    FLAGGED         claims non-empty, capability_gaps empty
    NO_FINDING      claims empty, OR every claim is VALUE_NOT_OBSERVED;
                    capability_gaps empty
    NOT_APPLICABLE  claims empty, capability_gaps non-empty
    INCONCLUSIVE    capability_gaps non-empty, claims may be either

This mirrors the postures a deterministic check can end up in: it found
something (FLAGGED), it looked and found nothing (NO_FINDING), it couldn't
run at all for lack of evidence (NOT_APPLICABLE), or it ran but with
incomplete evidence, so the result can't be trusted either way
(INCONCLUSIVE).

Note: a check searching for a value and not finding it anywhere in the
available evidence is NOT automatically FLAGGED — "not observed" is a
NO_FINDING with a VALUE_NOT_OBSERVED claim, not a defect in its own right.
This is the one narrow exception to "NO_FINDING means empty claims": a
VALUE_NOT_OBSERVED claim records the fact that a specific search was
carried out and came up empty (e.g. by Phase 3's First Observed Occurrence
analyzer), which is itself a legitimate, factual, non-causal claim — it is
still not a defect finding. Any other claim type remains forbidden on a
NO_FINDING Finding.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from agentdebug.core.claim import Claim
from agentdebug.core.enums import Capability, ClaimType, DiagnosisConfidence, InternalCheckStatus
from agentdebug.core.exceptions import TraceValidationError
from agentdebug.core.serialization import check_known_keys, enum_from_value, require_dict

_ALLOWED_KEYS = frozenset(
    {
        "id",
        "check_id",
        "status",
        "claims",
        "capability_gaps",
        "diagnosis_confidence",
        "triggering_signal_id",
    }
)


@dataclass(frozen=True, kw_only=True)
class Finding:
    """The result of running one deterministic check against a Trace.

    `check_id` is a plain string naming which check produced this Finding.
    Deliberately not an enum and not backed by a registry here — the set
    of checks is expected to grow, and a `CheckDefinition`/registry
    concept belongs to a later analysis layer, not to this core model.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    check_id: str
    status: InternalCheckStatus
    claims: list[Claim] = field(default_factory=list)
    capability_gaps: list[Capability] = field(default_factory=list)
    diagnosis_confidence: Optional[DiagnosisConfidence] = None
    triggering_signal_id: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.check_id, str) or not self.check_id:
            raise TraceValidationError("Finding.check_id must be a non-empty string")
        if self.triggering_signal_id is not None and (
            not isinstance(self.triggering_signal_id, str) or not self.triggering_signal_id
        ):
            raise TraceValidationError(
                "Finding.triggering_signal_id must be a non-empty string or None"
            )

        if self.status == InternalCheckStatus.FLAGGED:
            if not self.claims:
                raise TraceValidationError("FLAGGED Finding must have non-empty claims")
            if self.capability_gaps:
                raise TraceValidationError("FLAGGED Finding must have empty capability_gaps")
        elif self.status == InternalCheckStatus.NO_FINDING:
            if self.claims and any(c.type != ClaimType.VALUE_NOT_OBSERVED for c in self.claims):
                raise TraceValidationError(
                    "NO_FINDING Finding must have empty claims, or claims that are "
                    "all ClaimType.VALUE_NOT_OBSERVED"
                )
            if self.capability_gaps:
                raise TraceValidationError("NO_FINDING Finding must have empty capability_gaps")
        elif self.status == InternalCheckStatus.NOT_APPLICABLE:
            if self.claims:
                raise TraceValidationError("NOT_APPLICABLE Finding must have empty claims")
            if not self.capability_gaps:
                raise TraceValidationError(
                    "NOT_APPLICABLE Finding must have non-empty capability_gaps"
                )
        elif self.status == InternalCheckStatus.INCONCLUSIVE:
            if not self.capability_gaps:
                raise TraceValidationError(
                    "INCONCLUSIVE Finding must have non-empty capability_gaps"
                )
            # claims may be empty or non-empty for INCONCLUSIVE.

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "check_id": self.check_id,
            "status": self.status.value,
            "claims": [c.to_dict() for c in self.claims],
            "capability_gaps": [c.value for c in self.capability_gaps],
            "diagnosis_confidence": (
                self.diagnosis_confidence.value if self.diagnosis_confidence is not None else None
            ),
            "triggering_signal_id": self.triggering_signal_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Finding":
        data = require_dict(data, "Finding")
        check_known_keys(data, _ALLOWED_KEYS, "Finding")
        try:
            kwargs: dict[str, Any] = {
                "check_id": data["check_id"],
                "status": enum_from_value(InternalCheckStatus, data["status"], "status", "Finding"),
                "claims": [Claim.from_dict(c) for c in data.get("claims", [])],
                "capability_gaps": [
                    enum_from_value(Capability, cg, "capability_gaps", "Finding")
                    for cg in data.get("capability_gaps", [])
                ],
                "triggering_signal_id": data.get("triggering_signal_id"),
            }
        except KeyError as exc:
            raise TraceValidationError(f"Finding missing required field: {exc}") from exc

        raw_confidence = data.get("diagnosis_confidence")
        kwargs["diagnosis_confidence"] = (
            enum_from_value(
                DiagnosisConfidence, raw_confidence, "diagnosis_confidence", "Finding"
            )
            if raw_confidence is not None
            else None
        )
        if "id" in data:
            kwargs["id"] = data["id"]
        return cls(**kwargs)
