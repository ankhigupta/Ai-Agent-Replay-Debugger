"""CheckDefinition and TraceCheck: the Phase 3 deterministic-check contract.

A `TraceCheck` inspects one already-recorded `Trace` and returns exactly
one `Finding` describing whether a specific, narrowly-defined evidence
pattern was observed. A check never decides whether the agent was
"correct", never diagnoses a failure's cause, and never performs causal
or statistical inference — see each concrete check's module docstring for
the exact pattern it looks for and what it deliberately does not claim.

Checks are stateless: a `TraceCheck` instance must be safe to call
`run(trace)` on any number of unrelated Traces, in any order, without one
call's result depending on a previous call. No check may store the Trace
it was given, or anything derived from it, on `self`.

Capability awareness: nothing in Phase 1/Phase 2 currently declares which
`Capability` values a given Trace's originating integration can provide —
that is a future integration-profile concept, not yet built. Until it
exists, `run()` accepts an explicit, optional `available_capabilities`
parameter: the capabilities the caller asserts are available for this
Trace. Omitting it (`None`, the default) means "assume every capability
this check requires is available" — i.e. judge purely from the Trace's
evidence, with no capability gate exercised. This is a deliberate, minimal
stand-in; see `agentdebug.analysis`'s module docstring for why.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from agentdebug.core.enums import Capability
from agentdebug.core.finding import Finding
from agentdebug.core.trace import Trace


@dataclass(frozen=True, kw_only=True)
class CheckDefinition:
    """Static metadata describing a check — not analysis logic itself.

    Attributes:
        id: Stable string identifier for the check; matches the
            `Finding.check_id` it produces. Deliberately a plain string,
            not an enum or registry entry — see Phase 1's
            `Finding.check_id` docstring for why.
        basis: Concise, factual description of what observable evidence
            the check looks for. Describes the pattern, not a verdict.
        required_capabilities: The `Capability` values this check needs
            an integration to provide at all. Their absence means the
            check cannot run against this Trace's originating integration
            at all (-> `NOT_APPLICABLE`), as distinct from the capability
            being present but this particular Trace lacking the specific
            evidence needed (-> `INCONCLUSIVE`).
        known_false_positive_modes: Documented, general limitations of
            the check — ways it might flag something that isn't actually
            a defect. Static documentation about the check's design,
            never a conclusion about any specific Trace it has analyzed.
    """

    id: str
    basis: str
    required_capabilities: tuple[Capability, ...] = field(default_factory=tuple)
    known_false_positive_modes: tuple[str, ...] = field(default_factory=tuple)


def missing_capabilities(
    definition: CheckDefinition, available_capabilities: Optional[frozenset[Capability]]
) -> list[Capability]:
    """Capabilities `definition` requires that `available_capabilities`
    does not include, in `definition.required_capabilities` order
    (deterministic). `available_capabilities=None` means "assume
    everything required is available" -> always returns `[]`.
    """
    if available_capabilities is None:
        return []
    return [c for c in definition.required_capabilities if c not in available_capabilities]


class TraceCheck(ABC):
    """Base class every deterministic Phase 3 check implements.

    `definition` is a class-level `CheckDefinition`. `run` performs the
    (deterministic) analysis and returns exactly one `Finding`.
    """

    definition: CheckDefinition

    @abstractmethod
    def run(
        self, trace: Trace, *, available_capabilities: Optional[frozenset[Capability]] = None
    ) -> Finding:
        """Analyze `trace` and return a Finding.

        Must be deterministic: given the same `trace` content and the
        same `available_capabilities`, two calls — even on two separate
        instances of this check — must produce Findings with identical
        `status`, `claims`, `capability_gaps`, and `diagnosis_confidence`.
        Only `Finding.id` may differ, since it is an identity field, not
        analytical content.
        """
        raise NotImplementedError
