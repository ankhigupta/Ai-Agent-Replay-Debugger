"""Tests for TraceContext."""

from __future__ import annotations

import dataclasses
import unittest

from agentdebug.core.context import TraceContext
from agentdebug.core.exceptions import TraceValidationError


class TraceContextTests(unittest.TestCase):
    def test_defaults(self) -> None:
        ctx = TraceContext()
        self.assertEqual(ctx.initial_state, {})
        self.assertIsNone(ctx.user_input)
        self.assertIsNone(ctx.system_prompt)
        self.assertEqual(ctx.config, {})

    def test_valid_construction(self) -> None:
        ctx = TraceContext(
            initial_state={"refund_eligible": None},
            user_input="Can I get a refund?",
            system_prompt="You are a refund assistant.",
            config={"model": "gpt-4o"},
        )
        self.assertIsNone(ctx.initial_state["refund_eligible"])
        self.assertEqual(ctx.user_input, "Can I get a refund?")

    def test_invalid_initial_state_fails(self) -> None:
        with self.assertRaises(TraceValidationError):
            TraceContext(initial_state={"bad": object()})

    def test_invalid_config_fails(self) -> None:
        with self.assertRaises(TraceValidationError):
            TraceContext(config={"bad": object()})

    def test_invalid_user_input_fails(self) -> None:
        with self.assertRaises(TraceValidationError):
            TraceContext(user_input=object())

    def test_invalid_system_prompt_fails(self) -> None:
        with self.assertRaises(TraceValidationError):
            TraceContext(system_prompt=123)  # type: ignore[arg-type]

    def test_frozen(self) -> None:
        ctx = TraceContext()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            ctx.system_prompt = "nope"  # type: ignore[misc]

    def test_round_trip_dict(self) -> None:
        ctx = TraceContext(
            initial_state={"a": 1}, user_input="hi", system_prompt="sp", config={"x": True}
        )
        restored = TraceContext.from_dict(ctx.to_dict())
        self.assertEqual(ctx, restored)

    def test_from_dict_rejects_unknown_field(self) -> None:
        with self.assertRaises(TraceValidationError):
            TraceContext.from_dict({"bogus": 1})


if __name__ == "__main__":
    unittest.main()
