"""Tests for agentdebug.recording.context: contextvars-based active state."""

from __future__ import annotations

import asyncio
import unittest

from agentdebug.recording import context


class ActiveSpanContextTests(unittest.TestCase):
    def setUp(self) -> None:
        # Context vars persist across test methods in the same thread
        # (unittest doesn't run each test in its own contextvars.Context),
        # so clear explicitly to keep tests independent.
        context.clear()

    def test_no_active_span_initially(self) -> None:
        self.assertIsNone(context.get_current_span_id())

    def test_setting_active_span(self) -> None:
        token = context.bind_active_span_id("s1")
        try:
            self.assertEqual(context.get_current_span_id(), "s1")
        finally:
            context.reset_active_span_id(token)

    def test_nested_span_context(self) -> None:
        token1 = context.bind_active_span_id("parent")
        try:
            self.assertEqual(context.get_current_span_id(), "parent")
            token2 = context.bind_active_span_id("child")
            try:
                self.assertEqual(context.get_current_span_id(), "child")
            finally:
                context.reset_active_span_id(token2)
            self.assertEqual(context.get_current_span_id(), "parent")
        finally:
            context.reset_active_span_id(token1)

    def test_previous_context_restored_after_exit(self) -> None:
        outer_token = context.bind_active_span_id("outer")
        inner_token = context.bind_active_span_id("inner")
        context.reset_active_span_id(inner_token)
        self.assertEqual(context.get_current_span_id(), "outer")
        context.reset_active_span_id(outer_token)

    def test_context_cleared_after_outermost_span(self) -> None:
        token = context.bind_active_span_id("only")
        context.reset_active_span_id(token)
        self.assertIsNone(context.get_current_span_id())

    def test_independent_asyncio_tasks_have_independent_active_span(self) -> None:
        results: dict[str, list[object]] = {"a": [], "b": []}

        async def task_a() -> None:
            token = context.bind_active_span_id("span-a")
            try:
                await asyncio.sleep(0)
                results["a"].append(context.get_current_span_id())
                await asyncio.sleep(0)
                results["a"].append(context.get_current_span_id())
            finally:
                context.reset_active_span_id(token)

        async def task_b() -> None:
            await asyncio.sleep(0)
            token = context.bind_active_span_id("span-b")
            try:
                results["b"].append(context.get_current_span_id())
                await asyncio.sleep(0)
                results["b"].append(context.get_current_span_id())
            finally:
                context.reset_active_span_id(token)

        async def main() -> None:
            await asyncio.gather(task_a(), task_b())

        asyncio.run(main())

        self.assertEqual(results["a"], ["span-a", "span-a"])
        self.assertEqual(results["b"], ["span-b", "span-b"])
        self.assertIsNone(context.get_current_span_id())


if __name__ == "__main__":
    unittest.main()
