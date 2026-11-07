"""Tests for agentkit. Standard library unittest only; no network,
no API keys. Run with:  python -m unittest discover -s tests -v
"""

import json
import os
import shutil
import sys
import tempfile
import time
import unittest
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agentkit import AgentRunner, StubProvider, Task, Tool, ValidationError
from agentkit.model import ToolCall
from agentkit.providers import ProviderResponse
from agentkit.providers import OpenAICompatibleProvider
from agentkit.report import build_trace, render_text_report
from agentkit.tools import ToolRegistry, builtin_tools, calculator


def _registry(tmpdir):
    return builtin_tools(notes_dir=tmpdir, base_dir=tmpdir)


def _run(task_text, provider, tmpdir, **kwargs):
    runner = AgentRunner(provider, _registry(tmpdir), **kwargs)
    return runner.run(Task(text=task_text))


class TestCalculator(unittest.TestCase):
    def test_addition(self):
        self.assertEqual(calculator("17 + 25"), "17 + 25 = 42")

    def test_precedence_and_parens(self):
        self.assertEqual(calculator("(3 + 4) * 2"), "(3 + 4) * 2 = 14")

    def test_power_and_modulo(self):
        self.assertEqual(calculator("2 ** 10"), "2 ** 10 = 1024")
        self.assertEqual(calculator("10 % 3"), "10 % 3 = 1")

    def test_rejects_names_and_calls(self):
        for bad in ("__import__('os')", "open('/etc/passwd')", "x + 1", "1; 2"):
            with self.assertRaises(Exception, msg=bad):
                calculator(bad)

    def test_division_by_zero(self):
        with self.assertRaises(Exception):
            calculator("1 / 0")


class TestToolValidation(unittest.TestCase):
    def test_valid_tool_registers(self):
        reg = ToolRegistry()
        reg.register(Tool("echo", "Echoes.", {
            "type": "object", "properties": {"text": {"type": "string"}},
            "required": ["text"]}, func=lambda text: text))
        self.assertIn("echo", reg.names())

    def test_bad_name_rejected(self):
        reg = ToolRegistry()
        with self.assertRaises(ValidationError):
            reg.register(Tool("bad name!", "Bad.", {
                "type": "object", "properties": {"x": {"type": "string"}}},
                func=lambda x: x))

    def test_empty_description_rejected(self):
        reg = ToolRegistry()
        with self.assertRaises(ValidationError):
            reg.register(Tool("ok", "  ", {
                "type": "object", "properties": {"x": {"type": "string"}}},
                func=lambda x: x))

    def test_duplicate_name_rejected(self):
        reg = ToolRegistry()
        tool = Tool("dup", "D.", {
            "type": "object", "properties": {"x": {"type": "string"}}},
            func=lambda x: x)
        reg.register(tool)
        with self.assertRaises(ValidationError):
            reg.register(tool)

    def test_unknown_tool_call_rejected(self):
        reg = ToolRegistry()
        with self.assertRaises(ValidationError):
            reg.call("nope", {})

    def test_args_type_checked(self):
        reg = _registry(tempfile.mkdtemp())
        with self.assertRaises(ValidationError):
            reg.call("calculator", {"expression": 42})

    def test_missing_required_arg(self):
        reg = _registry(tempfile.mkdtemp())
        with self.assertRaises(ValidationError):
            reg.call("calculator", {})

    def test_unknown_arg_rejected(self):
        reg = _registry(tempfile.mkdtemp())
        with self.assertRaises(ValidationError):
            reg.call("calculator", {"expression": "1+1", "extra": 1})

    def test_bool_is_not_integer(self):
        reg = ToolRegistry()
        reg.register(Tool("n", "N.", {
            "type": "object", "properties": {"n": {"type": "integer"}}},
            func=lambda n: str(n)))
        with self.assertRaises(ValidationError):
            reg.call("n", {"n": True})


class TestTaskValidation(unittest.TestCase):
    def test_empty_task_rejected(self):
        with self.assertRaises(ValidationError):
            Task.from_dict({"task": "   "})

    def test_missing_task_rejected(self):
        with self.assertRaises(ValidationError):
            Task.from_dict({})

    def test_max_steps_bounds(self):
        with self.assertRaises(ValidationError):
            Task.from_dict({"task": "x", "max_steps": 0})
        with self.assertRaises(ValidationError):
            Task.from_dict({"task": "x", "max_steps": 101})
        self.assertEqual(Task.from_dict({"task": "x", "max_steps": 5}).max_steps, 5)

    def test_bad_json_file(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            fh.write("{not json")
            path = fh.name
        try:
            with self.assertRaises(ValidationError):
                Task.from_file(path)
        finally:
            os.unlink(path)

    def test_missing_file(self):
        with self.assertRaises(ValidationError):
            Task.from_file("/nonexistent/task.json")


class TestRunnerBehaviour(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_stub_add_task_succeeds(self):
        result = _run("Add 17 and 25", StubProvider(), self.tmpdir)
        self.assertEqual(result.status, "success")
        self.assertIn("42", result.final_answer)
        self.assertEqual(len(result.steps), 2)  # tool call + final answer

    def test_stub_provider_determinism(self):
        r1 = _run("Add 17 and 25", StubProvider(), self.tmpdir)
        r2 = _run("Add 17 and 25", StubProvider(), self.tmpdir)
        self.assertEqual(
            [s.to_dict() for s in r1.steps], [s.to_dict() for s in r2.steps]
        )
        self.assertEqual(r1.final_answer, r2.final_answer)

    def test_step_limit_stop(self):
        script = [
            ProviderResponse(tool_call=ToolCall(
                "calculator", {"expression": f"{i} + 1"}))
            for i in range(20)
        ]
        result = _run("loop forever", StubProvider(script=script),
                      self.tmpdir, max_steps=3)
        self.assertEqual(result.status, "step_limit")
        self.assertEqual(len(result.steps), 3)
        self.assertIsNone(result.final_answer)

    def test_loop_detection(self):
        same = ToolCall("calculator", {"expression": "1 + 1"})
        script = [ProviderResponse(tool_call=same),
                  ProviderResponse(tool_call=same)]
        result = _run("repeat", StubProvider(script=script), self.tmpdir)
        self.assertEqual(result.status, "loop_detected")
        self.assertIn("repeated", result.steps[-1].error)

    def test_unknown_tool_rejected_not_executed(self):
        script = [ProviderResponse(tool_call=ToolCall("rm_rf", {}))]
        result = _run("do bad thing", StubProvider(script=script), self.tmpdir)
        self.assertEqual(result.status, "failed")
        self.assertIn("unknown tool", result.steps[-1].error)
        self.assertEqual(len(result.steps), 1)

    def test_tool_not_in_allowlist_rejected(self):
        script = [ProviderResponse(tool_call=ToolCall(
            "search_notes", {"query": "x"}))]
        runner = AgentRunner(StubProvider(script=script), _registry(self.tmpdir))
        result = runner.run(Task(text="t", tools=["calculator"]))
        self.assertEqual(result.status, "failed")
        self.assertIn("unknown tool", result.steps[-1].error)

    def test_dry_run_executes_nothing(self):
        runner = AgentRunner(StubProvider(), _registry(self.tmpdir))
        result = runner.run(Task(text="Add 1 and 2"), dry_run=True)
        self.assertEqual(result.status, "dry_run")
        self.assertEqual(result.steps, [])
        plan = runner.plan(Task(text="Add 1 and 2"))
        self.assertEqual(plan["provider"], "StubProvider")
        self.assertEqual(
            [t["name"] for t in plan["tools"]],
            ["calculator", "read_file", "search_notes"],
        )

    def test_dry_run_with_unknown_tool_is_invalid(self):
        runner = AgentRunner(StubProvider(), _registry(self.tmpdir))
        with self.assertRaises(ValidationError):
            runner.plan(Task(text="t", tools=["nope"]))

    def test_timeout_handling(self):
        reg = ToolRegistry()
        reg.register(Tool("slow", "Sleeps.", {
            "type": "object", "properties": {"s": {"type": "number"}}},
            func=lambda s: time.sleep(s) or "done"))
        script = [ProviderResponse(tool_call=ToolCall("slow", {"s": 5}))]
        runner = AgentRunner(StubProvider(script=script), reg)
        started = time.monotonic()
        result = runner.run(Task(text="slow", timeout_seconds=0.2))
        self.assertEqual(result.status, "timeout")
        self.assertLess(time.monotonic() - started, 3.0)

    def test_args_size_limit(self):
        big = "x" * 5000
        script = [ProviderResponse(tool_call=ToolCall(
            "calculator", {"expression": big}))]
        runner = AgentRunner(StubProvider(script=script),
                             _registry(self.tmpdir), max_args_bytes=100)
        result = runner.run(Task(text="big"))
        self.assertEqual(result.status, "failed")
        self.assertIn("exceed", result.steps[-1].error)

    def test_tool_error_fails_run(self):
        script = [ProviderResponse(tool_call=ToolCall(
            "calculator", {"expression": "1 / 0"}))]
        result = _run("divide", StubProvider(script=script), self.tmpdir)
        self.assertEqual(result.status, "failed")
        self.assertIn("division by zero", result.steps[-1].error)

    def test_provider_error_fails_run(self):
        class Broken:
            def complete(self, messages, tools):
                raise RuntimeError("boom")
        result = _run("x", Broken(), self.tmpdir)
        self.assertEqual(result.status, "failed")
        self.assertIn("provider error", result.steps[-1].error)


class TestSandboxAndNotes(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        with open(os.path.join(self.tmpdir, "a.md"), "w") as fh:
            fh.write("# Notes\nThe launch date is Friday.\nSecond line.\n")
        with open(os.path.join(self.tmpdir, "b.md"), "w") as fh:
            fh.write("Nothing about launches here.\n")
        os.mkdir(os.path.join(self.tmpdir, "sub"))

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_read_file_inside_sandbox(self):
        reg = _registry(self.tmpdir)
        self.assertIn("Friday", reg.call("read_file", {"path": "a.md"}))

    def test_read_file_escape_rejected(self):
        reg = _registry(self.tmpdir)
        with self.assertRaises(Exception):
            reg.call("read_file", {"path": "../outside.md"})

    def test_read_file_missing(self):
        reg = _registry(self.tmpdir)
        with self.assertRaises(Exception):
            reg.call("read_file", {"path": "nope.md"})

    def test_search_notes_finds_match(self):
        reg = _registry(self.tmpdir)
        out = reg.call("search_notes", {"query": "launch date"})
        self.assertIn("a.md:2", out)
        self.assertIn("Friday", out)

    def test_search_notes_no_match(self):
        reg = _registry(self.tmpdir)
        out = reg.call("search_notes", {"query": "zzzqqq"})
        self.assertIn("No matches", out)

    def test_shell_not_registered_by_default(self):
        reg = _registry(self.tmpdir)
        self.assertNotIn("shell", reg.names())

    def test_shell_opt_in_warns(self):
        reg = builtin_tools(notes_dir=self.tmpdir, base_dir=self.tmpdir,
                            allow_shell=True)
        self.assertIn("shell", reg.names())
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = reg.call("shell", {"command": "echo hi"})
        self.assertEqual(out, "hi")
        self.assertTrue(any(issubclass(w.category, UserWarning) for w in caught))


class TestTraceAndReport(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_trace_contents(self):
        result = _run("Add 17 and 25", StubProvider(), self.tmpdir)
        trace = build_trace(result, Task(text="Add 17 and 25"), "StubProvider")
        self.assertEqual(trace["status"], "success")
        self.assertEqual(trace["provider"], "StubProvider")
        self.assertEqual(len(trace["steps"]), 2)
        step = trace["steps"][0]
        self.assertEqual(step["tool_call"]["name"], "calculator")
        self.assertEqual(step["tool_call"]["arguments"], {"expression": "17 + 25"})
        self.assertIn("42", step["tool_result"])
        self.assertEqual(trace["final_answer"], result.final_answer)
        json.dumps(trace)  # must be JSON-serializable

    def test_text_report_contains_status_and_steps(self):
        result = _run("Add 17 and 25", StubProvider(), self.tmpdir)
        text = render_text_report(result, Task(text="Add 17 and 25"), "StubProvider")
        self.assertIn("status: success", text)
        self.assertIn("calculator", text)
        self.assertIn("42", text)


class TestOpenAIProvider(unittest.TestCase):
    def test_missing_key_raises(self):
        env = dict(os.environ)
        env.pop("AGENTKIT_API_KEY", None)
        old = os.environ.pop("AGENTKIT_API_KEY", None)
        try:
            with self.assertRaises(Exception):
                OpenAICompatibleProvider()
        finally:
            if old is not None:
                os.environ["AGENTKIT_API_KEY"] = old


class TestCLI(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write_task(self, data):
        path = os.path.join(self.tmpdir, "task.json")
        with open(path, "w") as fh:
            json.dump(data, fh)
        return path

    def test_cli_run_success_exit_0(self):
        from agentkit.__main__ import main
        path = self._write_task({"task": "Add 3 and 4"})
        rc = main(["run", path, "--base-dir", self.tmpdir,
                   "--notes-dir", self.tmpdir])
        self.assertEqual(rc, 0)

    def test_cli_dry_run_exit_0(self):
        from agentkit.__main__ import main
        path = self._write_task({"task": "Add 3 and 4"})
        rc = main(["run", path, "--dry-run", "--base-dir", self.tmpdir])
        self.assertEqual(rc, 0)

    def test_cli_invalid_task_exit_2(self):
        from agentkit.__main__ import main
        path = self._write_task({"task": "   "})
        rc = main(["run", path])
        self.assertEqual(rc, 2)

    def test_cli_validate_exit_0(self):
        from agentkit.__main__ import main
        path = self._write_task({"task": "Add 3 and 4"})
        rc = main(["validate", path, "--base-dir", self.tmpdir])
        self.assertEqual(rc, 0)

    def test_cli_step_limit_exit_1(self):
        from agentkit.__main__ import main
        path = self._write_task({"task": "Add 3 and 4", "max_steps": 1})
        # max_steps 1 with the stub: step 1 calls the tool, no step
        # left for the final answer -> step limit.
        rc = main(["run", path, "--base-dir", self.tmpdir,
                   "--notes-dir", self.tmpdir])
        self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
