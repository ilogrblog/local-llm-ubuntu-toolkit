#!/usr/bin/env python3
"""Deterministic checks for the sizing planner and the concurrency probe.

No GPU, model, network egress or serving runtime is required. The probe tests
use a local stub that speaks the streaming response shape.
"""
from __future__ import annotations

import contextlib
import http.server
import io
import json
import math
import os
import re
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import probe  # noqa: E402
import vram_plan  # noqa: E402

SPEC = {
    "name": "test-model",
    "attention": "gqa",
    "num_hidden_layers": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "weight_file_bytes": 4 * 1024 ** 3,
}


def spec(**overrides):
    merged = dict(SPEC)
    merged.update(overrides)
    for key, value in list(merged.items()):
        if value is None:
            del merged[key]
    return merged


class KeyValueCacheTests(unittest.TestCase):
    def test_bytes_per_token_matches_the_stated_formula(self):
        # 2 entries * 32 layers * 8 kv heads * 128 dims * 2 bytes.
        self.assertEqual(vram_plan.kv_bytes_per_token(spec(), "f16"), 131072)

    def test_cache_dtype_scales_the_cache_linearly(self):
        f16 = vram_plan.kv_bytes_per_token(spec(), "f16")
        f32 = vram_plan.kv_bytes_per_token(spec(), "f32")
        self.assertEqual(f32, f16 * 2)

    def test_quantised_cache_type_uses_its_block_overhead(self):
        # q8_0 stores 32 weights plus a 16-bit scale: 1.0625 bytes per element.
        expected = math.ceil(2 * 32 * 8 * 128 * 1.0625)
        self.assertEqual(vram_plan.kv_bytes_per_token(spec(), "q8_0"), expected)

    def test_unknown_cache_dtype_is_refused(self):
        with self.assertRaises(vram_plan.SpecError):
            vram_plan.kv_bytes_per_token(spec(), "int3")

    def test_unsupported_attention_family_is_refused(self):
        with self.assertRaises(vram_plan.SpecError):
            vram_plan.kv_bytes_per_token(spec(attention="mla"), "f16")

    def test_attention_family_is_case_insensitive(self):
        self.assertEqual(
            vram_plan.kv_bytes_per_token(spec(attention="GQA"), "f16"), 131072
        )


class WeightTests(unittest.TestCase):
    def test_measured_file_size_is_not_marked_estimated(self):
        size, estimated = vram_plan.weight_bytes(spec())
        self.assertEqual(size, 4 * 1024 ** 3)
        self.assertFalse(estimated)

    def test_parameter_count_fallback_is_marked_estimated(self):
        size, estimated = vram_plan.weight_bytes(
            spec(weight_file_bytes=None, parameter_count=1000, bits_per_weight=4.5)
        )
        self.assertEqual(size, math.ceil(1000 * 4.5 / 8))
        self.assertTrue(estimated)

    def test_missing_weight_information_is_refused(self):
        with self.assertRaises(vram_plan.SpecError):
            vram_plan.weight_bytes(spec(weight_file_bytes=None))

    def test_boolean_is_not_accepted_as_a_byte_count(self):
        with self.assertRaises(vram_plan.SpecError):
            vram_plan.weight_bytes(spec(weight_file_bytes=True))


class PlanTests(unittest.TestCase):
    def build(self, **kwargs):
        arguments = {
            "spec": spec(),
            "device_bytes": 24 * 1024 ** 3,
            "reserve_bytes": 2 * 1024 ** 3,
            "context_tokens": 8192,
            "concurrency": 4,
        }
        arguments.update(kwargs)
        return vram_plan.plan(**arguments)

    def test_totals_add_up(self):
        result = self.build()
        parts = result["bytes"]
        self.assertEqual(
            parts["required"],
            parts["weights"] + parts["kv_cache_total"] + parts["operator_reserve"],
        )
        self.assertEqual(parts["headroom"], parts["device"] - parts["required"])

    def test_a_reasonable_configuration_fits(self):
        self.assertTrue(self.build()["fits"])

    def test_raising_concurrency_eventually_does_not_fit(self):
        result = self.build(concurrency=64)
        self.assertFalse(result["fits"])
        self.assertIn(
            "does not fit on the declared device.",
            " ".join(result["notes"]),
        )

    def test_reported_maximum_concurrency_is_the_last_one_that_fits(self):
        result = self.build()
        ceiling = result["limits_at_this_reserve"]["max_concurrency_at_requested_context"]
        self.assertTrue(self.build(concurrency=ceiling)["fits"])
        self.assertFalse(self.build(concurrency=ceiling + 1)["fits"])

    def test_reported_maximum_context_is_the_last_one_that_fits(self):
        result = self.build()
        ceiling = result["limits_at_this_reserve"]["max_context_at_requested_concurrency"]
        self.assertTrue(self.build(context_tokens=ceiling)["fits"])
        self.assertFalse(self.build(context_tokens=ceiling + 1)["fits"])

    def test_reserve_reduces_the_available_cache_budget(self):
        small = self.build(reserve_bytes=0)
        large = self.build(reserve_bytes=8 * 1024 ** 3)
        self.assertGreater(
            small["limits_at_this_reserve"]["max_concurrency_at_requested_context"],
            large["limits_at_this_reserve"]["max_concurrency_at_requested_context"],
        )

    def test_weights_larger_than_the_device_report_no_cache_budget(self):
        result = self.build(spec=spec(weight_file_bytes=64 * 1024 ** 3))
        self.assertFalse(result["fits"])
        self.assertEqual(
            result["limits_at_this_reserve"]["max_concurrency_at_requested_context"], 0
        )

    def test_thin_headroom_is_called_out(self):
        fitted = self.build()
        exact = self.build(device_bytes=fitted["bytes"]["required"] + 1024)
        self.assertTrue(exact["fits"])
        self.assertIn("Headroom is under five per cent", " ".join(exact["notes"]))

    def test_the_reserve_note_forbids_a_total_memory_reading(self):
        notes = " ".join(self.build()["notes"])
        self.assertIn("must exclude the weights", notes)
        self.assertIn("Do not pass a total resident device-memory reading", notes)
        self.assertIn("validate it under the intended prompt", notes)

    def test_invalid_dimensions_are_refused(self):
        for bad in ({"context_tokens": 0}, {"concurrency": 0}, {"device_bytes": 0}, {"reserve_bytes": -1}):
            with self.subTest(bad=bad):
                with self.assertRaises(vram_plan.SpecError):
                    self.build(**bad)

    def test_missing_required_spec_fields_are_named(self):
        with self.assertRaises(vram_plan.SpecError) as caught:
            self.build(spec={"name": "x"})
        self.assertIn("num_hidden_layers", str(caught.exception))

    def test_non_object_spec_is_refused(self):
        with self.assertRaises(vram_plan.SpecError):
            self.build(spec=["not", "an", "object"])


class SizeParsingTests(unittest.TestCase):
    def test_binary_and_decimal_suffixes_differ(self):
        self.assertEqual(vram_plan._parse_size("1GiB"), 1024 ** 3)
        self.assertEqual(vram_plan._parse_size("1GB"), 1000 ** 3)
        self.assertEqual(vram_plan._parse_size("24G"), 24 * 1024 ** 3)

    def test_plain_byte_counts_are_accepted(self):
        self.assertEqual(vram_plan._parse_size(" 2048 "), 2048)

    def test_nonsense_is_refused(self):
        with self.assertRaises(ValueError):
            vram_plan._parse_size("some")

    def test_non_finite_sizes_are_refused_within_the_documented_contract(self):
        # These raise OverflowError if parsed naively, which would escape the exit-code 2 path.
        for bad in ("infGiB", "nanGiB", "-5GiB", "1e400GiB", "infinity"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    vram_plan._parse_size(bad)

    def test_non_finite_bits_per_weight_is_refused(self):
        for bad in (float("inf"), float("nan")):
            with self.subTest(bad=bad):
                with self.assertRaises(vram_plan.SpecError):
                    vram_plan.weight_bytes(
                        spec(weight_file_bytes=None, parameter_count=1000, bits_per_weight=bad))


class CommandLineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.spec_path = Path(self.temp.name) / "spec.json"
        self.spec_path.write_text(json.dumps(spec()), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def run_cli(self, *extra):
        return subprocess.run(
            [sys.executable, str(ROOT / "vram_plan.py"), "--spec", str(self.spec_path)]
            + list(extra),
            capture_output=True,
            text=True,
            timeout=30,
        )

    def test_exit_zero_when_the_configuration_fits(self):
        done = self.run_cli(
            "--device", "24GiB", "--reserve", "2GiB", "--context", "8192", "--concurrency", "4"
        )
        self.assertEqual(done.returncode, 0)
        self.assertTrue(json.loads(done.stdout)["fits"])

    def test_exit_one_when_the_configuration_does_not_fit(self):
        done = self.run_cli(
            "--device", "8GiB", "--reserve", "2GiB", "--context", "32768", "--concurrency", "16"
        )
        self.assertEqual(done.returncode, 1)
        self.assertFalse(json.loads(done.stdout)["fits"])

    def test_exit_two_on_a_non_finite_size(self):
        done = self.run_cli(
            "--device", "infGiB", "--reserve", "2GiB", "--context", "1024", "--concurrency", "1"
        )
        self.assertEqual(done.returncode, 2)
        self.assertIn("error", json.loads(done.stderr))

    def test_exit_two_on_an_unreadable_spec(self):
        done = subprocess.run(
            [
                sys.executable,
                str(ROOT / "vram_plan.py"),
                "--spec", str(Path(self.temp.name) / "absent.json"),
                "--device", "24GiB",
                "--reserve", "2GiB",
                "--context", "1024",
                "--concurrency", "1",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(done.returncode, 2)
        self.assertIn("error", json.loads(done.stderr))

    def test_shipped_example_spec_is_usable(self):
        done = subprocess.run(
            [
                sys.executable,
                str(ROOT / "vram_plan.py"),
                "--spec", str(ROOT / "models.example.json"),
                "--device", "24GiB",
                "--reserve", "2GiB",
                "--context", "8192",
                "--concurrency", "2",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertIn(done.returncode, (0, 1))
        self.assertIn("kv_cache_total", done.stdout)


class ArticleExampleTests(unittest.TestCase):
    """The commands and figures the guide prints, run through the shipped CLI."""

    def run_example(self, context, concurrency):
        return subprocess.run(
            [sys.executable, str(ROOT / "vram_plan.py"), "--spec", str(ROOT / "models.example.json"),
             "--device", "24GiB", "--reserve", "2GiB",
             "--context", str(context), "--concurrency", str(concurrency)],
            capture_output=True, text=True, timeout=30,
        )

    def test_eight_sessions_at_32768_tokens_do_not_fit(self):
        done = self.run_example(32768, 8)
        self.assertEqual(done.returncode, 1)
        result = json.loads(done.stdout)
        self.assertFalse(result["fits"])
        self.assertEqual(result["bytes"]["weights"], 4920000000)
        self.assertEqual(result["bytes"]["kv_cache_per_sequence"], 4294967296)
        self.assertEqual(result["bytes"]["kv_cache_total"], 34359738368)
        self.assertEqual(result["bytes"]["headroom"], -15657418240)
        self.assertEqual(result["limits_at_this_reserve"]["max_concurrency_at_requested_context"], 4)
        self.assertEqual(result["limits_at_this_reserve"]["max_context_at_requested_concurrency"], 17835)

    def test_four_sessions_at_8192_tokens_fit(self):
        done = self.run_example(8192, 4)
        self.assertEqual(done.returncode, 0)
        result = json.loads(done.stdout)
        self.assertTrue(result["fits"])
        self.assertEqual(result["bytes"]["kv_cache_per_token_per_sequence"], 131072)
        self.assertEqual(result["bytes"]["kv_cache_per_sequence"], 1024 ** 3)


class ServiceUnitTests(unittest.TestCase):
    """Static checks on the properties the guide says the unit has."""

    @classmethod
    def setUpClass(cls):
        text = (ROOT / "systemd" / "llm-server.service").read_text(encoding="utf-8")
        cls.lines = [
            line.strip() for line in text.splitlines()
            if line.strip() and not line.strip().startswith(("#", ";"))
        ]
        cls.exec_start = " ".join(
            part.rstrip("\\").strip()
            for part in text.split("ExecStart=", 1)[1].split("Restart=", 1)[0].splitlines()
        )

    def test_the_device_boundary_is_named_not_removed(self):
        self.assertNotIn("PrivateDevices=yes", self.lines)
        self.assertIn("DevicePolicy=closed", self.lines)
        self.assertTrue(any(line.startswith("DeviceAllow=/dev/nvidia0 ") for line in self.lines))

    def test_the_host_memory_cap_and_start_timeout_are_unset(self):
        self.assertFalse(any(line.startswith(("MemoryMax=", "TimeoutStartSec=")) for line in self.lines))

    def test_readiness_is_not_claimed_by_the_service_type(self):
        self.assertIn("Type=exec", self.lines)

    def test_the_server_listens_on_loopback_only(self):
        self.assertIn("--host 127.0.0.1", self.exec_start)
        self.assertIn("IPAddressDeny=any", self.lines)
        self.assertIn("IPAddressAllow=localhost", self.lines)

    def test_the_example_flags_match_the_planner_example(self):
        self.assertIn("--ctx-size 32768", self.exec_start)
        self.assertIn("--parallel 4", self.exec_start)

    def test_the_service_is_unprivileged_and_confined(self):
        for required in ("NoNewPrivileges=yes", "ProtectSystem=strict", "ProtectHome=yes", "User=llm-server"):
            self.assertIn(required, self.lines)

    def test_the_unit_names_no_person_host_or_home_directory(self):
        text = (ROOT / "systemd" / "llm-server.service").read_text(encoding="utf-8")
        for forbidden in ("/home/", "/Users/"):
            self.assertNotIn(forbidden, text)
        self.assertIsNone(re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", text))


class StubHandler(http.server.BaseHTTPRequestHandler):
    mode = "stream"

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        if self.mode == "error":
            self.send_response(500)
            self.end_headers()
            return
        if self.mode == "redirect":
            self.send_response(302)
            self.send_header("Location", "http://model.example.net/v1/chat/completions")
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        if self.mode == "empty":
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return
        if self.mode == "late_byte":
            # One byte just before a 1.0 second deadline, then silence.
            time.sleep(0.9)
            try:
                self.wfile.write(b"x")
                self.wfile.flush()
                time.sleep(4)
            except OSError:
                pass  # the client hung up, which the deadline test expects
            return
        content = 1 if self.mode in ("truncated", "error_event", "list_event", "bad_choice", "trickle") else 3
        for index in range(content):
            frame = {"choices": [{"delta": {"content": "token%d " % index}}]}
            self.wfile.write(("data: %s\n\n" % json.dumps(frame)).encode("utf-8"))
            self.wfile.flush()
        if self.mode == "error_event":
            self.wfile.write(b'data: {"error":{"message":"backend failed"}}\n\n')
            self.wfile.flush()
            return
        if self.mode == "list_event":
            self.wfile.write(b"data: []\n\n")
            self.wfile.flush()
            return
        if self.mode == "bad_choice":
            self.wfile.write(b'data: {"choices":"not-a-list"}\n\n')
            self.wfile.flush()
            return
        if self.mode == "trickle":
            # An unterminated line, dripped slowly. A line-oriented reader waits
            # for a newline that never arrives and ignores its own deadline.
            try:
                for _ in range(60):
                    self.wfile.write(b"x")
                    self.wfile.flush()
                    time.sleep(0.2)
            except Exception:  # noqa: BLE001 - client hung up, which is the point
                pass
            return
        if self.mode == "truncated":
            return  # EOF with no [DONE] terminator
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def log_message(self, *args):  # silence the stub
        return


class ChunkedStubHandler(http.server.BaseHTTPRequestHandler):
    """HTTP/1.1 with chunked transfer coding, which is how llama-server streams."""

    protocol_version = "HTTP/1.1"
    mode = "chunked"

    def _chunk(self, data):
        self.wfile.write(b"%x\r\n" % len(data) + data + b"\r\n")
        self.wfile.flush()

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        try:
            frame = {"choices": [{"delta": {"content": "token "}}]}
            self._chunk(("data: %s\n\n" % json.dumps(frame)).encode("utf-8"))
            if self.mode == "chunked_trickle":
                for _ in range(60):
                    self._chunk(b"x")
                    time.sleep(0.2)
                return
            if self.mode == "chunked_truncated":
                return  # connection closed with no terminating chunk, no [DONE]
            self._chunk(b"data: [DONE]\n\n")
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except OSError:
            pass  # the client hung up, which the deadline test expects

    def log_message(self, *args):  # silence the stub
        return


class ProbeTests(unittest.TestCase):
    def serve(self, mode):
        base = ChunkedStubHandler if mode.startswith("chunked") else StubHandler
        handler = type("Handler", (base,), {"mode": mode})
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.shutdown)
        self.addCleanup(server.server_close)
        return "http://127.0.0.1:%d" % server.server_address[1]

    def test_nearest_rank_percentiles(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        self.assertEqual(probe.percentile(values, 0.50), 5.0)
        self.assertEqual(probe.percentile(values, 0.95), 10.0)
        self.assertEqual(probe.percentile([4.0], 0.95), 4.0)

    def test_empty_sample_set_is_refused(self):
        with self.assertRaises(probe.ProbeError):
            probe.percentile([], 0.5)

    def test_streamed_response_reports_first_token_and_chunks(self):
        base = self.serve("stream")
        result = probe.one_request(
            base + "/v1/chat/completions", {"stream": True}, None, 10.0
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["stream_chunks"], 3)
        self.assertGreaterEqual(result["total_seconds"], result["ttft_seconds"])

    def test_a_stream_with_no_content_is_a_failure(self):
        base = self.serve("empty")
        result = probe.one_request(
            base + "/v1/chat/completions", {"stream": True}, None, 10.0
        )
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "no_content")

    def test_server_error_is_recorded_not_raised(self):
        base = self.serve("error")
        result = probe.one_request(
            base + "/v1/chat/completions", {"stream": True}, None, 10.0
        )
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "HTTPError")

    def test_a_full_run_summarises_successes(self):
        base = self.serve("stream")
        prompt = Path(self.make_prompt())
        namespace = probe.argparse.Namespace(
            base_url=base,
            model="stub",
            prompt_file=str(prompt),
            requests=6,
            concurrency=3,
            max_tokens=8,
            timeout=10.0,
            allow_remote=False,
        )
        summary = probe.run(namespace)
        self.assertEqual(summary["successful"], 6)
        self.assertEqual(summary["failed"], 0)
        self.assertIn("p95", summary["ttft_seconds"])
        self.assertTrue(any("not tokens" in note for note in summary["caveats"]))

    def test_a_truncated_stream_is_a_failure_not_a_fast_success(self):
        base = self.serve("truncated")
        result = probe.one_request(base + "/v1/chat/completions", {"stream": True}, None, 10.0)
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "premature_eof")

    def test_a_trickled_unterminated_line_cannot_outlive_the_deadline(self):
        # A reader that iterates line by line checks the clock only after a
        # complete line arrives, so this stub held a 0.5 second request open
        # for 8.1 seconds.
        base = self.serve("trickle")
        for timeout in (0.4, 0.8):
            with self.subTest(timeout=timeout):
                started = time.monotonic()
                result = probe.one_request(
                    base + "/v1/chat/completions", {"stream": True}, None, timeout
                )
                elapsed = time.monotonic() - started
                self.assertFalse(result["ok"])
                self.assertEqual(result["error"], "deadline_exceeded")
                # Generous bound: the defect returned at 12s regardless of timeout.
                self.assertLess(elapsed, timeout + 1.0)

    def test_a_chunked_http_1_1_stream_is_read_to_its_terminator(self):
        base = self.serve("chunked")
        result = probe.one_request(base + "/v1/chat/completions", {"stream": True}, None, 10.0)
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["stream_chunks"], 1)

    def test_a_chunked_stream_cut_off_without_done_is_a_failure(self):
        base = self.serve("chunked_truncated")
        result = probe.one_request(base + "/v1/chat/completions", {"stream": True}, None, 10.0)
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "premature_eof")

    def test_a_trickled_chunked_stream_cannot_outlive_the_deadline(self):
        base = self.serve("chunked_trickle")
        started = time.monotonic()
        result = probe.one_request(base + "/v1/chat/completions", {"stream": True}, None, 0.6)
        elapsed = time.monotonic() - started
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "deadline_exceeded")
        self.assertLess(elapsed, 1.6)

    def test_a_byte_just_before_the_deadline_does_not_buy_another_full_timeout(self):
        # Without a per-read deadline the read that starts at 0.9 seconds is
        # allowed a further full second, so the request ends near 1.9 seconds.
        base = self.serve("late_byte")
        started = time.monotonic()
        result = probe.one_request(base + "/v1/chat/completions", {"stream": True}, None, 1.0)
        elapsed = time.monotonic() - started
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "deadline_exceeded")
        self.assertLess(elapsed, 1.4)

    def test_an_unreachable_socket_fails_closed(self):
        # Without the socket the deadline cannot be enforced per read, and the
        # request would silently run under a weaker guarantee: measured at 1.91
        # times the configured deadline. Refuse rather than degrade.
        base = self.serve("stream")
        original = probe._response_socket
        probe._response_socket = lambda response: None
        try:
            result = probe.one_request(
                base + "/v1/chat/completions", {"stream": True}, None, 10.0
            )
        finally:
            probe._response_socket = original
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "deadline_unavailable")

    def test_the_same_request_succeeds_once_the_socket_is_reachable(self):
        # Guards the check above against passing for the wrong reason.
        base = self.serve("stream")
        result = probe.one_request(
            base + "/v1/chat/completions", {"stream": True}, None, 10.0
        )
        self.assertTrue(result["ok"])

    def test_a_server_error_event_is_a_failure(self):
        base = self.serve("error_event")
        result = probe.one_request(base + "/v1/chat/completions", {"stream": True}, None, 10.0)
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "server_error_event")

    def test_a_non_object_event_is_a_failure_not_a_crash(self):
        base = self.serve("list_event")
        result = probe.one_request(base + "/v1/chat/completions", {"stream": True}, None, 10.0)
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "malformed_event")

    def test_a_non_list_choices_field_is_a_failure(self):
        base = self.serve("bad_choice")
        result = probe.one_request(base + "/v1/chat/completions", {"stream": True}, None, 10.0)
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "malformed_event")

    def test_a_redirect_is_refused_so_local_only_still_holds(self):
        base = self.serve("redirect")
        result = probe.one_request(base + "/v1/chat/completions", {"stream": True}, None, 10.0)
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "redirect_refused")

    def test_every_attempted_request_is_accounted_for(self):
        base = self.serve("list_event")
        namespace = probe.argparse.Namespace(
            base_url=base, model="stub", prompt_file=self.make_prompt(),
            requests=4, concurrency=2, max_tokens=8, timeout=10.0, allow_remote=False,
        )
        summary = probe.run(namespace)
        self.assertEqual(summary["attempted"], 4)
        self.assertEqual(summary["successful"] + summary["failed"], 4)
        self.assertNotIn("accounting_error", summary)

    def test_a_run_of_failures_reports_no_timings(self):
        base = self.serve("truncated")
        namespace = probe.argparse.Namespace(
            base_url=base, model="stub", prompt_file=self.make_prompt(),
            requests=3, concurrency=3, max_tokens=8, timeout=10.0, allow_remote=False,
        )
        summary = probe.run(namespace)
        self.assertEqual(summary["successful"], 0)
        self.assertEqual(summary["failed"], 3)
        self.assertNotIn("ttft_seconds", summary)

    def test_non_finite_or_zero_timeout_is_refused(self):
        for bad in (float("inf"), float("nan"), 0.0, -1.0):
            with self.subTest(timeout=bad):
                namespace = probe.argparse.Namespace(
                    base_url="http://127.0.0.1:8080", model="stub",
                    prompt_file=self.make_prompt(), requests=1, concurrency=1,
                    max_tokens=8, timeout=bad, allow_remote=False,
                )
                with self.assertRaises(probe.ProbeError):
                    probe.run(namespace)

    def test_successful_timings_carry_a_failure_caveat(self):
        base = self.serve("stream")
        namespace = probe.argparse.Namespace(
            base_url=base, model="stub", prompt_file=self.make_prompt(),
            requests=2, concurrency=1, max_tokens=8, timeout=10.0, allow_remote=False,
        )
        summary = probe.run(namespace)
        self.assertTrue(any("failure count" in c for c in summary["caveats"]))

    def test_remote_hosts_require_an_explicit_flag(self):
        namespace = probe.argparse.Namespace(
            base_url="http://model.example.net:8080",
            model="stub",
            prompt_file=self.make_prompt(),
            requests=1,
            concurrency=1,
            max_tokens=8,
            timeout=5.0,
            allow_remote=False,
        )
        with self.assertRaises(probe.ProbeError):
            probe.run(namespace)

    def test_an_api_key_is_never_sent_in_clear_to_a_remote_host(self):
        namespace = probe.argparse.Namespace(
            base_url="http://model.example.test:8080", model="stub",
            prompt_file=self.make_prompt(), requests=1, concurrency=1,
            max_tokens=8, timeout=5.0, allow_remote=True,
        )
        original = os.environ.get("LLM_API_KEY")
        os.environ["LLM_API_KEY"] = "not-a-real-key"
        self.addCleanup(
            lambda: os.environ.pop("LLM_API_KEY", None)
            if original is None else os.environ.__setitem__("LLM_API_KEY", original)
        )
        with self.assertRaises(probe.ProbeError) as caught:
            probe.run(namespace)
        self.assertIn("plain http", str(caught.exception))

    def test_a_remote_run_prints_a_one_line_warning_first(self):
        namespace = probe.argparse.Namespace(
            base_url="https://model.example.test", model="stub",
            prompt_file=self.make_prompt(), requests=1, concurrency=1,
            max_tokens=8, timeout=5.0, allow_remote=True,
        )
        original = probe.one_request
        probe.one_request = lambda *a, **k: {"ok": False, "error": "stubbed"}
        self.addCleanup(setattr, probe, "one_request", original)
        captured = io.StringIO()
        with contextlib.redirect_stderr(captured):
            probe.run(namespace)
        self.assertEqual(len(captured.getvalue().strip().splitlines()), 1)
        self.assertIn("non-local host model.example.test", captured.getvalue())

    def test_the_command_line_runs_end_to_end_against_a_local_stub(self):
        base = self.serve("stream")
        done = subprocess.run(
            [sys.executable, str(ROOT / "probe.py"), "--base-url", base,
             "--model", "stub", "--prompt-file", self.make_prompt(),
             "--requests", "4", "--concurrency", "2", "--max-tokens", "8"],
            capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(done.returncode, 0, done.stderr)
        summary = json.loads(done.stdout)
        self.assertEqual((summary["successful"], summary["failed"]), (4, 0))

    def test_the_command_line_exits_one_when_every_request_fails(self):
        base = self.serve("truncated")
        done = subprocess.run(
            [sys.executable, str(ROOT / "probe.py"), "--base-url", base,
             "--model", "stub", "--prompt-file", self.make_prompt(),
             "--requests", "2", "--concurrency", "1"],
            capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(done.returncode, 1)
        self.assertEqual(json.loads(done.stdout)["errors"], ["premature_eof"])

    def test_the_command_line_refuses_a_remote_host_without_the_flag(self):
        done = subprocess.run(
            [sys.executable, str(ROOT / "probe.py"), "--base-url", "http://model.example.test",
             "--model", "stub", "--prompt-file", self.make_prompt()],
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(done.returncode, 2)
        self.assertIn("--allow-remote", json.loads(done.stderr)["error"])

    def test_the_command_line_has_no_api_key_option(self):
        done = subprocess.run(
            [sys.executable, str(ROOT / "probe.py"), "--model", "stub",
             "--prompt-file", self.make_prompt(), "--api-key", "not-a-real-key"],
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(done.returncode, 2)
        self.assertIn("unrecognized arguments", done.stderr)

    def test_an_absurd_concurrency_is_refused(self):
        done = subprocess.run(
            [sys.executable, str(ROOT / "probe.py"), "--model", "stub",
             "--prompt-file", self.make_prompt(), "--concurrency", "100000"],
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(done.returncode, 2)
        self.assertIn("concurrency above", json.loads(done.stderr)["error"])

    def test_non_positive_run_sizes_exit_two(self):
        done = subprocess.run(
            [
                sys.executable,
                str(ROOT / "probe.py"),
                "--model", "stub",
                "--prompt-file", self.make_prompt(),
                "--requests", "0",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(done.returncode, 2)

    def make_prompt(self):
        temp = tempfile.NamedTemporaryFile(
            "w", suffix=".txt", delete=False, encoding="utf-8"
        )
        temp.write("Summarise the attached support ticket in three sentences.")
        temp.close()
        self.addCleanup(os.unlink, temp.name)
        return temp.name


if __name__ == "__main__":
    unittest.main(verbosity=2)
