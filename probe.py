#!/usr/bin/env python3
"""Concurrency probe for a local OpenAI-compatible completion endpoint.

Sends a fixed number of streaming chat completions at a fixed concurrency and
reports time to first token, generation rate and aggregate throughput. It
measures one server under one prompt shape. It is not a model comparison, a
capacity guarantee or a substitute for the runtime's own benchmark tool.

A request counts as successful only if the stream terminates the way the
protocol documents. A truncated stream, a server error event, a malformed
event or an exceeded deadline is recorded as a failure, so a server that fails
quickly cannot be mistaken for a server that is fast.

The timeout is a wall-clock deadline, not an inactivity timeout. The body is
read in bounded chunks and each read is allowed only the time remaining, so a
server trickling an unterminated line cannot hold the request open. Iterating
the response line by line does not have this property: against a stub that
drips one byte every 0.2 seconds it returned after 8.1 seconds under a
0.5 second timeout.

Enforcing that deadline needs the response's socket. If it cannot be reached,
the request fails with `deadline_unavailable` rather than running under a
weaker guarantee than this module documents.
"""
from __future__ import annotations

import argparse
import http.client
import json
import math
import os
import socket
import statistics
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost"}

# One thread per concurrent request. A typo such as 800 instead of 8 would
# otherwise start hundreds of threads against a single local server.
MAX_CONCURRENCY = 256


class ProbeError(RuntimeError):
    pass


class _RefuseRedirect(urllib.request.HTTPRedirectHandler):
    """A redirect could send a "local only" probe to an external host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ProbeError("refused a redirect to %s" % newurl)


_OPENER = urllib.request.build_opener(_RefuseRedirect)

# An unterminated line must not grow without bound while we wait for a newline.
MAX_LINE_BYTES = 1 << 20


def _response_socket(response):
    """The socket behind an HTTP response, or None if it cannot be reached.

    urllib exposes no public way to change a timeout after the request has been
    sent, and the deadline below depends on doing exactly that. If this returns
    None the caller must fail closed rather than continue: without it the socket
    keeps the timeout given to urlopen, and a read that begins just before the
    deadline can run a further full timeout past it. Measured at 1.91 times the
    configured deadline against a server that sends a byte at 0.9 seconds and
    then falls silent, under a 1.0 second timeout.
    """
    try:
        return response.fp.raw._sock
    except AttributeError:
        return None


def _set_read_deadline(sock, remaining: float) -> None:
    """Allow the next read only the time left before the deadline.

    Polling with a short timeout and retrying does not work: a socket file
    object latches after a timeout, and every later read raises
    "cannot read from timed out object". Giving each read exactly the remaining
    budget means a timeout *is* the deadline, so it is reported rather than
    retried.
    """
    sock.settimeout(max(remaining, 0.001))


def percentile(values: List[float], fraction: float) -> float:
    """Nearest-rank percentile over a non-empty list."""
    if not values:
        raise ProbeError("no samples")
    ordered = sorted(values)
    rank = max(1, int(-(-len(ordered) * fraction // 1)))
    return ordered[min(rank, len(ordered)) - 1]


def _delta_text(event: Dict[str, Any]) -> Optional[str]:
    """Extract the content delta, or None if the event shape is unusable."""
    choices = event.get("choices")
    if choices is None:
        return ""
    if not isinstance(choices, list):
        return None
    text = ""
    for choice in choices:
        if not isinstance(choice, dict):
            return None
        delta = choice.get("delta")
        if delta is None:
            continue
        if not isinstance(delta, dict):
            return None
        content = delta.get("content")
        if content is None:
            continue
        if not isinstance(content, str):
            return None
        text += content
    return text


def one_request(
    url: str,
    payload: Dict[str, Any],
    api_key: Optional[str],
    timeout: float,
) -> Dict[str, Any]:
    """Send one streaming request. Always returns an outcome; never raises.

    `timeout` bounds the whole request, not only each blocking read: a server
    that trickles bytes indefinitely is cut off at the deadline.
    """
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if api_key:
        headers["Authorization"] = "Bearer %s" % api_key

    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    started = time.monotonic()
    deadline = started + timeout
    first_token_at = None
    chunks = 0
    saw_done = False

    def fail(reason: str) -> Dict[str, Any]:
        return {"ok": False, "error": reason}

    buffer = b""

    try:
        with _OPENER.open(request, timeout=timeout) as response:
            sock = _response_socket(response)
            if sock is None:
                # Fail closed. Continuing would quietly downgrade the documented
                # wall-clock deadline to roughly twice the configured value.
                return fail("deadline_unavailable")
            while not saw_done:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return fail("deadline_exceeded")
                _set_read_deadline(sock, remaining)
                try:
                    chunk = response.read1(65536)
                except socket.timeout:
                    # The read was given exactly the remaining budget, so a
                    # timeout here means the deadline has passed.
                    return fail("deadline_exceeded")
                except http.client.IncompleteRead:
                    # A chunked stream closed before its terminating chunk.
                    return fail("premature_eof")
                except (http.client.HTTPException, urllib.error.URLError, OSError, ValueError) as error:
                    return fail(type(error).__name__)
                if not chunk:
                    break  # end of stream
                buffer += chunk
                if len(buffer) > MAX_LINE_BYTES:
                    return fail("oversized_line")
                while b"\n" in buffer and not saw_done:
                    raw, buffer = buffer.split(b"\n", 1)
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        saw_done = True
                        break
                    try:
                        event = json.loads(data)
                    except ValueError:
                        return fail("malformed_event")
                    if not isinstance(event, dict):
                        return fail("malformed_event")
                    if event.get("error") is not None:
                        return fail("server_error_event")
                    delta = _delta_text(event)
                    if delta is None:
                        return fail("malformed_event")
                    if not delta:
                        continue
                    chunks += 1
                    if first_token_at is None:
                        first_token_at = time.monotonic()
    except ProbeError:
        return fail("redirect_refused")
    except urllib.error.HTTPError:
        return fail("HTTPError")
    except (http.client.HTTPException, urllib.error.URLError, OSError, ValueError) as error:
        return fail(type(error).__name__)

    if not saw_done:
        # The stream ended without its documented terminator. The answer may be
        # truncated, so the timings describe an incomplete response.
        return fail("premature_eof")
    if first_token_at is None:
        return fail("no_content")

    finished = time.monotonic()
    return {
        "ok": True,
        "ttft_seconds": first_token_at - started,
        "total_seconds": finished - started,
        "generation_seconds": finished - first_token_at,
        "stream_chunks": chunks,
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    parsed = urllib.parse.urlparse(args.base_url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ProbeError("base-url must be an http or https URL")
    if parsed.hostname not in LOCAL_HOSTS and not args.allow_remote:
        raise ProbeError(
            "refusing to load a non-local host without --allow-remote; "
            "only probe a server you are responsible for"
        )
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        raise ProbeError("timeout must be a finite positive number of seconds")
    api_key = os.environ.get("LLM_API_KEY") or None
    is_local = parsed.hostname in LOCAL_HOSTS
    if api_key and parsed.scheme == "http" and not is_local:
        raise ProbeError(
            "refusing to send LLM_API_KEY over plain http to a non-local host; "
            "use an https URL"
        )
    if not is_local:
        print(
            "warning: sending %d streaming requests to non-local host %s"
            % (args.requests, parsed.hostname),
            file=sys.stderr,
        )

    url = args.base_url.rstrip("/") + "/v1/chat/completions"
    with open(args.prompt_file, "r", encoding="utf-8") as handle:
        prompt = handle.read()

    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "stream": True,
    }

    results: List[Dict[str, Any]] = []
    lock = threading.Lock()
    remaining = [args.requests]

    def worker() -> None:
        while True:
            with lock:
                if remaining[0] <= 0:
                    return
                remaining[0] -= 1
            try:
                outcome = one_request(url, payload, api_key, args.timeout)
            except BaseException as error:  # noqa: BLE001
                # Every attempted request must produce a recorded outcome, or a
                # run that lost workers would report a misleadingly small total.
                outcome = {"ok": False, "error": "internal_%s" % type(error).__name__}
            with lock:
                results.append(outcome)

    threads = [threading.Thread(target=worker) for _ in range(args.concurrency)]
    wall_start = time.monotonic()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    wall_seconds = time.monotonic() - wall_start

    successes = [r for r in results if r["ok"]]
    failures = [r for r in results if not r["ok"]]
    summary: Dict[str, Any] = {
        "concurrency": args.concurrency,
        "requests": args.requests,
        "attempted": len(results),
        "successful": len(successes),
        "failed": len(failures),
        "errors": sorted({r["error"] for r in failures}),
        "wall_seconds": round(wall_seconds, 3),
    }
    if len(results) != args.requests:
        summary["accounting_error"] = (
            "recorded %d outcomes for %d requests" % (len(results), args.requests)
        )
    if not successes:
        return summary

    ttft = [r["ttft_seconds"] for r in successes]
    total = [r["total_seconds"] for r in successes]
    chunk_rates = [
        r["stream_chunks"] / r["generation_seconds"]
        for r in successes
        if r["generation_seconds"] > 0
    ]
    summary.update({
        "completed_per_second": round(len(successes) / wall_seconds, 3),
        "ttft_seconds": {
            "p50": round(percentile(ttft, 0.50), 3),
            "p95": round(percentile(ttft, 0.95), 3),
            "max": round(max(ttft), 3),
        },
        "total_seconds": {
            "p50": round(percentile(total, 0.50), 3),
            "p95": round(percentile(total, 0.95), 3),
            "max": round(max(total), 3),
        },
        "stream_chunks_per_second_mean": (
            round(statistics.fmean(chunk_rates), 3) if chunk_rates else None
        ),
        "caveats": [
            "Timings cover successfully terminated streams only. Read the "
            "failure count and error list before quoting any latency.",
            "Stream chunks are not tokens. Use the runtime's own counters for "
            "a token rate.",
            "One prompt shape, one output limit, one server build. Rerun after "
            "any change to the model, quantisation, context setting or flags.",
        ],
    })
    return summary


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="total seconds allowed for one request, not only per read",
    )
    parser.add_argument(
        "--allow-remote",
        action="store_true",
        help="permit a host other than localhost",
    )
    args = parser.parse_args(argv)

    if args.requests <= 0 or args.concurrency <= 0 or args.max_tokens <= 0:
        print(json.dumps({"error": "requests, concurrency and max-tokens must be positive"}), file=sys.stderr)
        return 2
    if args.concurrency > MAX_CONCURRENCY:
        print(json.dumps({"error": "concurrency above %d is refused" % MAX_CONCURRENCY}), file=sys.stderr)
        return 2

    try:
        summary = run(args)
    except (ProbeError, OSError) as error:
        print(json.dumps({"error": str(error)}), file=sys.stderr)
        return 2

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary.get("successful") else 1


if __name__ == "__main__":
    sys.exit(main())
