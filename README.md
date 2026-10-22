# Local LLM on Ubuntu toolkit: memory planner, concurrency probe and service unit

Companion to the [iLogr](https://ilogr.com/) article **[Running Local Language Models on Ubuntu: An Operator's Guide](https://ilogr.com/running-local-language-models-ubuntu/)**.

Three small tools for running a self-hosted language model as a service, all in plain Python 3 (standard library only) and plain systemd:

- **`vram_plan.py`** works out how much device memory a serving configuration needs: weights, key/value cache and a runtime reserve you supply. It says whether it fits and how far context length or concurrency can go.
- **`probe.py`** sends streaming chat requests to a local OpenAI-compatible endpoint at a fixed concurrency and reports time to first token and total duration at the median, 95th percentile and maximum. A truncated or errored stream counts as a failure, never as a fast success.
- **`systemd/llm-server.service`** is a hardened single-host unit with an explicit device boundary, for a server such as `llama-server`.

> **Before you use it:** this is example code, provided as it is and without warranty. Use it entirely at your own risk. Run it on a development machine or a staging copy first, never against a service, store or database you cannot afford to disturb, and take a backup of anything you care about. Read the code before you run it. Nobody who wrote or published it accepts responsibility for any loss, damage or downtime.
>
> The planner and the checks only read local files. The probe sends requests to the server you point it at, and refuses any host other than the local machine unless you add `--allow-remote`. The service unit changes your machine only if you install and start it yourself, which this repository never does for you.

## Requirements

- Python 3.10 or later. Python 3.10 is the system Python on Ubuntu 22.04 LTS, the oldest Ubuntu release still in standard support; python.org ended 3.10 support in October 2026, so 3.11 or later is the better choice (Ubuntu 24.04 ships 3.12, Ubuntu 26.04 ships 3.14).
- `bash` for `bin/check.sh`.
- Optional, for the unit check: Docker. Nothing else is needed to read, run or test the Python.
- Not included: a model runtime, weights or a GPU. The tools do arithmetic, talk to a server you provide, or describe a service for one.

## Run it

Check that everything works on your machine (about 15 seconds, no network access beyond stub servers on `127.0.0.1`):

```bash
./bin/check.sh
```

Size a configuration. The example model in `models.example.json` is illustrative: replace every field from the model you actually intend to serve.

```bash
python3 vram_plan.py --spec models.example.json \
    --device 24GiB --reserve 2GiB --context 32768 --concurrency 8
```

Measure a server you run yourself (the prompt file is any text file you choose):

```bash
python3 probe.py --base-url http://127.0.0.1:8080 \
    --model local-model --prompt-file prompt.txt \
    --requests 24 --concurrency 4 --max-tokens 256
```

Check the service unit in throwaway Ubuntu containers. This downloads Ubuntu images and packages inside Docker, so it refuses to run until you confirm:

```bash
ALLOW_DOCKER_DOWNLOADS=yes ./bin/check-unit-in-docker.sh
```

It installs `systemd` inside each container, runs `systemd-analyze verify` and `systemd-analyze security --offline` on the unit, and removes the container afterwards. It starts no service and installs nothing on your machine. Choose releases with `UBUNTU_VERSIONS="24.04 26.04"`.

## Expected results

| Command | Expected |
| --- | --- |
| `vram_plan.py` with the example spec, 24 GiB, 2 GiB reserve, 8,192 tokens, 4 sessions | exit 0, `"fits": true`; at most 17 sessions or 35,671 tokens |
| Same, 32,768 tokens, 8 sessions | exit 1, `"fits": false`, headroom -15,657,418,240 bytes; at most 4 sessions or 17,835 tokens |
| Unreadable spec or non-finite size | exit 2 and a JSON error on stderr |
| `probe.py` against a healthy server | exit 0 and a JSON summary |
| `probe.py` when every request fails | exit 1 and the named error reasons |
| `probe.py --base-url` with a non-local host, no `--allow-remote` | exit 2 |
| `bin/check.sh` | `All checks passed.` |
| `bin/check-unit-in-docker.sh` | `PASS  Ubuntu 22.04`, `24.04` and `26.04`; exposure level 1.5 on systemd 255 and later |

The cache formula behind the planner is `2 x layers x kv_heads x head_dim x bytes_per_element` per token per sequence. For the example model that is 131,072 bytes per token, or 1 GiB for one 8,192-token sequence. All figures come from that arithmetic over an illustrative model shape. **No GPU was measured**, and nothing here is a benchmark.

## Planner contracts

- `--device` is the memory available to this process on the accelerator, not the number on the box.
- `--reserve` is the runtime overhead **outside the weights and the key/value cache**: activations, the CUDA context, compute buffers and allocator fragmentation. **Do not pass a total resident device-memory reading.** That already contains the weights, and on a runtime that preallocates its cache it contains that too, so it would be counted twice. There is deliberately no default reserve.
- `weight_file_bytes` should be the total size of the downloaded weight files. The `parameter_count` and `bits_per_weight` fallback is an estimate and the output says `"weights_estimated": true`.
- The cache formula covers multi-head, grouped-query and multi-query attention only. Other designs are refused rather than given a confident wrong number.
- `max_concurrency_at_requested_context` and `max_context_at_requested_concurrency` assume every resident sequence reaches the full context length.

## Probe contracts

- A request counts as successful only if the stream ends the way the protocol documents, with `[DONE]`. A truncated stream, a server error event, a malformed event or an exceeded deadline is a failure with a named reason. Read the failure count and the error list before quoting any latency.
- `--timeout` is a wall-clock deadline for the whole request, not an inactivity timeout. A server that trickles bytes is cut off at the deadline.
- A redirect is refused, because it could send a local-only run to another host.
- An API key is read from the `LLM_API_KEY` environment variable only; there is no command-line option, so it does not enter your shell history. The probe refuses to send a key over plain `http` to a non-local host.
- `--concurrency` above 256 is refused. A run against a non-local host (with `--allow-remote`) prints a one-line warning first.
- Stream chunks are not tokens. Take token counts from the runtime's own metrics. One run measures one prompt shape, one output limit, one concurrency level and one server build, so run it at 1, 2, 4 and 8 and read the curve.

## Service unit

`systemd/llm-server.service` is a starting point, not something to paste unexamined. Review every path, the account, the runtime and the flags, then validate it on the target host:

```bash
systemd-analyze verify /etc/systemd/system/llm-server.service
systemd-analyze security llm-server.service
```

The settings that carry comments, because each is a common and slow-to-diagnose failure:

- `PrivateDevices=yes` is absent. It would remove the accelerator device nodes, after which the runtime either fails to start or falls back to the CPU without saying so. The unit sets `DevicePolicy=closed` and names the devices instead; adjust the `/dev/nvidia0` entry for the devices you have.
- `MemoryMax=` is unset. A host-memory cap that clips pinned staging buffers during model load produces an out-of-memory kill rather than a readable error. Set one only after measuring a completed load.
- `ProcSubset=all` is used because some runtimes read `/proc` entries that the stricter setting hides.
- With `Type=exec`, a started unit does not mean the model is loaded, so there is no `TimeoutStartSec=`. Poll the runtime's `/health` endpoint for readiness.
- `--ctx-size 32768 --parallel 4` is the configuration the planner supports for the example model on a 24 GiB device. It is not a default; use your own plan's numbers and confirm from the runtime's load-time report that the model is executing on the accelerator.
- The `/dev/nvidia*` nodes must exist before the service starts. `NoNewPrivileges=yes` stops the runtime creating them through the setuid `nvidia-modprobe` helper, so run `nvidia-persistenced` on the host.

The unit binds the server to loopback and expects a reverse proxy in front that terminates TLS and authenticates callers. It does not install a driver, fetch weights, manage model versions or provide authentication. Installing it means creating the `llm-server` account, the `/opt/llm-server` and `/var/lib/llm-server/models` paths and the runtime binary, copying the file into `/etc/systemd/system/` and running `systemctl daemon-reload`. Do that by hand, on a machine you are prepared to change, after reading the file.

## What is in this repository

| Path | What it is |
| --- | --- |
| `vram_plan.py` | The memory planner (command line and importable). |
| `models.example.json` | The illustrative model specification the guide uses. |
| `probe.py` | The concurrency probe. |
| `systemd/llm-server.service` | The service unit. |
| `tests/test_toolkit.py` | 72 checks: arithmetic, both command lines, stub servers (plain and chunked streams, truncation, error events, redirects, trickled bytes against the deadline) and static properties of the unit. |
| `bin/check.sh` | Runs the tests and the printed examples through the real command lines. |
| `bin/check-unit-in-docker.sh` | Verifies and scores the unit in throwaway Ubuntu containers. Needs `ALLOW_DOCKER_DOWNLOADS=yes`. |
| `LICENSE` | MIT. |

## Tested

- Python 3.9 to 3.15 (the unit tests pass on all of them; 3.10 is the supported minimum, see Requirements). The latest stable release at the time of writing was Python 3.15.1.
- Ubuntu 22.04.5 LTS (systemd 249), 24.04.5 LTS (systemd 255) and 26.04.1 LTS (systemd 259): `systemd-analyze verify` is clean on all three; `systemd-analyze security --offline` needs systemd 253 or later and scores the unit 1.5 (OK) on the latter two. On Ubuntu 24.04 the unit also started under systemd with a stand-in server, answered on loopback and restarted after a crash.
- The server flags in the unit (`--model`, `--host`, `--port`, `--ctx-size`, `--parallel`, `--metrics`) and the public `/health` endpoint match the current `llama-server` documentation. No GPU run was made, so the device rules and the system-call filter have not been exercised against a real accelerator.

## Licence

MIT, see `LICENSE`. The software is provided "as is", without warranty of any kind, and nobody who wrote or published it accepts liability for its use.

More code that accompanies iLogr guides: [github.com/ilogrblog](https://github.com/ilogrblog).
