#!/usr/bin/env bash
# Checks systemd/llm-server.service with systemd-analyze inside throwaway Ubuntu
# containers (22.04, 24.04 and 26.04 by default). Nothing is installed on your
# machine and no service is started: systemd is installed inside each container,
# the unit is only verified and scored, and the container is removed afterwards.
#
# It downloads Ubuntu images and packages (apt) inside Docker, so it will not run
# until you confirm that:
#     ALLOW_DOCKER_DOWNLOADS=yes ./bin/check-unit-in-docker.sh
# Choose releases with UBUNTU_VERSIONS="24.04 26.04".
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ "${ALLOW_DOCKER_DOWNLOADS:-}" != "yes" ]; then
  echo "This check downloads Ubuntu images and apt packages inside throwaway Docker containers." >&2
  echo "It installs nothing on this machine and starts no service. To run it:" >&2
  echo "    ALLOW_DOCKER_DOWNLOADS=yes $0" >&2
  exit 2
fi
if ! command -v docker >/dev/null 2>&1 || ! docker info >/dev/null 2>&1; then
  echo "error: Docker is not installed or not running." >&2
  exit 2
fi

echo "warning: pulling Ubuntu images and installing systemd inside throwaway containers (no service is started)." >&2

fail=0
for version in ${UBUNTU_VERSIONS:-22.04 24.04 26.04}; do
  echo "== Ubuntu $version"
  log="$(mktemp)"
  docker run --rm -i --label llm-toolkit-unit-check -v "$ROOT/systemd:/src:ro" "ubuntu:$version" bash -s >"$log" 2>&1 <<'INSIDE' || true
set -eu
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq >/dev/null
apt-get install -y -qq systemd >/dev/null 2>&1
echo "systemd $(systemctl --version | head -1 | cut -d' ' -f2) inside this container"
# Stand-ins so verify can resolve the account and the program. Container only.
useradd --system --no-create-home llm-server
mkdir -p /opt/llm-server/bin /var/lib/llm-server/models
printf '#!/bin/sh\n' > /opt/llm-server/bin/llama-server
chmod +x /opt/llm-server/bin/llama-server
cp /src/llm-server.service /tmp/llm-server.service

out="$(systemd-analyze verify /tmp/llm-server.service 2>&1 || true)"
if [ -n "$out" ]; then echo "FAIL  systemd-analyze verify reported:"; echo "$out"; exit 1; fi
echo "PASS  systemd-analyze verify: no warnings"

# Negative control: a misspelled directive must make verify speak up.
sed 's/^NoNewPrivileges=yes/NoNewPrivilegez=yes/' /tmp/llm-server.service > /tmp/broken.service
bad="$(systemd-analyze verify /tmp/broken.service 2>&1 || true)"
if [ -z "$bad" ]; then echo "FAIL  verify did not notice a misspelled directive"; exit 1; fi
echo "PASS  verify flags a misspelled directive (negative control)"

if systemd-analyze security --help 2>&1 | grep -q -- '--offline'; then
  systemd-analyze security --offline=true /tmp/llm-server.service 2>&1 | tail -1
else
  echo "note  this systemd is too old for 'security --offline'; run it on a host with systemd 253 or later"
fi
echo "INSIDE-FINISHED"
INSIDE
  cat "$log"
  # The marker proves the script ran to its end; a container that never read it
  # must not count as a pass.
  if grep -q '^INSIDE-FINISHED$' "$log" && ! grep -q '^FAIL' "$log"; then
    echo "PASS  Ubuntu $version"
  else
    echo "FAIL  Ubuntu $version"; fail=1
  fi
  rm -f "$log"
done
exit "$fail"
