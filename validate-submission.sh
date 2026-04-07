#!/usr/bin/env bash

set -uo pipefail

DOCKER_BUILD_TIMEOUT=600

if [ -t 1 ]; then
  RED='\033[0;31m'
  GREEN='\033[0;32m'
  YELLOW='\033[1;33m'
  BOLD='\033[1m'
  NC='\033[0m'
else
  RED='' GREEN='' YELLOW='' BOLD='' NC=''
fi

run_with_timeout() {
  local secs="$1"
  shift

  if command -v timeout >/dev/null 2>&1; then
    timeout "$secs" "$@"
  elif command -v gtimeout >/dev/null 2>&1; then
    gtimeout "$secs" "$@"
  else
    "$@" &
    local pid=$!

    (
      sleep "$secs"
      kill "$pid" 2>/dev/null
    ) &
    local watcher=$!

    wait "$pid" 2>/dev/null
    local rc=$?

    kill "$watcher" 2>/dev/null
    wait "$watcher" 2>/dev/null

    return $rc
  fi
}

portable_mktemp() {
  local prefix="${1:-validate}"
  mktemp "${TMPDIR:-/tmp}/${prefix}-XXXXXX" 2>/dev/null || mktemp
}

CLEANUP_FILES=()

cleanup() {
  rm -f "${CLEANUP_FILES[@]}"
}

trap cleanup EXIT

PING_URL="${1:-}"
REPO_DIR="${2:-.}"

if [ -z "$PING_URL" ]; then
  echo "Usage: $0 <ping_url> [repo_dir]"
  exit 1
fi

if ! REPO_DIR="$(cd "$REPO_DIR" 2>/dev/null && pwd)"; then
  echo "Error: repo directory not found"
  exit 1
fi

PING_URL="${PING_URL%/}"

log() {
  printf "[%s] %b\n" "$(date -u +%H:%M:%S)" "$*"
}

pass() {
  log "${GREEN}PASSED${NC} -- $1"
}

fail() {
  log "${RED}FAILED${NC} -- $1"
}

stop_at() {
  echo
  echo "Validation stopped at $1"
  exit 1
}

echo
echo "========================================"
echo "  OpenEnv Submission Validator"
echo "========================================"

log "Repo:     $REPO_DIR"
log "Ping URL: $PING_URL"

echo
log "${BOLD}Step 1/3: Pinging HF Space${NC}"

CURL_OUTPUT=$(portable_mktemp "validate-curl")
CLEANUP_FILES+=("$CURL_OUTPUT")

HTTP_CODE=$(curl -s -o "$CURL_OUTPUT" -w "%{http_code}" \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{}' \
  "$PING_URL/reset" \
  --max-time 30 || printf "000")

if [ "$HTTP_CODE" = "200" ]; then
  pass "HF Space is live and responds to /reset"
else
  fail "HF Space /reset failed with HTTP $HTTP_CODE"
  stop_at "Step 1"
fi

echo
log "${BOLD}Step 2/3: Running docker build${NC}"

if ! command -v docker >/dev/null 2>&1; then
  fail "docker command not found"
  stop_at "Step 2"
fi

if [ -f "$REPO_DIR/Dockerfile" ]; then
  DOCKER_CONTEXT="$REPO_DIR"
elif [ -f "$REPO_DIR/server/Dockerfile" ]; then
  DOCKER_CONTEXT="$REPO_DIR/server"
else
  fail "No Dockerfile found"
  stop_at "Step 2"
fi

BUILD_OK=false
BUILD_OUTPUT=$(run_with_timeout "$DOCKER_BUILD_TIMEOUT" docker build "$DOCKER_CONTEXT" 2>&1) && BUILD_OK=true

if [ "$BUILD_OK" = true ]; then
  pass "Docker build succeeded"
else
  fail "Docker build failed"
  echo "$BUILD_OUTPUT"
  stop_at "Step 2"
fi

echo
log "${BOLD}Step 3/3: Running openenv validate${NC}"

if ! command -v openenv >/dev/null 2>&1; then
  fail "openenv command not found"
  stop_at "Step 3"
fi

VALIDATE_OK=false
VALIDATE_OUTPUT=$(cd "$REPO_DIR" && openenv validate 2>&1) && VALIDATE_OK=true

if [ "$VALIDATE_OK" = true ]; then
  pass "openenv validate passed"
else
  fail "openenv validate failed"
  echo "$VALIDATE_OUTPUT"
  stop_at "Step 3"
fi

echo
echo "========================================"
echo "  All 3/3 checks passed!"
echo "  Your submission is ready to submit."
echo "========================================"

exit 0