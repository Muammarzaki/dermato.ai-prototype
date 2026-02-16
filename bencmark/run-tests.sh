#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RESULTS_DIR="./results"
EXP=${EXP:-default}
TS=$(date +%Y%m%d_%H%M%S)

mkdir -p "$RESULTS_DIR"

info(){ echo -e "\033[0;34mℹ $1\033[0m"; }
ok(){ echo -e "\033[0;32m✓ $1\033[0m"; }
warn(){ echo -e "\033[1;33m⚠ $1\033[0m"; }

run_one() {
    local proto=$1
    local scenario=$2
    local out="$RESULTS_DIR/${EXP}_${proto}_${scenario}_${TS}.csv"

    info "Running $proto $scenario"
    if k6 run -e SCENARIO="$scenario" "src/tests/${proto}.test.js" --out csv="$out"; then
        ok "$proto $scenario OK"
    else
        warn "$proto $scenario FAILED"
        rm -f "$out"
    fi
}

run_scenario() {
    local s=$1
    run_one grpc "$s"
    sleep 60
    run_one rest "$s"
    sleep 60
}

case "$1" in
    scenario) run_scenario "$2" ;;
    *) echo "Usage: ./run-tests.sh scenario [smoke|load|stress|spike|soak]" ;;
esac
