#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RESULTS_DIR="./results"
LOG_DIR="./logs"
BUCKET="gs://benchmark-2026"
EXP_NAME=${EXP_NAME:-percobaan-1}

mkdir -p "$LOG_DIR"

logf="$LOG_DIR/${EXP_NAME}_$(date +%Y%m%d_%H%M%S).log"
log(){ echo "[$(date '+%F %T')] $1" | tee -a "$logf"; }

trap 'sudo ./net-sim.sh normal >/dev/null 2>&1 || true' EXIT

run_block() {
    local net=$1
    local scen=$2

    log "Network=$net Scenario=$scen"

    [ "$net" != "normal" ] && sudo ./net-sim.sh "$net"
    EXP="$EXP_NAME" ./run-tests.sh scenario "$scen"
    [ "$net" != "normal" ] && sudo ./net-sim.sh normal
}

log "BENCHMARK START"

for net in normal poor worst 3g 4g; do
  for s in smoke load stress spike soak; do
    run_block "$net" "$s"
  done
done

log "UPLOAD START"
gsutil -m cp -r "$RESULTS_DIR" "$BUCKET/$EXP_NAME/"

log "CLEANUP"
rm -rf "$RESULTS_DIR"

log "DONE"
