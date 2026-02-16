#!/bin/bash
set -e

RESULTS_DIR="./results"
LOG_DIR="./logs"
SERVER_IP=${SERVER_IP:-127.0.0.1}
GRPC_ADDR=${GRPC_ADDR:-127.0.0.1:8008}
REST_ADDR=${REST_ADDR:-http://127.0.0.1:8088}
EXP_NAME=${EXP_NAME:-percobaan-1}

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

info()  { echo -e "\033[0;34mℹ $1\033[0m"; }
ok()    { echo -e "\033[0;32m✓ $1\033[0m"; }
warn()  { echo -e "\033[1;33m⚠ $1\033[0m"; }
error() { echo -e "\033[0;31m✗ $1\033[0m"; }

if [ "$EUID" -ne 0 ]; then
    error "This script requires root privileges"
    warn "Run as: sudo -E EXP_NAME=percobaan-1 SERVER_IP=127.0.0.1 $0"
    exit 1
fi

LOGFILE="$LOG_DIR/${EXP_NAME}-$(date +%Y%m%d_%H%M%S).log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOGFILE"
}

trap 'sudo ./net-sim.sh normal >/dev/null 2>&1 || true' INT TERM EXIT

run_experiment() {
    local network=$1
    local scenario=$2

    log "=========================================="
    log "Experiment: $EXP_NAME"
    log "Network: $network"
    log "Scenario: $scenario"
    log "gRPC: $GRPC_ADDR"
    log "REST: $REST_ADDR"
    log "=========================================="

    if [ "$network" != "normal" ]; then
        log "Applying network simulation: $network"
        sudo SERVER_IP=$SERVER_IP ./net-sim.sh "$network" >>"$LOGFILE" 2>&1 \
            || warn "net-sim failed"
        sleep 5
    fi

    log "Running gRPC via run-tests.sh"
    EXP="${EXP_NAME}/${network}" ./run-tests.sh grpc "$scenario" >>"$LOGFILE" 2>&1 \
        || warn "gRPC $scenario failed"

    sleep 60

    log "Running REST via run-tests.sh"
    EXP="${EXP_NAME}/${network}" ./run-tests.sh rest "$scenario" >>"$LOGFILE" 2>&1 \
        || warn "REST $scenario failed"

    if [ "$network" != "normal" ]; then
        log "Resetting network..."
        sudo ./net-sim.sh normal >>"$LOGFILE" 2>&1 \
            || warn "net reset failed"
        sleep 5
    fi

    log "Complete: $network / $scenario"
    log ""
}

info "Starting FULL benchmark suite"
info "Experiment name: $EXP_NAME"
info "Server IP: $SERVER_IP"
info "gRPC Address: $GRPC_ADDR"
info "REST Address: $REST_ADDR"
info "Log file: $LOGFILE"
info "This will run in background. Monitor: tail -f $LOGFILE"
echo ""

{
    set +e

    log "=========================================="
    log "BENCHMARK SUITE: $EXP_NAME"
    log "Server: $SERVER_IP"
    log "Started: $(date)"
    log "=========================================="

    for net in normal poor worst 3g 4g; do
        for sc in smoke load stress spike soak; do
            run_experiment "$net" "$sc"
        done
    done

    log "=========================================="
    log "BENCHMARK SUITE COMPLETED: $EXP_NAME"
    log "Finished: $(date)"
    log "=========================================="

} &

BACKGROUND_PID=$!

ok "Benchmark suite started in background (PID: $BACKGROUND_PID)"
ok "Experiment: $EXP_NAME"
ok "Monitor progress: tail -f $LOGFILE"
ok "Stop anytime: sudo kill $BACKGROUND_PID"
echo ""
info "Estimated completion time: ~10-12 hours"
info "Total experiments: 25 (5 networks × 5 scenarios)"
echo ""
