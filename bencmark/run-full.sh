#!/bin/bash
set -e

RESULTS_DIR="./results"
LOG_DIR="./logs"
SERVER_IP=${SERVER_IP:-127.0.0.1}
GRPC_ADDR=${GRPC_ADDR:-127.0.0.1:8008}
REST_ADDR=${REST_ADDR:-http://127.0.0.1:8088}
BUCKET="gs://benchmark-2026"
EXP_NAME=${EXP_NAME:-percobaan-1}

mkdir -p "$RESULTS_DIR"
mkdir -p "$LOG_DIR"

info() { echo -e "\033[0;34mℹ $1\033[0m"; }
ok()   { echo -e "\033[0;32m✓ $1\033[0m"; }
warn() { echo -e "\033[1;33m⚠ $1\033[0m"; }
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

run_single_test() {
    local protocol=$1
    local network=$2
    local scenario=$3
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local filename="${protocol}_${scenario}_${timestamp}.csv"
    local local_file="$RESULTS_DIR/$filename"
    local remote_path="$BUCKET/${EXP_NAME}/${network}/${scenario}/"

    log "Running $protocol test..."
    k6 run -e SCENARIO="$scenario" \
           -e GRPC_ADDR="$GRPC_ADDR" \
           -e REST_ADDR="$REST_ADDR" \
           "src/tests/${protocol}.test.js" \
        --out csv="$local_file" >> "$LOGFILE" 2>&1

    log "Uploading to $remote_path"
    if gsutil cp "$local_file" "$remote_path" >> "$LOGFILE" 2>&1; then
        log "Upload success: $filename"
    else
        log "Upload failed: $filename (kept locally)"
    fi

    sleep 60
}

run_experiment() {
    local network_condition=$1
    local scenario=$2

    log "=========================================="
    log "Experiment: $EXP_NAME"
    log "Network: $network_condition"
    log "Scenario: $scenario"
    log "gRPC: $GRPC_ADDR"
    log "REST: $REST_ADDR"
    log "=========================================="

    if [ "$network_condition" != "normal" ]; then
        log "Applying network simulation: $network_condition"
        SERVER_IP=$SERVER_IP ./net-sim.sh "$network_condition" >> "$LOGFILE" 2>&1
        sleep 5
    fi

    run_single_test "grpc" "$network_condition" "$scenario"
    run_single_test "rest" "$network_condition" "$scenario"

    if [ "$network_condition" != "normal" ]; then
        log "Resetting network..."
        ./net-sim.sh normal >> "$LOGFILE" 2>&1
        sleep 5
    fi

    log "Complete: $network_condition / $scenario"
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
    log "=========================================="
    log "BENCHMARK SUITE: $EXP_NAME"
    log "Server: $SERVER_IP"
    log "gRPC: $GRPC_ADDR"
    log "REST: $REST_ADDR"
    log "Started: $(date)"
    log "=========================================="

    # Normal network
    run_experiment "normal" "smoke"
    run_experiment "normal" "load"
    run_experiment "normal" "stress"
    run_experiment "normal" "spike"
    run_experiment "normal" "soak"

    # Poor network
    run_experiment "poor" "smoke"
    run_experiment "poor" "load"
    run_experiment "poor" "stress"
    run_experiment "poor" "spike"
    run_experiment "poor" "soak"

    # Worst network
    run_experiment "worst" "smoke"
    run_experiment "worst" "load"
    run_experiment "worst" "stress"
    run_experiment "worst" "spike"
    run_experiment "worst" "soak"

    # 3G network
    run_experiment "3g" "smoke"
    run_experiment "3g" "load"
    run_experiment "3g" "stress"
    run_experiment "3g" "spike"
    run_experiment "3g" "soak"

    # 4G network
    run_experiment "4g" "smoke"
    run_experiment "4g" "load"
    run_experiment "4g" "stress"
    run_experiment "4g" "spike"
    run_experiment "4g" "soak"

    log "=========================================="
    log "BENCHMARK SUITE COMPLETED: $EXP_NAME"
    log "Finished: $(date)"
    log "=========================================="
    log "Results uploaded to: $BUCKET/$EXP_NAME/"
    log "Local results: $RESULTS_DIR"
    log "Log file: $LOGFILE"

} &

BACKGROUND_PID=$!

ok "Benchmark suite started in background (PID: $BACKGROUND_PID)"
ok "Experiment: $EXP_NAME"
ok "gRPC: $GRPC_ADDR"
ok "REST: $REST_ADDR"
ok "Monitor progress: tail -f $LOGFILE"
ok "Stop anytime: sudo kill $BACKGROUND_PID"
echo ""
info "Estimated completion time: ~10-12 hours"
info "Total experiments: 25 (5 networks × 5 scenarios)"
echo ""
info "GCS structure after completion:"
echo "  gs://benchmark-2026/${EXP_NAME}/normal/smoke/"
echo "  gs://benchmark-2026/${EXP_NAME}/normal/load/"
echo "  gs://benchmark-2026/${EXP_NAME}/poor/smoke/"
echo "  gs://benchmark-2026/${EXP_NAME}/worst/stress/"
echo "  gs://benchmark-2026/${EXP_NAME}/3g/load/"
echo "  gs://benchmark-2026/${EXP_NAME}/4g/soak/"
echo ""