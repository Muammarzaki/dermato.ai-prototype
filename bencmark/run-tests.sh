#!/bin/bash
set -e

RESULTS_DIR="./results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PREFIX=${PREFIX:-run}
BUCKET="gs://benchmark-2026"

mkdir -p "$RESULTS_DIR"

info() { echo -e "\033[0;34mℹ $1\033[0m"; }
ok()   { echo -e "\033[0;32m✓ $1\033[0m"; }
warn() { echo -e "\033[1;33m⚠ $1\033[0m"; }

# ==========================
# CHECK GCS ACCESS
# ==========================
UPLOAD_ENABLED=true
info "Checking access to $BUCKET ..."

if ! command -v gsutil >/dev/null 2>&1; then
    warn "gsutil not found — upload disabled"
    UPLOAD_ENABLED=false
elif ! gsutil ls "$BUCKET" >/dev/null 2>&1; then
    warn "Cannot access bucket (not logged in or no permission)"
    warn "Results will be saved locally only"
    UPLOAD_ENABLED=false
else
    ok "Bucket access OK — upload enabled"
fi

# ==========================
# Warmup (NO DATA)
# ==========================
run_warmup() {
    info "Warmup test (no data recorded)"
    k6 run src/tests/warmup.test.js
    ok "Warmup finished"
    sleep 30
}

# ==========================
# Smoke
# ==========================
run_smoke() {
    local file="$RESULTS_DIR/${PREFIX}_smoke_${TIMESTAMP}.csv"
    info "Smoke test"
    k6 run -e SCENARIO=smoke src/tests/balanced.test.js \
        --out csv="$file"
    upload "$file"
    ok "Smoke finished"
}

# ==========================
# Balanced
# ==========================
run_balanced() {
    local scenario=${1:-load}
    local file="$RESULTS_DIR/${PREFIX}_balanced_${scenario}_${TIMESTAMP}.csv"
    info "Balanced test: $scenario"
    k6 run -e SCENARIO="$scenario" src/tests/balanced.test.js \
        --out csv="$file"
    upload "$file"
    ok "Balanced $scenario finished"
}

# ==========================
# Comparison
# ==========================
run_comparison() {
    local file="$RESULTS_DIR/${PREFIX}_comparison_${TIMESTAMP}.csv"
    info "Comparison test (gRPC vs REST)"
    k6 run src/tests/comparison.test.js \
        --out csv="$file"
    upload "$file"
    ok "Comparison finished"
}

# ==========================
# Upload handler (SAFE)
# ==========================
upload() {
    local file=$1

    if [ "$UPLOAD_ENABLED" = false ]; then
        warn "Upload skipped — file saved locally: $file"
        return 0
    fi

    info "Uploading $(basename "$file") to $BUCKET"
    if gsutil cp "$file" "$BUCKET/"; then
        ok "Uploaded: $(basename "$file")"
    else
        warn "Upload failed — file kept locally"
    fi
}

# ==========================
# Full Suite
# ==========================
run_full_suite() {
    run_warmup
    run_smoke

    run_balanced load
    sleep 30

    run_balanced stress
    sleep 30

    run_comparison
}

# ==========================
# Main
# ==========================
case "${1:-help}" in
    smoke)      run_smoke ;;
    balanced)   run_balanced "${2:-load}" ;;
    comparison) run_warmup; run_comparison ;;
    full-suite) run_full_suite ;;
    *)
        echo "Usage:"
        echo "PREFIX=local ./run-tests.sh full-suite"
        echo "PREFIX=server ./run-tests.sh comparison"
        echo "PREFIX=simulate_worst_net ./run-tests.sh balanced stress"
        exit 1
        ;;
esac

ok "Experiment finished"
ok "Local results directory: $RESULTS_DIR"

if [ "$UPLOAD_ENABLED" = true ]; then
    ok "Cloud bucket: $BUCKET"
else
    warn "Cloud upload disabled — local only"
fi
