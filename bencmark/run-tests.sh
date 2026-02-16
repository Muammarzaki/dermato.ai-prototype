#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RESULTS_DIR="./results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXP=${EXP:-default}
BUCKET="gs://benchmark-2026"

mkdir -p "$RESULTS_DIR"

info() { echo -e "\033[0;34mℹ $1\033[0m"; }
ok()   { echo -e "\033[0;32m✓ $1\033[0m"; }
warn() { echo -e "\033[1;33m⚠ $1\033[0m"; }

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

upload() {
    local file=$1
    local scenario=$2

    if [ "$UPLOAD_ENABLED" = false ]; then
        warn "Upload skipped — file saved locally: $file"
        return 0
    fi

    local remote_path="$BUCKET/${EXP}/${scenario}/"
    info "Uploading $(basename "$file") to $remote_path"
    if gsutil cp "$file" "$remote_path"; then
        ok "Uploaded: $(basename "$file")"
    else
        warn "Upload failed — file kept locally"
    fi
}

run_warmup() {
    info "Warmup test (no data recorded)"
    k6 run src/tests/warmup.test.js
    ok "Warmup finished"
    sleep 30
}

run_grpc() {
    local scenario=${1:-load}
    local file="$RESULTS_DIR/${EXP}_grpc_${scenario}_${TIMESTAMP}.csv"

    info "Running gRPC test: $scenario"

    if k6 run -e SCENARIO="$scenario" src/tests/grpc.test.js --out csv="$file"; then
        upload "$file" "$scenario"
        ok "gRPC $scenario finished"
    else
        warn "gRPC $scenario failed — upload skipped"
        return 0
    fi
}

run_rest() {
    local scenario=${1:-load}
    local file="$RESULTS_DIR/${EXP}_rest_${scenario}_${TIMESTAMP}.csv"

    info "Running REST test: $scenario"
    k6 run -e SCENARIO="$scenario" src/tests/rest.test.js \
        --out csv="$file"
    upload "$file" "$scenario"
    ok "REST $scenario finished"
}

run_rest() {
    local scenario=${1:-load}
    local file="$RESULTS_DIR/${EXP}_rest_${scenario}_${TIMESTAMP}.csv"

    info "Running REST test: $scenario"

    if k6 run -e SCENARIO="$scenario" src/tests/rest.test.js --out csv="$file"; then
        upload "$file" "$scenario"
        ok "REST $scenario finished"
    else
        warn "REST $scenario failed — upload skipped"
        return 0
    fi
}

run_full_suite() {
    run_warmup

    run_scenario smoke
    run_scenario load
    run_scenario stress
    run_scenario spike
    run_scenario soak
}

case "${1:-help}" in
    grpc)
        run_grpc "${2:-load}"
        ;;
    rest)
        run_rest "${2:-load}"
        ;;
    scenario)
        run_scenario "${2:-load}"
        ;;
    full-suite)
        run_full_suite
        ;;
    *)
        echo "Usage:"
        echo "  EXP=percobaan-2 ./run-tests.sh grpc [smoke|load|stress|spike|soak]"
        echo "  EXP=percobaan-2 ./run-tests.sh rest [smoke|load|stress|spike|soak]"
        echo "  EXP=percobaan-2 ./run-tests.sh scenario [smoke|load|stress|spike|soak]"
        echo "  EXP=percobaan-2 ./run-tests.sh full-suite"
        echo ""
        echo "Examples:"
        echo "  EXP=baseline ./run-tests.sh scenario load"
        echo "  EXP=p1 ./run-tests.sh full-suite"
        echo "  EXP=poor-net ./run-tests.sh scenario stress"
        echo ""
        echo "GCS Structure:"
        echo "  gs://benchmark-2026/percobaan-2/smoke/..."
        echo "  gs://benchmark-2026/percobaan-2/load/..."
        echo "  gs://benchmark-2026/p1/load/..."
        exit 1
        ;;
esac

ok "Experiment finished: $EXP"
ok "Local results directory: $RESULTS_DIR"

if [ "$UPLOAD_ENABLED" = true ]; then
    ok "Cloud bucket: $BUCKET/$EXP"
else
    warn "Cloud upload disabled — local only"
fi