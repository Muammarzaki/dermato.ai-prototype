#!/bin/bash
set -e

### ========= GLOBAL CONFIG =========
IFACE=${IFACE:-enp4}
EXP_NAME=${EXP_NAME:-percobaan-1}
BUCKET="gs://benchmark-2026"
RESULTS_DIR="./results"
LOG_DIR="./logs"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

LOGFILE="$LOG_DIR/${EXP_NAME}_$(date +%Y%m%d_%H%M%S).log"
log(){ echo "[$(date '+%F %T')] $1" | tee -a "$LOGFILE"; }

info(){ echo -e "\033[0;34mℹ $1\033[0m"; }
warn(){ echo -e "\033[1;33m⚠ $1\033[0m"; }
error(){ echo -e "\033[0;31m✗ $1\033[0m"; }

### ========= NETWORK =========
net_reset() {
  tc qdisc del dev "$IFACE" root 2>/dev/null || true
}

doctor() {
    echo "=========================================="
    echo "BENCHMARK DOCTOR CHECK"
    echo "=========================================="

    FAIL=0

    if [ "$EUID" -ne 0 ]; then
        echo "✗ Must run as root (use sudo)"
        FAIL=1
    else
        echo "✓ Running as root"
    fi

    if [ -z "$IFACE" ]; then
        echo "✗ IFACE not set (export IFACE=enp4s0)"
        FAIL=1
    elif ip link show "$IFACE" >/dev/null 2>&1; then
        echo "✓ Network interface found: $IFACE"
    else
        echo "✗ Network interface not found: $IFACE"
        FAIL=1
    fi

    for cmd in tc k6; do
        if command -v "$cmd" >/dev/null 2>&1; then
            echo "✓ $cmd installed"
        else
            echo "✗ $cmd not installed"
            FAIL=1
        fi
    done

    if command -v gsutil >/dev/null 2>&1; then
        echo "✓ gsutil installed"
        if gsutil ls "$BUCKET" >/dev/null 2>&1; then
            echo "✓ GCS bucket accessible: $BUCKET"
        else
            echo "⚠ gsutil installed but bucket not accessible"
            echo "  → upload will be skipped"
        fi
    else
        echo "⚠ gsutil not installed (upload disabled)"
    fi

    if [ -n "$IFACE" ] && command -v tc >/dev/null 2>&1; then
        echo "✓ Testing netem on $IFACE"
        if tc qdisc add dev "$IFACE" root netem delay 10ms 2>/dev/null; then
            tc qdisc del dev "$IFACE" root 2>/dev/null
            echo "✓ netem works"
        else
            echo "✗ netem failed (interface busy or permission issue)"
            FAIL=1
        fi
    fi

    echo "------------------------------------------"
    if [ "$FAIL" -eq 0 ]; then
        echo "DOCTOR RESULT: OK — READY TO RUN"
        return 0
    else
        echo "DOCTOR RESULT: FAILED — FIX ISSUES ABOVE"
        return 1
    fi
}

net_apply() {
  local mode=$1
  net_reset

  case "$mode" in
    normal) ;;
    poor)   tc qdisc add dev "$IFACE" root netem delay 100ms loss 1% ;;
    worst)  tc qdisc add dev "$IFACE" root netem delay 300ms loss 3% ;;
    *) error "Unknown net mode"; exit 1 ;;
  esac
}

### ========= BENCHMARK =========
run_test() {
  local proto=$1
  local scenario=$2
  local net=$3
  local ts=$(date +%Y%m%d_%H%M%S)
  local out="$RESULTS_DIR/${proto}_${net}_${scenario}_${ts}.csv"

  log "Running $proto | $scenario | $net"

  if k6 run -e SCENARIO="$scenario" src/tests/$proto.test.js \
      --out csv="$out" >>"$LOGFILE" 2>&1; then
    echo "$out"
  else
    warn "$proto $scenario failed"
    rm -f "$out"
    return 1
  fi
}

### ========= UPLOAD =========
upload_file() {
  local file=$1
  local net=$2
  local scenario=$3

  log "Uploading $(basename "$file")"
  gsutil cp "$file" \
    "$BUCKET/$EXP_NAME/$net/$scenario/" >>"$LOGFILE" 2>&1 \
    || warn "Upload failed"
}

upload_all() {
  for f in "$RESULTS_DIR"/*.csv; do
    [ -f "$f" ] || continue
    gsutil cp "$f" "$BUCKET/$EXP_NAME/raw/"
  done
}

### ========= ORCHESTRATION =========
orchestrate() {
  for net in normal poor worst; do
    net_apply "$net"

    for scenario in smoke load stress; do
      for proto in grpc rest; do
        file=$(run_test "$proto" "$scenario" "$net") || continue
        upload_file "$file" "$net" "$scenario"
      done
    done

    net_reset
  done
}

### ========= MAIN =========
case "$1" in
 doctor)
        doctor
        exit $?
        ;;
  net)
    [ -z "$2" ] && error "net <mode>" && exit 1
    net_apply "$2"
    ;;
  test)
    run_test grpc smoke normal
    ;;
  upload)
    upload_all
    ;;
  run|"")
    orchestrate
    ;;
  *)
    echo "Usage:"
    echo "  ./benchmark.sh net <normal|poor|worst>"
    echo "  ./benchmark.sh test"
    echo "  ./benchmark.sh upload"
    echo "  ./benchmark.sh run"
    exit 1
esac
