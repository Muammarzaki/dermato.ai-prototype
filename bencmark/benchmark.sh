#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TEST_DIR="$SCRIPT_DIR/src/tests"

### ========= GLOBAL CONFIG =========
IFACE=${IFACE:-enp4}
EXP_NAME=${EXP_NAME:-percobaan-1}
BUCKET="gs://benchmark-2026"
RESULTS_DIR="./results"
LOG_DIR="./logs"

NETWORKS=(normal poor worst 3g 4g)
SCENARIOS=(smoke load stress spike soak)
PROTOS=(grpc rest)

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

LOGFILE="$LOG_DIR/${EXP_NAME}_$(date +%Y%m%d_%H%M%S).log"

log()  { echo "[$(date '+%F %T')] $1" | tee -a "$LOGFILE"; }
info() { echo -e "\033[0;34mℹ $1\033[0m"; }
warn() { echo -e "\033[1;33m⚠ $1\033[0m"; }
error(){ echo -e "\033[0;31m✗ $1\033[0m"; }

trap 'tc qdisc del dev "$IFACE" root 2>/dev/null || true' INT TERM EXIT

### ========= DOCTOR =========
doctor() {
  echo "=========================================="
  echo "BENCHMARK DOCTOR CHECK"
  echo "=========================================="

  FAIL=0

  if [ "$EUID" -ne 0 ]; then
    echo "✗ Must run as root (sudo)"
    FAIL=1
  else
    echo "✓ Running as root"
  fi

  if ip link show "$IFACE" >/dev/null 2>&1; then
    echo "✓ Interface found: $IFACE"
  else
    echo "✗ Interface not found: $IFACE"
    FAIL=1
  fi

  for cmd in tc k6; do
    if command -v "$cmd" >/dev/null; then
      echo "✓ $cmd installed"
    else
      echo "✗ $cmd missing"
      FAIL=1
    fi
  done

  if command -v gsutil >/dev/null; then
    echo "✓ gsutil installed"
    if gsutil ls "$BUCKET" >/dev/null 2>&1; then
      echo "✓ Bucket accessible: $BUCKET"
    else
      echo "⚠ Bucket not accessible (upload skipped)"
    fi
  else
    echo "⚠ gsutil not installed (upload skipped)"
  fi

  echo "------------------------------------------"
  if [ "$FAIL" -eq 0 ]; then
    echo "DOCTOR RESULT: OK — READY"
    return 0
  else
    echo "DOCTOR RESULT: FAILED"
    return 1
  fi
}

### ========= NETWORK =========
net_reset() {
  tc qdisc del dev "$IFACE" root 2>/dev/null || true
}

net_apply() {
  local mode=$1
  net_reset

  case "$mode" in
    normal) ;;
    poor)  tc qdisc add dev "$IFACE" root netem delay 100ms loss 1% ;;
    worst) tc qdisc add dev "$IFACE" root netem delay 300ms loss 3% ;;
    3g)    tc qdisc add dev "$IFACE" root netem delay 200ms loss 2% rate 384kbit ;;
    4g)    tc qdisc add dev "$IFACE" root netem delay 80ms loss 0.5% rate 10mbit ;;
    *) error "Unknown network mode: $mode"; exit 1 ;;
  esac
}

### ========= TEST =========
run_test() {
  local proto=$1
  local scenario=$2
  local net=$3
  local ts=$(date +%Y%m%d_%H%M%S)
  local out="$RESULTS_DIR/${proto}_${net}_${scenario}_${ts}.csv"
  local test_file="$TEST_DIR/${proto}.test.js"

 if [ ! -f "$test_file" ]; then
    error "Test file not found: $test_file"
    return 1
  fi

  log "Running $proto | $scenario | $net"

  if k6 run -e SCENARIO="$scenario" "$test_file" \
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

  if ! command -v gsutil >/dev/null; then
    warn "Upload skipped (gsutil missing)"
    return 0
  fi

  log "Uploading $(basename "$file")"
  gsutil cp "$file" \
    "$BUCKET/$EXP_NAME/$net/$scenario/" >>"$LOGFILE" 2>&1 \
    || warn "Upload failed"
}

### ========= FULL SUITE =========
full_suite() {
  info "Starting FULL BENCHMARK"
  info "Experiment: $EXP_NAME"
  info "Interface: $IFACE"
  info "Log: $LOGFILE"
  echo ""

  for net in "${NETWORKS[@]}"; do
    info "Network: $net"
    net_apply "$net"
    sleep 3

    for scenario in "${SCENARIOS[@]}"; do
      info "Scenario: $scenario"

      for proto in "${PROTOS[@]}"; do
        file=$(run_test "$proto" "$scenario" "$net") || continue
        upload_file "$file" "$net" "$scenario"
      done
    done

    net_reset
    sleep 5
  done

  info "FULL BENCHMARK COMPLETED"
}

### ========= MAIN =========
case "$1" in
  doctor)
    doctor
    ;;
  net)
    net_apply "$2"
    ;;
  test)
    run_test grpc smoke normal
    ;;
  full)
    full_suite
    ;;
  run|"")
    full_suite
    ;;
  *)
    echo "Usage:"
    echo "  sudo IFACE=enp4 EXP_NAME=percobaan-2 ./benchmark.sh doctor"
    echo "  sudo IFACE=enp4 ./benchmark.sh net <normal|poor|worst|3g|4g>"
    echo "  sudo IFACE=enp4 ./benchmark.sh test"
    echo "  sudo IFACE=enp4 EXP_NAME=percobaan-2 ./benchmark.sh full"
    exit 1
esac
