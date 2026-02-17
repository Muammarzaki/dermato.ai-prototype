#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TEST_DIR="$SCRIPT_DIR/src/tests"

IFACE=${IFACE:-enp4}
EXP_NAME=${EXP_NAME:-percobaan-1}
BUCKET=${BUCKET:-"gs://benchmark-2026"}
RESULTS_DIR="$SCRIPT_DIR/results"
LOG_DIR="$SCRIPT_DIR/logs"

NETWORKS=(normal poor worst 3g 4g)
SCENARIOS=(smoke load stress spike soak)
PROTOS=(grpc rest)

REAL_USER=${SUDO_USER:-$USER}
REAL_GROUP=$(id -gn "$REAL_USER")

mkdir -p "$RESULTS_DIR" "$LOG_DIR"
LOGFILE="$LOG_DIR/${EXP_NAME}_$(date +%Y%m%d_%H%M%S).log"
touch "$LOGFILE"

chown -R "$REAL_USER:$REAL_GROUP" "$RESULTS_DIR" "$LOG_DIR"

# Helper Logging
log()  { echo "[$(date '+%F %T')] $1" | tee -a "$LOGFILE"; }
info() { echo -e "\033[0;34mℹ $1\033[0m" | tee -a "$LOGFILE"; }
warn() { echo -e "\033[1;33m⚠ $1\033[0m" | tee -a "$LOGFILE"; }
error(){ echo -e "\033[0;31m✗ $1\033[0m" >&2 | tee -a "$LOGFILE"; }

net_reset() {
  tc qdisc del dev "$IFACE" root 2>/dev/null || true
  log "Network traffic control reset on $IFACE."
}
trap 'net_reset' INT TERM EXIT ERR

doctor() {
  echo "=========================================="
  echo " BENCHMARK DOCTOR CHECK (PRE-FLIGHT)"
  echo "=========================================="
  local fail=0

  if [ "$EUID" -ne 0 ]; then
    error "Must run as root (use: sudo ./benchmark.sh)"
    fail=1
  else
    echo "✓ Running with root privileges (sudo mode detected: user $REAL_USER)"
  fi

  if ip link show "$IFACE" >/dev/null 2>&1; then
    echo "✓ Network interface found: $IFACE"
  else
    error "Network interface not found: $IFACE. Override with IFACE=your_eth ./benchmark.sh"
    fail=1
  fi

  for cmd in tc gsutil; do
    if command -v "$cmd" >/dev/null; then
      echo "✓ Command '$cmd' is installed"
    else
      warn "Command '$cmd' is missing. Some features may not work."
      [ "$cmd" == "tc" ] && fail=1
    fi
  done

  if sudo -u "$REAL_USER" -H bash -c 'command -v k6' >/dev/null 2>&1; then
    echo "✓ Command 'k6' is installed for user $REAL_USER"
  else
    error "Command 'k6' is missing for user $REAL_USER"
    fail=1
  fi

  echo "------------------------------------------"
  if [ "$fail" -eq 0 ]; then
    info "DOCTOR RESULT: OK — READY"
    return 0
  else
    error "DOCTOR RESULT: FAILED — Please fix issues above."
    exit 1
  fi
}

net_apply() {
  local mode=$1
  net_reset

  case "$mode" in
    normal) ;;
    poor)  tc qdisc add dev "$IFACE" root netem delay 100ms loss 1% || warn "Failed to set poor network" ;;
    worst) tc qdisc add dev "$IFACE" root netem delay 300ms loss 3% || warn "Failed to set worst network" ;;
    3g)    tc qdisc add dev "$IFACE" root netem delay 200ms loss 2% rate 384kbit || warn "Failed to set 3G network" ;;
    4g)    tc qdisc add dev "$IFACE" root netem delay 80ms loss 0.5% rate 10mbit || warn "Failed to set 4G network" ;;
    *)     error "Unknown network mode: $mode"; exit 1 ;;
  esac
}

upload_file() {
  local file=$1
  local net=$2
  local scenario=$3

  if ! sudo -u "$REAL_USER" -H bash -c 'command -v gsutil' >/dev/null 2>&1; then
    warn "Upload skipped: gsutil missing"
    return 0
  fi

  info "Temporarily unthrottling network for fast upload..."
  net_reset
  sleep 2

  info "Uploading $(basename "$file") to GCP..."

  local max_retries=3
  local attempt=1
  local success=0

  # Sistem Retry: Mengulang upload jika terputus/gagal
  while [ $attempt -le $max_retries ]; do
    set +e
    sudo -u "$REAL_USER" -H gsutil cp "$file" "$BUCKET/$EXP_NAME/$net/$scenario/" >>"$LOGFILE" 2>&1
    local upload_status=$?
    set -e

    if [ $upload_status -eq 0 ]; then
      log "Upload successful."
      success=1
      break
    else
      warn "Upload failed (Attempt $attempt of $max_retries)."
      sleep 2
      ((attempt++))
    fi
  done

  if [ $success -eq 0 ]; then
    error "Upload permanently failed for $(basename "$file") after $max_retries attempts. File kept locally."
  fi

  info "Restoring network profile: [$net]"
  net_apply "$net"
  sleep 2
}

run_test() {
  local proto=$1
  local scenario=$2
  local net=$3
  local ts=$(date +%Y%m%d_%H%M%S)
  local out="$RESULTS_DIR/${proto}_${net}_${scenario}_${ts}.csv"
  local test_file="$TEST_DIR/${proto}.test.js"

  if [ ! -f "$test_file" ]; then
    error "Test script not found: $test_file"
    return 1
  fi

  info "Starting k6: [$proto] | Scenario: [$scenario] | Network: [$net]"

  # EXECUTION CORE: Redirection log dihapus agar metrik k6 tampil mulus di Terminal
  set +e
  sudo -u "$REAL_USER" -H k6 run \
    -e SCENARIO="$scenario" \
    -e REST_ADDR="${REST_ADDR:-}" \
    -e GRPC_ADDR="${GRPC_ADDR:-}" \
    "$test_file" --out csv="$out"
  local exit_code=$?
  set -e

  # Validasi file CSV, pastikan file benar-benar ada dan berisi data sebelum di-upload
  if [ $exit_code -eq 0 ] && [ -s "$out" ]; then
    upload_file "$out" "$net" "$scenario"
    return 0
  else
    error "Test failed or interrupted for $proto $scenario on $net network (Exit code: $exit_code)"
    sudo -u "$REAL_USER" -H rm -f "$out" # Hapus file CSV yang gagal/kosong
    return 1
  fi
}

full_suite() {
  info "=========================================="
  info " STARTING FULL BENCHMARK SUITE"
  info " Experiment : $EXP_NAME"
  info " Interface  : $IFACE"
  info " Bucket     : $BUCKET"
  info " Target Log : $LOGFILE"
  info "=========================================="

  for net in "${NETWORKS[@]}"; do
    info ">> Applying Network Profile: $net"
    net_apply "$net"
    sleep 2

    for scenario in "${SCENARIOS[@]}"; do
      for proto in "${PROTOS[@]}"; do
        # '|| true' memastikan script tetap lanjut ke skenario berikutnya meskipun tes ini gagal
        run_test "$proto" "$scenario" "$net" || true
      done
    done

    net_reset
    sleep 3
  done

  info "=========================================="
  info " FULL BENCHMARK COMPLETED SUCCESSFULLY"
  info "=========================================="
}

ACTION=${1:-full}

case "$ACTION" in
  doctor) doctor ;;
  net)    net_apply "${2:-normal}" ;;
  test)   run_test grpc smoke normal ;;
  full)   doctor && full_suite ;;
  *)
    echo "Usage:"
    echo "  sudo IFACE=ens4 EXP_NAME=percobaan-1 ./benchmark.sh doctor"
    echo "  sudo IFACE=ens4 ./benchmark.sh net <normal|poor|worst|3g|4g>"
    echo "  sudo IFACE=ens4 ./benchmark.sh test"
    echo "  sudo GRPC_ADDR=10.128.0.2:8008 REST_ADDR=http://10.128.0.2:8088 IFACE=ens4 ./benchmark.sh full"
    exit 1
  ;;
esac