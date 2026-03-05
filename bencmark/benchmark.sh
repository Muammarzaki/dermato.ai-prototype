#!/bin/bash
# benchmark.sh — Locust benchmark runner
# Jalankan dari dalam folder bencmark/
#
#   sudo ./benchmark.sh full
#   sudo ./benchmark.sh doctor
#   sudo ./benchmark.sh net poor
#   sudo ./benchmark.sh test grpc smoke normal
#   sudo GRPC_ADDR=10.0.0.1:8008 REST_ADDR=http://10.0.0.1:8088 EXP_NAME=percobaan-1 ./benchmark.sh full

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ─── Auto-detect: real user (bukan root) ─────────────────────────────────────
REAL_USER=${SUDO_USER:-$USER}
REAL_GROUP=$(id -gn "$REAL_USER")
REAL_HOME=$(eval echo "~$REAL_USER")

# ─── Auto-detect: network interface (ambil interface pertama yang UP, bukan lo) ─
IFACE=${IFACE:-$(ip -o link show up | awk -F': ' '$2 != "lo" {print $2; exit}')}

# ─── Auto-detect: CPU count untuk locust workers ──────────────────────────────
CPU_COUNT=$(nproc --all 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2)
# Sisakan 1 core untuk OS, minimal 1 worker
LOCUST_WORKERS=$(( CPU_COUNT > 2 ? CPU_COUNT - 1 : 1 ))

# ─── Auto-detect: venv locust ────────────────────────────────────────────────
_find_locust() {
  # Cari di: venv lokal → virtualenv user → PATH biasa
  for candidate in \
    "$SCRIPT_DIR/locust-env/bin/locust" \
    "$SCRIPT_DIR/.venv/bin/locust" \
    "$REAL_HOME/.local/bin/locust" \
    "$(sudo -u "$REAL_USER" -H bash -c 'command -v locust 2>/dev/null || true')"
  do
    [ -x "$candidate" ] && { echo "$candidate"; return; }
  done
  echo ""
}
LOCUST_BIN=${LOCUST_BIN:-$(_find_locust)}

# ─── Config ──────────────────────────────────────────────────────────────────
EXP_NAME=${EXP_NAME:-percobaan-1}
BUCKET=${BUCKET:-"gs://benchmark-2026"}
GRPC_ADDR=${GRPC_ADDR:-"127.0.0.1:8008"}
REST_ADDR=${REST_ADDR:-"http://127.0.0.1:8088"}
RESULTS_DIR="$SCRIPT_DIR/results"
LOG_DIR="$SCRIPT_DIR/logs"

NETWORKS=(normal poor worst 3g 4g)
SCENARIOS=(smoke load stress spike soak)
PROTOS=(grpc rest)

# ─── Setup ───────────────────────────────────────────────────────────────────
mkdir -p "$RESULTS_DIR" "$LOG_DIR"
LOGFILE="$LOG_DIR/${EXP_NAME}_$(date +%Y%m%d_%H%M%S).log"
touch "$LOGFILE"
chown -R "$REAL_USER:$REAL_GROUP" "$RESULTS_DIR" "$LOG_DIR"

# ─── Logging ─────────────────────────────────────────────────────────────────
log()   { echo "[$(date '+%F %T')] $*" | tee -a "$LOGFILE"; }
info()  { echo -e "\033[0;34mℹ $*\033[0m"  | tee -a "$LOGFILE"; }
warn()  { echo -e "\033[1;33m⚠ $*\033[0m"  | tee -a "$LOGFILE"; }
ok()    { echo -e "\033[0;32m✓ $*\033[0m"  | tee -a "$LOGFILE"; }
error() { echo -e "\033[0;31m✗ $*\033[0m" | tee -a "$LOGFILE"; }

# ─── Network — fail-safe: error di tc tidak hentikan script ──────────────────
net_reset() {
  # || true supaya tidak trigger set -e kalau tidak ada qdisc
  tc qdisc del dev "$IFACE" root 2>/dev/null || true
  log "Network reset on $IFACE"
}

net_apply() {
  local mode=$1
  net_reset

  local tc_cmd=""
  case "$mode" in
    normal) ok "Network: normal (no throttle)"; return ;;
    poor)   tc_cmd="tc qdisc add dev $IFACE root netem delay 100ms loss 1%" ;;
    worst)  tc_cmd="tc qdisc add dev $IFACE root netem delay 300ms loss 3%" ;;
    3g)     tc_cmd="tc qdisc add dev $IFACE root netem delay 200ms loss 2% rate 384kbit" ;;
    4g)     tc_cmd="tc qdisc add dev $IFACE root netem delay 80ms loss 0.5% rate 10mbit" ;;
    *)      error "Unknown network mode: $mode"; return 1 ;;
  esac

  # Fail-safe: kalau tc crash (misal module netem belum load), log warning
  # dan lanjut — data tetap dikumpulkan meski tanpa throttle
  if $tc_cmd 2>>"$LOGFILE"; then
    ok "Network: $mode applied on $IFACE"
  else
    warn "tc gagal untuk mode '$mode' — test jalan tanpa throttle (data tetap valid)"
  fi
}

# Pastikan network selalu di-reset walau script crash / Ctrl+C
trap 'net_reset; log "Interrupted — network cleaned up"' INT TERM EXIT ERR

# ─── Doctor ──────────────────────────────────────────────────────────────────
doctor() {
  echo "=========================================="
  echo " BENCHMARK DOCTOR CHECK"
  echo " Interface : $IFACE"
  echo " CPU cores : $CPU_COUNT (workers: $LOCUST_WORKERS)"
  echo " Locust    : ${LOCUST_BIN:-NOT FOUND}"
  echo "=========================================="
  local fail=0

  [ "$EUID" -eq 0 ] \
    && ok "Root privileges (real user: $REAL_USER)" \
    || { error "Must run as root — use: sudo ./benchmark.sh"; fail=1; }

  [ -n "$IFACE" ] && ip link show "$IFACE" >/dev/null 2>&1 \
    && ok "Interface: $IFACE" \
    || { error "Interface '$IFACE' not found — set IFACE=... manually"; fail=1; }

  command -v tc >/dev/null \
    && ok "tc (iproute2) found" \
    || { error "tc not found — apt install iproute2"; fail=1; }

  [ -n "$LOCUST_BIN" ] && [ -x "$LOCUST_BIN" ] \
    && ok "locust found: $LOCUST_BIN" \
    || { error "locust not found — activate venv or pip install locust"; fail=1; }

  command -v gsutil >/dev/null 2>&1 \
    && ok "gsutil found" \
    || warn "gsutil not found — GCS upload will be skipped"

  [ -f "$SCRIPT_DIR/locustfile.py" ] \
    && ok "locustfile.py found" \
    || { error "locustfile.py not found"; fail=1; }

  [ -d "$SCRIPT_DIR/test-images" ] \
    && ok "test-images/ found ($(ls "$SCRIPT_DIR/test-images"/*.jpg 2>/dev/null | wc -l) images)" \
    || { error "test-images/ not found"; fail=1; }

  [ -d "$SCRIPT_DIR/../protobuf" ] \
    && ok "protobuf/ found" \
    || warn "protobuf/ not found — gRPC compilation may fail"

  echo "------------------------------------------"
  if [ "$fail" -eq 0 ]; then
    info "DOCTOR: OK — READY"
  else
    error "DOCTOR: FAILED — fix issues above"
    exit 1
  fi
}

# ─── GCS Upload ──────────────────────────────────────────────────────────────
upload_file() {
  local file=$1 net=$2 scenario=$3 proto=$4

  command -v gsutil >/dev/null 2>&1 || { warn "Upload skipped: gsutil missing"; return 0; }

  info "Unthrottling for upload..."
  net_reset; sleep 1

  local dest="$BUCKET/$EXP_NAME/$net/$scenario/$proto/"
  local attempt=1

  while [ $attempt -le 3 ]; do
    gsutil cp "$file" "$dest" >>"$LOGFILE" 2>&1 \
      && { ok "Uploaded $(basename "$file") → $dest"; break; } \
      || { warn "Upload attempt $attempt/3 failed"; sleep 3; ((attempt++)); }
  done

  [ $attempt -gt 3 ] && error "Upload permanently failed: $(basename "$file")"

  info "Restoring network: $net"
  net_apply "$net"; sleep 1
}

# ─── Run single test ─────────────────────────────────────────────────────────
run_test() {
  local proto=$1 scenario=$2 net=$3
  local ts; ts=$(date +%Y%m%d_%H%M%S)
  local user_class csv_prefix

  case "$proto" in
    grpc) user_class="GrpcUser" ;;
    rest) user_class="RestUser" ;;
    *)    error "Unknown proto: $proto"; return 1 ;;
  esac

  csv_prefix="$RESULTS_DIR/${proto}_${net}_${scenario}_${ts}"
  info "▶ [$proto] scenario=$scenario network=$net workers=$LOCUST_WORKERS"

  set +e
  sudo -u "$REAL_USER" -H \
    env SCENARIO="$scenario" NETWORK="$net" EXP_NAME="$EXP_NAME" \
        RESULTS_DIR="$RESULTS_DIR" GRPC_ADDR="$GRPC_ADDR" REST_ADDR="$REST_ADDR" \
    "$LOCUST_BIN" \
      -f "$SCRIPT_DIR/locustfile.py" "$user_class" \
      --headless \
      --csv="$csv_prefix" \
      2>&1 | tee -a "$LOGFILE"
  local rc=$?
  set -e

  if [ $rc -eq 0 ]; then
    ok "Done: $proto $scenario $net"
    for f in "$csv_prefix"*.csv; do
      [ -s "$f" ] && upload_file "$f" "$net" "$scenario" "$proto" || true
    done
    local metrics_csv
    metrics_csv=$(ls "$RESULTS_DIR/${EXP_NAME}_${net}_${scenario}_"*"_metrics.csv" 2>/dev/null | tail -1 || true)
    [ -n "$metrics_csv" ] && [ -s "$metrics_csv" ] \
      && upload_file "$metrics_csv" "$net" "$scenario" "$proto" || true
  else
    error "Failed: $proto $scenario $net (exit $rc)"
  fi
}

# ─── Full suite ──────────────────────────────────────────────────────────────
full_suite() {
  info "=========================================="
  info " FULL BENCHMARK SUITE"
  info " Experiment : $EXP_NAME | Interface: $IFACE"
  info " Workers    : $LOCUST_WORKERS / $CPU_COUNT cores"
  info " Log        : $LOGFILE"
  info "=========================================="

  for net in "${NETWORKS[@]}"; do
    info "▶▶ Network: $net"
    net_apply "$net"; sleep 2

    for scenario in "${SCENARIOS[@]}"; do
      for proto in "${PROTOS[@]}"; do
        run_test "$proto" "$scenario" "$net" || true
        sleep 2
      done
    done

    net_reset; sleep 3
  done

  info "=========================================="
  info " BENCHMARK COMPLETED"
  info "=========================================="
}

# ─── Entry point ─────────────────────────────────────────────────────────────
case "${1:-full}" in
  doctor) doctor ;;
  net)    net_apply "${2:-normal}" ;;
  test)   run_test "${2:-grpc}" "${3:-smoke}" "${4:-normal}" ;;
  full)   doctor && full_suite ;;
  *)
    echo "Usage:"
    echo "  sudo ./benchmark.sh doctor"
    echo "  sudo ./benchmark.sh net <normal|poor|worst|3g|4g>"
    echo "  sudo ./benchmark.sh test <grpc|rest> <scenario> <network>"
    echo "  sudo GRPC_ADDR=10.0.0.1:8008 REST_ADDR=http://10.0.0.1:8088 EXP_NAME=percobaan-1 ./benchmark.sh full"
    exit 1 ;;
esac