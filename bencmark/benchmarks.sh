#!/bin/bash
# benchmark.sh — Menggantikan benchmark.sh k6
# Jalankan dari dalam folder bencmark/
#
# Contoh:
#   sudo IFACE=ens4 EXP_NAME=percobaan-1 ./benchmark.sh full
#   sudo IFACE=ens4 ./benchmark.sh doctor
#   sudo IFACE=ens4 ./benchmark.sh net poor
#   sudo IFACE=ens4 ./benchmark.sh test grpc smoke normal

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ─── Config ──────────────────────────────────────────────────────────────────
IFACE=${IFACE:-enp4s0}
EXP_NAME=${EXP_NAME:-percobaan-1}
BUCKET=${BUCKET:-"gs://benchmark-2026"}
RESULTS_DIR="$SCRIPT_DIR/results"
LOG_DIR="$SCRIPT_DIR/logs"

NETWORKS=(normal poor worst 3g 4g)
SCENARIOS=(smoke load stress spike soak)
PROTOS=(grpc rest)

REAL_USER=${SUDO_USER:-$USER}
REAL_GROUP=$(id -gn "$REAL_USER")

# ─── Setup dirs ───────────────────────────────────────────────────────────────
mkdir -p "$RESULTS_DIR" "$LOG_DIR"
LOGFILE="$LOG_DIR/${EXP_NAME}_$(date +%Y%m%d_%H%M%S).log"
touch "$LOGFILE"
chown -R "$REAL_USER:$REAL_GROUP" "$RESULTS_DIR" "$LOG_DIR"

# ─── Logging ─────────────────────────────────────────────────────────────────
log()   { echo "[$(date '+%F %T')] $1" | tee -a "$LOGFILE"; }
info()  { echo -e "\033[0;34mℹ $1\033[0m"  | tee -a "$LOGFILE"; }
warn()  { echo -e "\033[1;33m⚠ $1\033[0m"  | tee -a "$LOGFILE"; }
error() { echo -e "\033[0;31m✗ $1\033[0m" >&2 | tee -a "$LOGFILE"; }
ok()    { echo -e "\033[0;32m✓ $1\033[0m"  | tee -a "$LOGFILE"; }

# ─── Network ─────────────────────────────────────────────────────────────────
net_reset() {
  tc qdisc del dev "$IFACE" root 2>/dev/null || true
  log "Network reset on $IFACE"
}
trap 'net_reset' INT TERM EXIT ERR

net_apply() {
  local mode=$1
  net_reset
  case "$mode" in
    normal) ok  "Network: normal (no throttle)" ;;
    poor)   tc qdisc add dev "$IFACE" root netem delay 100ms loss 1%                    && ok  "Network: poor  (100ms, 1% loss)" || warn "Failed to set poor" ;;
    worst)  tc qdisc add dev "$IFACE" root netem delay 300ms loss 3%                    && ok  "Network: worst (300ms, 3% loss)" || warn "Failed to set worst" ;;
    3g)     tc qdisc add dev "$IFACE" root netem delay 200ms loss 2% rate 384kbit       && ok  "Network: 3G    (200ms, 2% loss, 384kbit)" || warn "Failed to set 3G" ;;
    4g)     tc qdisc add dev "$IFACE" root netem delay 80ms  loss 0.5% rate 10mbit      && ok  "Network: 4G    (80ms, 0.5% loss, 10mbit)" || warn "Failed to set 4G" ;;
    *)      error "Unknown network mode: $mode"; exit 1 ;;
  esac
}

# ─── Doctor ───────────────────────────────────────────────────────────────────
doctor() {
  echo "=========================================="
  echo " BENCHMARK DOCTOR CHECK"
  echo "=========================================="
  local fail=0

  [ "$EUID" -eq 0 ] \
    && ok "Running as root (user: $REAL_USER)" \
    || { error "Must run as root (sudo)"; fail=1; }

  ip link show "$IFACE" >/dev/null 2>&1 \
    && ok "Interface found: $IFACE" \
    || { error "Interface not found: $IFACE — override with IFACE=..."; fail=1; }

  command -v tc >/dev/null \
    && ok "tc (iproute2) installed" \
    || { error "tc not found — install iproute2"; fail=1; }

  sudo -u "$REAL_USER" -H bash -c 'command -v locust' >/dev/null 2>&1 \
    && ok "locust installed for $REAL_USER" \
    || { error "locust not found for $REAL_USER — activate venv or pip install locust"; fail=1; }

  sudo -u "$REAL_USER" -H bash -c 'command -v gsutil' >/dev/null 2>&1 \
    && ok "gsutil installed" \
    || warn "gsutil not found — GCS upload will be skipped"

  [ -f "$SCRIPT_DIR/locustfile.py" ] \
    && ok "locustfile.py found" \
    || { error "locustfile.py not found in $SCRIPT_DIR"; fail=1; }

  [ -d "$SCRIPT_DIR/../protobuf" ] \
    && ok "protobuf/ directory found" \
    || warn "protobuf/ not found at $(realpath "$SCRIPT_DIR/../protobuf" 2>/dev/null || echo '../protobuf')"

  [ -d "$SCRIPT_DIR/test-images" ] \
    && ok "test-images/ directory found" \
    || { error "test-images/ not found"; fail=1; }

  echo "------------------------------------------"
  if [ "$fail" -eq 0 ]; then
    info "DOCTOR: OK — READY TO BENCHMARK"
    return 0
  else
    error "DOCTOR: FAILED — fix issues above"
    exit 1
  fi
}

# ─── GCS Upload ───────────────────────────────────────────────────────────────
upload_file() {
  local file=$1
  local net=$2
  local scenario=$3
  local proto=$4

  sudo -u "$REAL_USER" -H bash -c 'command -v gsutil' >/dev/null 2>&1 || {
    warn "Upload skipped: gsutil missing"
    return 0
  }

  info "Unthrottling network for upload..."
  net_reset
  sleep 1

  local dest="$BUCKET/$EXP_NAME/$net/$scenario/$proto/"
  local max=3 attempt=1 success=0

  while [ $attempt -le $max ]; do
    set +e
    sudo -u "$REAL_USER" -H gsutil cp "$file" "$dest" >>"$LOGFILE" 2>&1
    local rc=$?
    set -e

    if [ $rc -eq 0 ]; then
      ok "Uploaded $(basename "$file") → $dest"
      success=1
      break
    else
      warn "Upload failed (attempt $attempt/$max)"
      sleep 3
      ((attempt++))
    fi
  done

  [ $success -eq 0 ] && error "Upload permanently failed for $(basename "$file")"

  info "Restoring network: $net"
  net_apply "$net"
  sleep 1
}

# ─── Run single test ──────────────────────────────────────────────────────────
run_test() {
  local proto=$1
  local scenario=$2
  local net=$3
  local ts
  ts=$(date +%Y%m%d_%H%M%S)

  # Tentukan User class
  local user_class
  case "$proto" in
    grpc) user_class="GrpcUser" ;;
    rest) user_class="RestUser" ;;
    *)    error "Unknown proto: $proto"; return 1 ;;
  esac

  # CSV summary dari locust (--csv flag)
  local csv_prefix="$RESULTS_DIR/${proto}_${net}_${scenario}_${ts}"

  info "▶ [$proto] scenario=$scenario network=$net"

  set +e
  sudo -u "$REAL_USER" -H \
    env \
      SCENARIO="$scenario" \
      NETWORK="$net" \
      EXP_NAME="$EXP_NAME" \
      RESULTS_DIR="$RESULTS_DIR" \
      GRPC_ADDR="${GRPC_ADDR:-127.0.0.1:8008}" \
      REST_ADDR="${REST_ADDR:-http://127.0.0.1:8088}" \
    locust \
      -f "$SCRIPT_DIR/locustfile.py" \
      "$user_class" \
      --headless \
      --csv="$csv_prefix" \
      2>&1 | tee -a "$LOGFILE"
  local exit_code=$?
  set -e

  if [ $exit_code -eq 0 ]; then
    ok "Test selesai: $proto $scenario $net"
    # Upload semua file CSV yang dihasilkan
    for f in "$csv_prefix"*.csv; do
      [ -s "$f" ] && upload_file "$f" "$net" "$scenario" "$proto" || true
    done
    # Upload juga metrics CSV custom
    local metrics_csv
    metrics_csv=$(ls "$RESULTS_DIR/${EXP_NAME}_${net}_${scenario}_"*"_metrics.csv" 2>/dev/null | tail -1 || true)
    [ -n "$metrics_csv" ] && [ -s "$metrics_csv" ] && upload_file "$metrics_csv" "$net" "$scenario" "$proto" || true
  else
    error "Test gagal: $proto $scenario $net (exit $exit_code)"
  fi

  return 0  # lanjut ke skenario berikutnya
}

# ─── Full suite ───────────────────────────────────────────────────────────────
full_suite() {
  info "=========================================="
  info " FULL BENCHMARK SUITE"
  info " Experiment : $EXP_NAME"
  info " Interface  : $IFACE"
  info " Bucket     : $BUCKET"
  info " Log        : $LOGFILE"
  info "=========================================="

  for net in "${NETWORKS[@]}"; do
    info "▶▶ Network profile: $net"
    net_apply "$net"
    sleep 2

    for scenario in "${SCENARIOS[@]}"; do
      for proto in "${PROTOS[@]}"; do
        run_test "$proto" "$scenario" "$net" || true
        sleep 2
      done
    done

    net_reset
    sleep 3
  done

  info "=========================================="
  info " FULL BENCHMARK COMPLETED"
  info "=========================================="
}

# ─── Entry point ─────────────────────────────────────────────────────────────
ACTION=${1:-full}

case "$ACTION" in
  doctor)
    doctor
    ;;
  net)
    net_apply "${2:-normal}"
    ;;
  test)
    # sudo IFACE=ens4 ./benchmark.sh test grpc smoke normal
    run_test "${2:-grpc}" "${3:-smoke}" "${4:-normal}"
    ;;
  full)
    doctor && full_suite
    ;;
  *)
    echo "Usage:"
    echo "  sudo IFACE=ens4 EXP_NAME=percobaan-1 ./benchmark.sh doctor"
    echo "  sudo IFACE=ens4 ./benchmark.sh net <normal|poor|worst|3g|4g>"
    echo "  sudo IFACE=ens4 ./benchmark.sh test <grpc|rest> <scenario> <network>"
    echo "  sudo IFACE=ens4 GRPC_ADDR=10.0.0.1:8008 REST_ADDR=http://10.0.0.1:8088 EXP_NAME=percobaan-1 ./benchmark.sh full"
    exit 1
    ;;
esac