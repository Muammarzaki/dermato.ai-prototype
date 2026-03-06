#!/bin/bash
# benchmark.sh — Locust benchmark runner dengan resume state
#
#   sudo ./benchmark.sh full        # jalankan / lanjutkan dari state
#   sudo ./benchmark.sh resume      # alias full (sama saja)
#   sudo ./benchmark.sh reset       # hapus state, mulai dari awal
#   sudo ./benchmark.sh doctor
#   sudo ./benchmark.sh net poor
#   sudo ./benchmark.sh test grpc smoke normal

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ─── Auto-detect ─────────────────────────────────────────────────────────────
REAL_USER=${SUDO_USER:-$USER}
REAL_GROUP=$(id -gn "$REAL_USER")
REAL_HOME=$(eval echo "~$REAL_USER")
IFACE=${IFACE:-$(ip -o link show up | awk -F': ' '$2 != "lo" {print $2; exit}')}
CPU_COUNT=$(nproc --all 2>/dev/null || echo 2)
LOCUST_WORKERS=$(( CPU_COUNT > 2 ? CPU_COUNT - 1 : 1 ))

_find_locust() {
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

# ─── State file — satu file per EXP_NAME ─────────────────────────────────────
# Format: satu baris per job yang SELESAI → "proto:scenario:network"
# Contoh: grpc:load:poor
STATE_FILE="$SCRIPT_DIR/.state_${EXP_NAME}"

state_done()  {
  # Cek apakah kombinasi ini sudah selesai
  grep -qxF "$1:$2:$3" "$STATE_FILE" 2>/dev/null
}

state_mark()  {
  # Tandai kombinasi ini sebagai selesai
  echo "$1:$2:$3" >> "$STATE_FILE"
}

state_reset() {
  rm -f "$STATE_FILE"
  ok "State dihapus — percobaan akan mulai dari awal"
}

state_progress() {
  local total=$(( ${#NETWORKS[@]} * ${#SCENARIOS[@]} * ${#PROTOS[@]} ))
  local done=0
  [ -f "$STATE_FILE" ] && done=$(wc -l < "$STATE_FILE") || done=0
  echo "$done / $total job selesai"
}

# ─── Setup ───────────────────────────────────────────────────────────────────
mkdir -p "$RESULTS_DIR" "$LOG_DIR"
LOGFILE="$LOG_DIR/${EXP_NAME}_$(date +%Y%m%d_%H%M%S).log"
touch "$LOGFILE"
chown -R "$REAL_USER:$REAL_GROUP" "$RESULTS_DIR" "$LOG_DIR"

# ─── Logging ─────────────────────────────────────────────────────────────────
_log()  { echo "[$(date '+%F %T')] $*" >> "$LOGFILE"; }
info()  { local msg="ℹ $*";  echo -e "\033[0;34m$msg\033[0m"; _log "$msg"; }
warn()  { local msg="⚠ $*";  echo -e "\033[1;33m$msg\033[0m"; _log "$msg"; }
ok()    { local msg="✓ $*";  echo -e "\033[0;32m$msg\033[0m"; _log "$msg"; }
error() { local msg="✗ $*";  echo -e "\033[0;31m$msg\033[0m"; _log "$msg"; }

# ─── Network ─────────────────────────────────────────────────────────────────
net_reset() {
  tc qdisc del dev "$IFACE" root 2>/dev/null || true
  _log "Network reset on $IFACE"
}

net_apply() {
  local mode=$1
  net_reset
  local tc_cmd=""
  case "$mode" in
    normal) ok "Network: normal"; return ;;
    poor)   tc_cmd="tc qdisc add dev $IFACE root netem delay 100ms loss 1%" ;;
    worst)  tc_cmd="tc qdisc add dev $IFACE root netem delay 300ms loss 3%" ;;
    3g)     tc_cmd="tc qdisc add dev $IFACE root netem delay 200ms loss 2% rate 384kbit" ;;
    4g)     tc_cmd="tc qdisc add dev $IFACE root netem delay 80ms loss 0.5% rate 10mbit" ;;
    *)      error "Unknown network mode: $mode"; return 1 ;;
  esac

  if $tc_cmd 2>>"$LOGFILE"; then
    ok "Network: $mode"
  else
    warn "tc gagal mode '$mode' — lanjut tanpa throttle"
  fi
}

trap 'net_reset; _log "Interrupted — network cleaned up"' INT TERM EXIT ERR

# ─── Doctor ──────────────────────────────────────────────────────────────────
doctor() {
  echo "=========================================="
  echo " DOCTOR  |  iface=$IFACE  cores=$CPU_COUNT"
  echo "=========================================="
  local fail=0

  [ "$EUID" -eq 0 ]                                         && ok  "root ok (user: $REAL_USER)"   || { error "Perlu sudo"; fail=1; }
  [ -n "$IFACE" ] && ip link show "$IFACE" >/dev/null 2>&1  && ok  "iface: $IFACE"                || { error "iface '$IFACE' tidak ditemukan"; fail=1; }
  command -v tc >/dev/null                                   && ok  "tc ok"                        || { error "tc tidak ada — apt install iproute2"; fail=1; }
  [ -n "$LOCUST_BIN" ] && [ -x "$LOCUST_BIN" ]              && ok  "locust: $LOCUST_BIN"          || { error "locust tidak ditemukan"; fail=1; }
  command -v gsutil >/dev/null 2>&1                          && ok  "gsutil ok"                    || warn "gsutil tidak ada — upload dilewati"
  [ -f "$SCRIPT_DIR/locustfile.py" ]                         && ok  "locustfile.py ok"             || { error "locustfile.py tidak ada"; fail=1; }
  [ -d "$SCRIPT_DIR/test-images" ]                           && ok  "test-images ok"               || { error "test-images/ tidak ada"; fail=1; }
  [ -d "$SCRIPT_DIR/../protobuf" ]                           && ok  "protobuf ok"                  || warn "protobuf/ tidak ada"

  echo "------------------------------------------"
  [ "$fail" -eq 0 ] && info "DOCTOR: READY" || { error "DOCTOR: FAILED"; exit 1; }
}

# ─── GCS Upload ──────────────────────────────────────────────────────────────
upload_file() {
  local file=$1 net=$2 scenario=$3 proto=$4
  command -v gsutil >/dev/null 2>&1 || { warn "Upload skip: no gsutil"; return 0; }

  net_reset; sleep 1
  local dest="$BUCKET/$EXP_NAME/$net/$scenario/$proto/"
  local attempt=1

  while [ $attempt -le 3 ]; do
    gsutil cp "$file" "$dest" >>"$LOGFILE" 2>&1 \
      && { ok "Uploaded $(basename "$file") → $dest"; break; } \
      || { warn "Upload attempt $attempt/3 gagal"; sleep 3; ((attempt++)); }
  done

  [ $attempt -gt 3 ] && error "Upload gagal permanen: $(basename "$file")"
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
  info "▶ [$proto] $scenario / $net"

  set +e
  sudo -u "$REAL_USER" -H \
    env SCENARIO="$scenario" NETWORK="$net" EXP_NAME="$EXP_NAME" \
        RESULTS_DIR="$RESULTS_DIR" GRPC_ADDR="$GRPC_ADDR" REST_ADDR="$REST_ADDR" \
    "$LOCUST_BIN" \
      -f "$SCRIPT_DIR/locustfile.py" "$user_class" \
      --headless \
      --csv="$csv_prefix" \
      --exit-code-on-error 0 \
      >>"$LOGFILE" 2>&1
  local rc=$?
  set -e

  if [ $rc -eq 0 ]; then
    ok "Selesai: $proto $scenario $net"

    # Tandai selesai di state SETELAH sukses
    state_mark "$proto" "$scenario" "$net"

    for f in "$csv_prefix"*.csv; do
      [ -s "$f" ] && upload_file "$f" "$net" "$scenario" "$proto" || true
    done
    local metrics_csv
    metrics_csv=$(ls "$RESULTS_DIR/${EXP_NAME}_${net}_${scenario}_"*"_metrics.csv" 2>/dev/null | tail -1 || true)
    [ -n "$metrics_csv" ] && [ -s "$metrics_csv" ] \
      && upload_file "$metrics_csv" "$net" "$scenario" "$proto" || true
  else
    # Tidak mark state — akan dicoba lagi saat resume
    error "Gagal: $proto $scenario $net (exit $rc) — akan dicoba ulang saat resume"
  fi
}

# ─── Full suite ──────────────────────────────────────────────────────────────
full_suite() {
  info "=========================================="
  info " $EXP_NAME | iface=$IFACE | workers=$LOCUST_WORKERS/$CPU_COUNT"
  info " state → $STATE_FILE"
  info " log   → $LOGFILE"
  info " progress: $(state_progress)"
  info "=========================================="

  for net in "${NETWORKS[@]}"; do
    info "── Network: $net ──"
    net_apply "$net"; sleep 2

    for scenario in "${SCENARIOS[@]}"; do
      for proto in "${PROTOS[@]}"; do

        if state_done "$proto" "$scenario" "$net"; then
          echo "  ⏭ skip $proto $scenario $net (sudah selesai)"
          _log "SKIP $proto:$scenario:$net (sudah ada di state)"
          continue
        fi

        run_test "$proto" "$scenario" "$net" || true
        sleep 2

      done
    done

    net_reset; sleep 3
  done

  info "=========================================="
  info " SELESAI: $EXP_NAME — $(state_progress)"
  info "=========================================="
}

# ─── Entry point ─────────────────────────────────────────────────────────────
case "${1:-full}" in
  doctor) doctor ;;
  net)    net_apply "${2:-normal}" ;;
  test)   run_test "${2:-grpc}" "${3:-smoke}" "${4:-normal}" ;;
  full|resume) doctor && full_suite ;;
  reset)
    EXP_NAME=${EXP_NAME:-percobaan-1}
    STATE_FILE="$SCRIPT_DIR/.state_${EXP_NAME}"
    state_reset ;;
  *)
    echo "Usage:"
    echo "  sudo ./benchmark.sh full              # jalankan / lanjutkan otomatis"
    echo "  sudo ./benchmark.sh resume            # alias full"
    echo "  sudo EXP_NAME=percobaan-1 ./benchmark.sh reset  # hapus state, mulai ulang"
    echo "  sudo ./benchmark.sh doctor"
    echo "  sudo ./benchmark.sh net <normal|poor|worst|3g|4g>"
    echo "  sudo ./benchmark.sh test <grpc|rest> <scenario> <network>"
    exit 1 ;;
esac