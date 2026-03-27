#!/bin/bash
# benchmark.sh — Locust benchmark runner dengan resume state
#
#   sudo ./benchmark.sh full        # jalankan / lanjutkan dari state
#   sudo ./benchmark.sh resume      # alias full (sama saja)
#   sudo ./benchmark.sh reset       # hapus state, mulai dari awal
#   sudo ./benchmark.sh doctor
#   sudo ./benchmark.sh net poor
#   sudo ./benchmark.sh test grpc smoke normal

set -uo pipefail
# CATATAN: set -e dihapus dari global karena bentrok dengan trap ERR —
# setiap perintah yang gagal (termasuk yang sudah di-handle) akan trigger trap.
# Kita tangani exit code secara eksplisit di setiap fungsi.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ─── Auto-detect ─────────────────────────────────────────────────────────────
REAL_USER=${SUDO_USER:-$USER}
REAL_GROUP=$(id -gn "$REAL_USER")
# FIX #10: hindari eval untuk ekspansi home — gunakan getent atau ~username
REAL_HOME=$(getent passwd "$REAL_USER" | cut -d: -f6)
IFACE=${IFACE:-$(ip -o link show up | awk -F': ' '$2 != "lo" {print $2; exit}')}
CPU_COUNT=$(nproc --all 2>/dev/null || echo 2)
# FIX #13: LOCUST_WORKERS sebelumnya dihitung tapi tidak dipakai — hapus saja
# agar tidak menyesatkan pembaca

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
STATE_FILE="$SCRIPT_DIR/.state_${EXP_NAME}"
# FIX #2: lock file untuk state_mark agar atomic (hindari race condition)
STATE_LOCK="$SCRIPT_DIR/.state_${EXP_NAME}.lock"

state_done() {
  grep -qxF "$1:$2:$3" "$STATE_FILE" 2>/dev/null
}

state_mark() {
  # FIX #2 & #3: tulis atomic via lock, dan cek duplikat sebelum tulis
  # sehingga state_progress tidak bisa overcount
  (
    flock -x 200
    if ! grep -qxF "$1:$2:$3" "$STATE_FILE" 2>/dev/null; then
      echo "$1:$2:$3" >> "$STATE_FILE"
    fi
  ) 200>"$STATE_LOCK"
}

state_reset() {
  rm -f "$STATE_FILE" "$STATE_LOCK"
  ok "State dihapus — percobaan akan mulai dari awal"
}

state_progress() {
  local total=$(( ${#NETWORKS[@]} * ${#SCENARIOS[@]} * ${#PROTOS[@]} ))
  local done=0
  # FIX #3: sort -u untuk deduplikasi sebelum hitung, walaupun state_mark
  # sudah mencegah duplikat — ini defence in depth
  if [ -f "$STATE_FILE" ]; then
    done=$(sort -u "$STATE_FILE" | wc -l)
  fi
  echo "$done / $total job selesai"
}

# ─── Setup ───────────────────────────────────────────────────────────────────
mkdir -p "$RESULTS_DIR" "$LOG_DIR"
LOGFILE="$LOG_DIR/${EXP_NAME}_$(date +%Y%m%d_%H%M%S).log"
touch "$LOGFILE"
# FIX #9: chown hanya pada LOGFILE baru, bukan rekursif seluruh direktori
chown "$REAL_USER:$REAL_GROUP" "$LOGFILE"

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
  # FIX #11: jangan simpan tc command sebagai string lalu eval/run —
  # jalankan langsung dengan argumen eksplisit per case
  case "$mode" in
    normal)
      ok "Network: normal"
      return 0
      ;;
    poor)
      tc qdisc add dev "$IFACE" root netem delay 100ms loss 1% 2>>"$LOGFILE" \
        && ok "Network: poor" || warn "tc gagal mode 'poor' — lanjut tanpa throttle"
      ;;
    worst)
      tc qdisc add dev "$IFACE" root netem delay 300ms loss 3% 2>>"$LOGFILE" \
        && ok "Network: worst" || warn "tc gagal mode 'worst' — lanjut tanpa throttle"
      ;;
    3g)
      tc qdisc add dev "$IFACE" root netem delay 200ms loss 2% rate 384kbit 2>>"$LOGFILE" \
        && ok "Network: 3g" || warn "tc gagal mode '3g' — lanjut tanpa throttle"
      ;;
    4g)
      tc qdisc add dev "$IFACE" root netem delay 80ms loss 0.5% rate 10mbit 2>>"$LOGFILE" \
        && ok "Network: 4g" || warn "tc gagal mode '4g' — lanjut tanpa throttle"
      ;;
    *)
      error "Unknown network mode: $mode"
      return 1
      ;;
  esac
}

# FIX #4 & #16: trap hanya INT/TERM/EXIT — hapus ERR agar tidak trigger
# pada perintah yang gagalnya sudah ditangani (|| true, set +e, dsb)
# EXIT sudah cukup untuk cleanup saat script selesai normal maupun abort
trap 'net_reset; _log "Script selesai/interrupted — network cleaned up"' INT TERM EXIT

# ─── Doctor ──────────────────────────────────────────────────────────────────
doctor() {
  echo "=========================================="
  echo " DOCTOR  |  iface=$IFACE  cores=$CPU_COUNT"
  echo "=========================================="
  local fail=0

  [ "$EUID" -eq 0 ]                                         && ok  "root ok (user: $REAL_USER)"   || { error "Perlu sudo"; fail=1; }
  [ -n "$IFACE" ] && ip link show "$IFACE" >/dev/null 2>&1  && ok  "iface: $IFACE"                || { error "iface '$IFACE' tidak ditemukan"; fail=1; }
  command -v tc >/dev/null                                   && ok  "tc ok"                        || { error "tc tidak ada — apt install iproute2"; fail=1; }
  # FIX #1: validasi LOCUST_BIN lebih dulu sebelum cek -x
  [ -n "$LOCUST_BIN" ] && [ -x "$LOCUST_BIN" ]              && ok  "locust: $LOCUST_BIN"          || { error "locust tidak ditemukan (cek venv atau PATH)"; fail=1; }
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

  # FIX #8 (relevan juga di sini): pastikan file ada dan tidak kosong
  if [ ! -s "$file" ]; then
    warn "Upload skip: file tidak ada atau kosong — $(basename "$file")"
    return 0
  fi

  # FIX #6: simpan state jaringan saat ini agar bisa di-restore dengan benar
  # net_reset dulu agar upload tidak throttled, lalu kembalikan ke $net
  net_reset
  sleep 1
  local dest="$BUCKET/$EXP_NAME/$net/$scenario/$proto/"
  local attempt=1

  while [ $attempt -le 3 ]; do
    if gsutil cp "$file" "$dest" >>"$LOGFILE" 2>&1; then
      ok "Uploaded $(basename "$file") → $dest"
      break
    else
      warn "Upload attempt $attempt/3 gagal"
      sleep 3
      (( attempt++ )) || true
    fi
  done

  if [ $attempt -gt 3 ]; then
    error "Upload gagal permanen: $(basename "$file")"
  fi

  # Kembalikan throttle ke kondisi semula
  net_apply "$net"
  sleep 1
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

  local rc=0
  sudo -u "$REAL_USER" -H \
    env SCENARIO="$scenario" NETWORK="$net" EXP_NAME="$EXP_NAME" \
        PROTO="$proto" \
        RESULTS_DIR="$RESULTS_DIR" GRPC_ADDR="$GRPC_ADDR" REST_ADDR="$REST_ADDR" \
    "$LOCUST_BIN" \
      -f "$SCRIPT_DIR/locustfile.py" "$user_class" \
      --headless \
      --csv="$csv_prefix" \
      --exit-code-on-error 0 \
      >>"$LOGFILE" 2>&1 || rc=$?

  if [ $rc -eq 0 ]; then
    ok "Selesai: $proto $scenario $net"
    state_mark "$proto" "$scenario" "$net"

    # FIX #8: cek eksistensi file sebelum loop — hindari upload literal "*.csv"
    local f
    for f in "$csv_prefix"_stats.csv \
              "$csv_prefix"_stats_history.csv \
              "$csv_prefix"_failures.csv \
              "$csv_prefix"_exceptions.csv; do
      [ -s "$f" ] && upload_file "$f" "$net" "$scenario" "$proto" || true
    done

    # Upload custom metrics CSV dari CsvListener — nama eksak, tidak pakai wildcard
    local metrics_csv="$RESULTS_DIR/${EXP_NAME}_${net}_${scenario}_${proto}_metrics.csv"
    [ -s "$metrics_csv" ] && upload_file "$metrics_csv" "$net" "$scenario" "$proto" || true

  else
    # FIX #14: bedakan "test failure" vs "script error" dengan exit code
    error "Gagal (exit $rc): $proto $scenario $net — akan dicoba ulang saat resume"
    _log "FAILED rc=$rc proto=$proto scenario=$scenario net=$net"
  fi
}

# ─── Full suite ──────────────────────────────────────────────────────────────
full_suite() {
  info "=========================================="
  info " $EXP_NAME | iface=$IFACE | cores=$CPU_COUNT"
  info " state → $STATE_FILE"
  info " log   → $LOGFILE"
  info " progress: $(state_progress)"
  info "=========================================="

  local net scenario proto
  for net in "${NETWORKS[@]}"; do
    info "── Network: $net ──"
    net_apply "$net"
    sleep 2

    for scenario in "${SCENARIOS[@]}"; do
      for proto in "${PROTOS[@]}"; do

        if state_done "$proto" "$scenario" "$net"; then
          echo "  ⏭ skip $proto $scenario $net (sudah selesai)"
          _log "SKIP $proto:$scenario:$net"
          continue
        fi

        # FIX #14: tangkap error run_test secara eksplisit, bukan swallow semua
        if ! run_test "$proto" "$scenario" "$net"; then
          warn "run_test gagal untuk $proto $scenario $net — lanjut ke job berikutnya"
        fi
        sleep 2

      done
    done

    net_reset
    sleep 3
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
  # FIX #15: reset tidak perlu assign ulang EXP_NAME/STATE_FILE — sudah ada di atas
  reset)  state_reset ;;
  *)
    echo "Usage:"
    echo "  sudo ./benchmark.sh full              # jalankan / lanjutkan otomatis"
    echo "  sudo ./benchmark.sh resume            # alias full"
    echo "  sudo EXP_NAME=percobaan-1 ./benchmark.sh reset"
    echo "  sudo ./benchmark.sh doctor"
    echo "  sudo ./benchmark.sh net <normal|poor|worst|3g|4g>"
    echo "  sudo ./benchmark.sh test <grpc|rest> <scenario> <network>"
    exit 1 ;;
esac