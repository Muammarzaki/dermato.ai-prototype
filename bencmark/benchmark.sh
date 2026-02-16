#!/bin/bash
# ==============================================================================
# Skin Analyzer Benchmark Suite - Hardened & Error Tolerant Version
# ==============================================================================

# Strict bash mode:
# -e: exit on error, -u: exit on undefined variable, -o pipefail: catch errors in pipelines
set -euo pipefail

# Menentukan direktori script absolute untuk menghindari relative path error
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TEST_DIR="$SCRIPT_DIR/src/tests"

### ========= GLOBAL CONFIG =========
# Menggunakan default value, tapi bisa di-override via env vars
IFACE=${IFACE:-enp4}
EXP_NAME=${EXP_NAME:-percobaan-1}
BUCKET=${BUCKET:-"gs://benchmark-2026"} # Pastikan nama bucket ini valid/unik!
RESULTS_DIR="$SCRIPT_DIR/results"
LOG_DIR="$SCRIPT_DIR/logs"

NETWORKS=(normal poor worst 3g 4g)
SCENARIOS=(smoke load stress spike soak)
PROTOS=(grpc rest)

# Deteksi user asli yang mengeksekusi sudo, fallback ke user saat ini jika bukan sudo
REAL_USER=${SUDO_USER:-$USER}
REAL_GROUP=$(id -gn "$REAL_USER")

# Setup folder & file log
mkdir -p "$RESULTS_DIR" "$LOG_DIR"
LOGFILE="$LOG_DIR/${EXP_NAME}_$(date +%Y%m%d_%H%M%S).log"
touch "$LOGFILE"

# Kembalikan kepemilikan folder log & result ke user asli, bukan root!
chown -R "$REAL_USER:$REAL_GROUP" "$RESULTS_DIR" "$LOG_DIR"

### ========= LOGGING HELPERS =========
# Log otomatis tercatat ke file dan terminal dengan warna
log()  { echo "[$(date '+%F %T')] $1" | tee -a "$LOGFILE"; }
info() { echo -e "\033[0;34mℹ $1\033[0m" | tee -a "$LOGFILE"; }
warn() { echo -e "\033[1;33m⚠ $1\033[0m" | tee -a "$LOGFILE"; }
error(){ echo -e "\033[0;31m✗ $1\033[0m" >&2 | tee -a "$LOGFILE"; }

### ========= CLEANUP TRAP =========
# Trap akan SELALU dieksekusi ketika script berhenti (sukses, error, atau di-cancel Ctrl+C)
net_reset() {
  # 2>/dev/null menyembunyikan error jika qdisc memang belum ada (idempotent)
  tc qdisc del dev "$IFACE" root 2>/dev/null || true
  info "Network traffic control reset on $IFACE."
}
trap 'net_reset' INT TERM EXIT ERR

### ========= DOCTOR =========
doctor() {
  echo "=========================================="
  echo " BENCHMARK DOCTOR CHECK (PRE-FLIGHT)"
  echo "=========================================="
  local fail=0

  # 1. Cek Root
  if [ "$EUID" -ne 0 ]; then
    error "Must run as root (use: sudo ./benchmark.sh)"
    fail=1
  else
    echo "✓ Running with root privileges (sudo mode detected: user $REAL_USER)"
  fi

  # 2. Cek Network Interface
  if ip link show "$IFACE" >/dev/null 2>&1; then
    echo "✓ Network interface found: $IFACE"
  else
    error "Network interface not found: $IFACE. Override with IFACE=your_eth ./benchmark.sh"
    fail=1
  fi

  # 3. Cek Dependensi
  for cmd in tc gsutil; do
    if command -v "$cmd" >/dev/null; then
      echo "✓ Command '$cmd' is installed"
    else
      warn "Command '$cmd' is missing. Some features may not work."
      [ "$cmd" == "tc" ] && fail=1 # tc is mandatory
    fi
  done

  # 4. Cek k6 via real user
  if sudo -u "$REAL_USER" command -v k6 >/dev/null; then
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

### ========= NETWORK =========
net_apply() {
  local mode=$1
  net_reset # Pastikan aturan lama bersih sebelum pasang yang baru

  # Pasang qdisc baru. Jika gagal, script tidak langsung crash berkat penanganan || log
  case "$mode" in
    normal) ;;
    poor)  tc qdisc add dev "$IFACE" root netem delay 100ms loss 1% || warn "Failed to set poor network" ;;
    worst) tc qdisc add dev "$IFACE" root netem delay 300ms loss 3% || warn "Failed to set worst network" ;;
    3g)    tc qdisc add dev "$IFACE" root netem delay 200ms loss 2% rate 384kbit || warn "Failed to set 3G network" ;;
    4g)    tc qdisc add dev "$IFACE" root netem delay 80ms loss 0.5% rate 10mbit || warn "Failed to set 4G network" ;;
    *)     error "Unknown network mode: $mode"; exit 1 ;;
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
    error "Test script not found: $test_file"
    return 1
  fi

  info "Starting k6: [$proto] | Scenario: [$scenario] | Network: [$net]"

  # EXECUTION CORE: Jalankan k6 sebagai user asli (-u) dan bawa environment variables (-E)
  # Set +e agar k6 failure tidak mematikan bash script utama
  set +e
  sudo -u "$REAL_USER" -E k6 run -e SCENARIO="$scenario" "$test_file" --out csv="$out" >>"$LOGFILE" 2>&1
  local exit_code=$?
  set -e

  if [ $exit_code -eq 0 ] && [ -f "$out" ]; then
    echo "$out" # Print filename for upload phase
    return 0
  else
    error "Test failed for $proto $scenario on $net network (Exit code: $exit_code)"
    sudo -u "$REAL_USER" rm -f "$out" # Clean up empty/corrupt CSV
    return 1
  fi
}

### ========= UPLOAD =========
upload_file() {
  local file=$1
  local net=$2
  local scenario=$3

  if ! sudo -u "$REAL_USER" command -v gsutil >/dev/null 2>&1; then
    warn "Upload skipped: gsutil missing"
    return 0
  fi

  info "Uploading $(basename "$file") to GCP..."

  # EXECUTION CORE: gsutil dijalankan sebagai user asli agar otentikasi GCP terbaca
  set +e
  sudo -u "$REAL_USER" gsutil cp "$file" "$BUCKET/$EXP_NAME/$net/$scenario/" >>"$LOGFILE" 2>&1
  local upload_status=$?
  set -e

  if [ $upload_status -ne 0 ]; then
    error "Upload failed for $(basename "$file"). Check log for details."
  else
    log "Upload successful."
  fi
}

### ========= FULL SUITE =========
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

        # Eksekusi test. Jika gagal, catat error dan lanjut iterasi berikutnya (Fault Tolerant)
        if result_file=$(run_test "$proto" "$scenario" "$net"); then
          upload_file "$result_file" "$net" "$scenario"
        else
          warn "Skipping upload for failed test: $proto $scenario"
        fi

      done
    done

    net_reset
    sleep 3
  done

  info "=========================================="
  info " FULL BENCHMARK COMPLETED SUCCESSFULLY"
  info "=========================================="
}

### ========= MAIN ROUTINE =========
# Memastikan setidaknya satu argument diberikan, jika tidak jalankan 'full'
ACTION=${1:-full}

case "$ACTION" in
  doctor) doctor ;;
  net)    net_apply "${2:-normal}" ;;
  test)   run_test grpc smoke normal ;;
  full)   doctor && full_suite ;;
  *)
    echo "Usage:"
    echo "  sudo IFACE=enp4 EXP_NAME=percobaan-1 ./benchmark.sh doctor"
    echo "  sudo IFACE=enp4 ./benchmark.sh net <normal|poor|worst|3g|4g>"
    echo "  sudo IFACE=enp4 ./benchmark.sh test"
    echo "  sudo REST_ADDR=http://10.128.0.2:8088 IFACE=enp4 ./benchmark.sh full"
    exit 1
  ;;
esac