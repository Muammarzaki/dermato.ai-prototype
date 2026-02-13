#!/bin/bash
set -e

# ==========================
# CONFIG
# ==========================
IFACE=${IFACE:-eth0}

# ==========================
# COLOR LOG
# ==========================
info() { echo -e "\033[0;34mℹ $1\033[0m"; }
ok()   { echo -e "\033[0;32m✓ $1\033[0m"; }
warn() { echo -e "\033[1;33m⚠ $1\033[0m"; }

# ==========================
# CHECK ROOT
# ==========================
if [ "$EUID" -ne 0 ]; then
  warn "Run as root: sudo $0 $*"
  exit 1
fi

# ==========================
# RESET NETWORK
# ==========================
reset_net() {
    info "Resetting network simulation on $IFACE"
    tc qdisc del dev "$IFACE" root 2>/dev/null || true
    ok "Network back to NORMAL"
}

# ==========================
# POOR NETWORK
# ==========================
poor_net() {
    reset_net
    info "Applying POOR network simulation"
    tc qdisc add dev "$IFACE" root netem delay 300ms loss 2%
    ok "POOR network active (300ms, 2% loss)"
}

# ==========================
# WORST NETWORK
# ==========================
worst_net() {
    reset_net
    info "Applying WORST network simulation"
    tc qdisc add dev "$IFACE" root netem delay 800ms loss 5%
    ok "WORST network active (800ms, 5% loss)"
}

# ==========================
# STATUS
# ==========================
status() {
    info "Current qdisc on $IFACE:"
    tc qdisc show dev "$IFACE"
}

# ==========================
# MAIN
# ==========================
case "${1:-help}" in
    normal)
        reset_net
        ;;
    poor)
        poor_net
        ;;
    worst)
        worst_net
        ;;
    status)
        status
        ;;
    *)
        echo "Usage:"
        echo "  sudo IFACE=eth0 ./net-sim.sh normal"
        echo "  sudo IFACE=eth0 ./net-sim.sh poor"
        echo "  sudo IFACE=eth0 ./net-sim.sh worst"
        echo "  sudo IFACE=eth0 ./net-sim.sh status"
        exit 1
        ;;
esac
