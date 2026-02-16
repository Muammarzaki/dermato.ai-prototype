#!/bin/bash
set -e

IFACE=${IFACE:-eth0}

info() { echo -e "\033[0;34mℹ $1\033[0m"; }
ok()   { echo -e "\033[0;32m✓ $1\033[0m"; }
warn() { echo -e "\033[1;33m⚠ $1\033[0m"; }
error(){ echo -e "\033[0;31m✗ $1\033[0m"; }

[ "$EUID" -eq 0 ] || { error "Run as root"; exit 1; }
ip link show "$IFACE" &>/dev/null || { error "Interface $IFACE not found"; exit 1; }

reset_net() {
    tc qdisc del dev "$IFACE" root 2>/dev/null || true
    ok "Network NORMAL"
}

poor() {
    reset_net
    tc qdisc add dev "$IFACE" root netem delay 100ms 20ms loss 1% rate 10mbit
    ok "POOR network active"
}

worst() {
    reset_net
    tc qdisc add dev "$IFACE" root netem delay 300ms 50ms loss 3% rate 2mbit
    ok "WORST network active"
}

g3() {
    reset_net
    tc qdisc add dev "$IFACE" root netem delay 100ms 50ms loss 1% rate 384kbit
    ok "3G network active"
}

g4() {
    reset_net
    tc qdisc add dev "$IFACE" root netem delay 50ms 10ms loss 0.1% rate 10mbit
    ok "4G network active"
}

case "${1:-help}" in
    normal) reset_net ;;
    poor)   poor ;;
    worst)  worst ;;
    3g)     g3 ;;
    4g)     g4 ;;
    *) error "Usage: normal | poor | worst | 3g | 4g" ;;
esac
