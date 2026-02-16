#!/bin/bash
set -e

IFACE=${IFACE:-eth0}
SERVER_IP=${SERVER_IP:-127.0.0.1}

info() { echo -e "\033[0;34mℹ $1\033[0m"; }
ok()   { echo -e "\033[0;32m✓ $1\033[0m"; }
warn() { echo -e "\033[1;33m⚠ $1\033[0m"; }
error() { echo -e "\033[0;31m✗ $1\033[0m"; }

if [ "$EUID" -ne 0 ]; then
  error "This script requires root privileges"
  warn "Run as: sudo $0 $*"
  exit 1
fi

if ! ip link show "$IFACE" &>/dev/null; then
    error "Network interface '$IFACE' not found"
    echo "Available interfaces:"
    ip -br link show
    exit 1
fi

reset_net() {
    info "Resetting network simulation on $IFACE"
    tc qdisc del dev "$IFACE" root 2>/dev/null || true
    tc filter del dev "$IFACE" 2>/dev/null || true
    ok "Network back to NORMAL (no simulation active)"
}

setup_filter() {
    tc qdisc add dev "$IFACE" root handle 1: prio
    tc qdisc add dev "$IFACE" parent 1:3 handle 30: netem "$@"
    tc filter add dev "$IFACE" protocol ip parent 1:0 prio 3 u32 \
        match ip dst "$SERVER_IP" flowid 1:3
    tc filter add dev "$IFACE" protocol ip parent 1:0 prio 3 u32 \
        match ip src "$SERVER_IP" flowid 1:3
}

poor_net() {
    reset_net
    info "Applying POOR network simulation (only to $SERVER_IP)"

    setup_filter delay 100ms 20ms distribution normal loss 1% rate 10mbit

    ok "POOR network active (target: $SERVER_IP)"
    echo "   RTT: 100ms ± 20ms"
    echo "   Packet loss: 1%"
    echo "   Bandwidth: 10 Mbps"
    echo "   Other traffic: UNAFFECTED"
}

worst_net() {
    reset_net
    info "Applying WORST network simulation (only to $SERVER_IP)"

    setup_filter delay 300ms 50ms distribution normal loss 3% rate 2mbit

    ok "WORST network active (target: $SERVER_IP)"
    echo "   RTT: 300ms ± 50ms"
    echo "   Packet loss: 3%"
    echo "   Bandwidth: 2 Mbps"
    echo "   Other traffic: UNAFFECTED"
}

mobile_3g() {
    reset_net
    info "Simulating 3G mobile network (only to $SERVER_IP)"

    setup_filter delay 100ms 50ms distribution normal loss 1% rate 384kbit

    ok "3G network active (target: $SERVER_IP)"
    echo "   RTT: 100ms ± 50ms"
    echo "   Packet loss: 1%"
    echo "   Bandwidth: 384 kbps"
    echo "   Other traffic: UNAFFECTED"
}

mobile_4g() {
    reset_net
    info "Simulating 4G/LTE mobile network (only to $SERVER_IP)"

    setup_filter delay 50ms 10ms distribution normal loss 0.1% rate 10mbit

    ok "4G network active (target: $SERVER_IP)"
    echo "   RTT: 50ms ± 10ms"
    echo "   Packet loss: 0.1%"
    echo "   Bandwidth: 10 Mbps"
    echo "   Other traffic: UNAFFECTED"
}

status() {
    info "Current traffic control config on $IFACE:"
    echo ""
    tc qdisc show dev "$IFACE"
    echo ""
    tc filter show dev "$IFACE"
    echo ""

    if tc qdisc show dev "$IFACE" | grep -q "netem"; then
        ok "Network simulation is ACTIVE (target: $SERVER_IP)"
    else
        info "Network simulation is NOT ACTIVE (normal operation)"
    fi
}

show_help() {
    cat << EOF
Network Simulation Tool for Load Testing (Targeted)
====================================================

Usage: sudo SERVER_IP=x.x.x.x ./net-sim.sh [COMMAND]

Commands:
  normal      Remove all simulation
  poor        Poor network (100ms RTT, 1% loss, 10Mbps)
  worst       Worst network (300ms RTT, 3% loss, 2Mbps)
  3g          3G mobile (100ms RTT, 1% loss, 384kbps)
  4g          4G/LTE mobile (50ms RTT, 0.1% loss, 10Mbps)
  status      Show current configuration
  help        Show this help message

Environment Variables:
  SERVER_IP   Target server IP (default: 127.0.0.1)
  IFACE       Network interface (default: eth0)

Examples:
  # Local server
  sudo SERVER_IP=127.0.0.1 ./net-sim.sh poor

  # Remote server
  sudo SERVER_IP=34.101.123.45 ./net-sim.sh worst

  # Custom interface
  sudo SERVER_IP=192.168.1.100 IFACE=wlan0 ./net-sim.sh 3g

  # Check status
  sudo ./net-sim.sh status

  # Reset
  sudo ./net-sim.sh normal

Testing Workflow:
  1. Run baseline test:
     EXP=baseline ./run-tests.sh scenario load

  2. Apply poor network to server:
     sudo SERVER_IP=127.0.0.1 ./net-sim.sh poor
     EXP=poor-net ./run-tests.sh scenario load

  3. Reset:
     sudo ./net-sim.sh normal

Note: Only traffic to/from SERVER_IP is affected!
      GCS uploads and other traffic remain normal.

EOF
}

case "${1:-help}" in
    normal)     reset_net ;;
    poor)       poor_net ;;
    worst)      worst_net ;;
    3g)         mobile_3g ;;
    4g)         mobile_4g ;;
    status)     status ;;
    help|--help|-h)
        show_help
        ;;
    *)
        error "Unknown command: $1"
        echo ""
        echo "Run './net-sim.sh help' for usage information"
        exit 1
        ;;
esac

echo ""
info "Done! Run './net-sim.sh status' to verify"