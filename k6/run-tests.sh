#!/bin/bash

# K6 Load Testing Suite Runner
# Usage: ./run-tests.sh [test-type] [scenario]
# Example: ./run-tests.sh comparison
#          ./run-tests.sh balanced load
#          ./run-tests.sh full-suite

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
RESULTS_DIR="./results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create results directory
mkdir -p "$RESULTS_DIR"

# Helper functions
print_header() {
    echo -e "${BLUE}╔$(printf '═%.0s' {1..78})╗${NC}"
    printf "${BLUE}║${NC} %-76s ${BLUE}║${NC}\n" "$1"
    echo -e "${BLUE}╚$(printf '═%.0s' {1..78})╝${NC}"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

# Check if k6 is installed
check_k6() {
    if ! command -v k6 &> /dev/null; then
        print_error "k6 is not installed"
        echo "Install from: https://k6.io/docs/getting-started/installation/"
        exit 1
    fi
    print_success "k6 is installed: $(k6 version)"
}

# Check if servers are running
check_servers() {
    print_header "Checking Server Availability"

    # Check gRPC server
    if nc -z 127.0.0.1 8008 2>/dev/null; then
        print_success "gRPC server is running on port 8008"
    else
        print_error "gRPC server is not running on port 8008"
        exit 1
    fi

    # Check REST server
    if nc -z 127.0.0.1 8088 2>/dev/null; then
        print_success "REST server is running on port 8088"
    else
        print_error "REST server is not running on port 8088"
        exit 1
    fi

    echo ""
}

# Run warmup
run_warmup() {
    print_header "Running Warmup Test"
    print_info "Purpose: Prepare server for load testing"
    print_info "Duration: 1 minute"
    echo ""

    k6 run src/tests/warmup.test.js \
        --out json="$RESULTS_DIR/warmup_${TIMESTAMP}.json" \
        --summary-export="$RESULTS_DIR/warmup_${TIMESTAMP}_summary.json"

    print_success "Warmup completed"
    print_info "Waiting 30 seconds for server stabilization..."
    sleep 30
    echo ""
}

# Run smoke test
run_smoke() {
    print_header "Running Smoke Test"
    print_info "Scenario: Single user, basic functionality check"
    print_info "Duration: 1 minute"
    echo ""

    k6 run -e SCENARIO=smoke src/tests/balanced.test.js \
        --out json="$RESULTS_DIR/smoke_${TIMESTAMP}.json" \
        --summary-export="$RESULTS_DIR/smoke_${TIMESTAMP}_summary.json"

    print_success "Smoke test completed"
    echo ""
}

# Run balanced test
run_balanced() {
    local scenario=${1:-load}

    print_header "Running Balanced Test - ${scenario^^}"
    print_info "Testing both gRPC and REST with ${scenario} scenario"
    echo ""

    k6 run -e SCENARIO="$scenario" src/tests/balanced.test.js \
        --out json="$RESULTS_DIR/balanced_${scenario}_${TIMESTAMP}.json" \
        --summary-export="$RESULTS_DIR/balanced_${scenario}_${TIMESTAMP}_summary.json"

    print_success "Balanced test completed"
    echo ""
}

# Run comparison test
run_comparison() {
    print_header "Running Comparison Test"
    print_info "Direct side-by-side comparison with identical load"
    print_info "Duration: 5 minutes, 10 VUs each"
    echo ""

    k6 run src/tests/comparison.test.js \
        --out json="$RESULTS_DIR/comparison_${TIMESTAMP}.json" \
        --summary-export="$RESULTS_DIR/comparison_${TIMESTAMP}_summary.json"

    print_success "Comparison test completed"
    echo ""
}

# Run detailed analysis
run_detailed() {
    print_header "Running Detailed Analysis"
    print_info "Detailed timing breakdown with logging"
    print_info "Duration: 2 minutes, 5 VUs each"
    echo ""

    k6 run src/tests/detailed.analysis.test.js \
        --out json="$RESULTS_DIR/detailed_${TIMESTAMP}.json" \
        --summary-export="$RESULTS_DIR/detailed_${TIMESTAMP}_summary.json"

    print_success "Detailed analysis completed"
    echo ""
}

# Run full test suite
run_full_suite() {
    print_header "Running Full Test Suite"
    echo ""

    check_servers
    run_warmup
    run_smoke

    print_info "Running multiple scenarios..."
    echo ""

    run_balanced "load"
    sleep 30

    run_balanced "stress"
    sleep 30

    run_comparison
    sleep 30

    run_detailed

    print_header "Full Test Suite Completed"
    print_success "All tests finished successfully"
    print_info "Results saved in: $RESULTS_DIR"
    echo ""
}

# Generate comparison report
generate_report() {
    print_header "Generating Comparison Report"

    local comparison_file="$RESULTS_DIR/comparison_${TIMESTAMP}_summary.json"

    if [ -f "$comparison_file" ]; then
        echo ""
        print_info "Backend Processing Time Comparison:"
        echo ""

        # Extract key metrics using jq (if available)
        if command -v jq &> /dev/null; then
            echo "gRPC Backend Processing:"
            jq '.metrics.grpc_backend_processing_time' "$comparison_file" 2>/dev/null || echo "N/A"
            echo ""
            echo "REST Backend Processing:"
            jq '.metrics.rest_backend_processing_time' "$comparison_file" 2>/dev/null || echo "N/A"
        else
            print_warning "Install 'jq' for detailed JSON parsing"
            print_info "View results manually: cat $comparison_file"
        fi
    else
        print_warning "Comparison results not found"
    fi

    echo ""
}

# Main script logic
main() {
    local test_type=${1:-help}
    local scenario=${2:-load}

    echo ""
    print_header "K6 Load Testing Suite"
    echo ""

    check_k6

    case "$test_type" in
        warmup)
            check_servers
            run_warmup
            ;;
        smoke)
            check_servers
            run_smoke
            ;;
        balanced)
            check_servers
            run_balanced "$scenario"
            ;;
        comparison)
            check_servers
            run_warmup
            run_comparison
            generate_report
            ;;
        detailed)
            check_servers
            run_detailed
            ;;
        full-suite)
            run_full_suite
            generate_report
            ;;
        help|*)
            print_header "Usage Instructions"
            echo ""
            echo "Usage: ./run-tests.sh [test-type] [scenario]"
            echo ""
            echo "Test Types:"
            echo "  warmup              - Run warmup test only"
            echo "  smoke               - Run smoke test (basic functionality)"
            echo "  balanced [scenario] - Run balanced test with specific scenario"
            echo "  comparison          - Run direct comparison test"
            echo "  detailed            - Run detailed analysis test"
            echo "  full-suite          - Run complete test suite"
            echo "  help                - Show this help message"
            echo ""
            echo "Scenarios (for balanced test):"
            echo "  smoke  - 1 VU for 1 minute"
            echo "  load   - Ramp to 10 VUs (default)"
            echo "  stress - Ramp to 40 VUs"
            echo "  spike  - Sudden spike to 50 VUs"
            echo "  soak   - 15 VUs for 30 minutes"
            echo ""
            echo "Examples:"
            echo "  ./run-tests.sh smoke"
            echo "  ./run-tests.sh balanced load"
            echo "  ./run-tests.sh balanced stress"
            echo "  ./run-tests.sh comparison"
            echo "  ./run-tests.sh full-suite"
            echo ""
            exit 0
            ;;
    esac

    print_header "Test Completed Successfully"
    print_success "Results directory: $RESULTS_DIR"
    echo ""
}

# Run main function
main "$@"