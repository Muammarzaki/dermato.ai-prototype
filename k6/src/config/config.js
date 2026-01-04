// src/config/config.js
const imgData = open('../../test-images/sample.jpg', 'b');

export const CONFIG = {
    // Server endpoints
    GRPC_ADDR: '127.0.0.1:8008',
    REST_ADDR: 'http://127.0.0.1:8088',

    // Image data
    IMAGE_DATA: imgData,

    // Performance settings
    CHUNK_SIZE: 64 * 1024, // 64KB chunks for optimal performance
    TIMEOUT: '30s',

    // Test metadata
    METADATA: {
        user_id: 'user-k6-test',
        image_type: 'image/jpeg',
        meta_tags: {
            source: 'k6-load-test',
            environment: 'testing'
        }
    }
};

// Balanced test scenarios
export const SCENARIOS = {
    // Smoke test - verify basic functionality
    smoke: {
        executor: 'constant-vus',
        vus: 1,
        duration: '1m',
        gracefulStop: '10s',
    },

    // Load test - test normal load conditions
    load: {
        executor: 'ramping-vus',
        startVUs: 0,
        stages: [
            { duration: '2m', target: 10 },  // Ramp up
            { duration: '5m', target: 10 },  // Stay at load
            { duration: '2m', target: 0 },   // Ramp down
        ],
        gracefulStop: '10s',
    },

    // Stress test - push beyond normal load
    stress: {
        executor: 'ramping-vus',
        startVUs: 0,
        stages: [
            { duration: '2m', target: 20 },  // Ramp to normal load
            { duration: '5m', target: 20 },  // Stay at normal load
            { duration: '2m', target: 40 },  // Ramp to stress load
            { duration: '5m', target: 40 },  // Stay at stress load
            { duration: '2m', target: 0 },   // Ramp down
        ],
        gracefulStop: '10s',
    },

    // Spike test - sudden load increase
    spike: {
        executor: 'ramping-vus',
        startVUs: 0,
        stages: [
            { duration: '1m', target: 10 },  // Normal load
            { duration: '30s', target: 50 }, // Spike!
            { duration: '3m', target: 50 },  // Maintain spike
            { duration: '1m', target: 10 },  // Recovery
            { duration: '1m', target: 0 },   // Ramp down
        ],
        gracefulStop: '10s',
    },

    // Soak test - sustained load over time
    soak: {
        executor: 'constant-vus',
        vus: 15,
        duration: '30m',
        gracefulStop: '10s',
    },
};

// Thresholds for performance validation
export const THRESHOLDS = {
    // HTTP/gRPC request duration
    'http_req_duration': ['p(95)<5000', 'p(99)<10000'],
    'grpc_req_duration': ['p(95)<5000', 'p(99)<10000'],

    // Request failure rate
    'http_req_failed': ['rate<0.05'],  // Less than 5% failures
    'checks': ['rate>0.95'],            // More than 95% success

    // Data transfer
    'data_received': ['rate>1000000'],  // At least 1MB/s
};