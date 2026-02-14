// src/tests/balanced.test.js
import {CONFIG, SCENARIOS, TEST_DATASET, THRESHOLDS} from '../config/config.js';
import {GrpcClient} from '../utils/grpc.utils.js';
import {RestClient} from '../utils/rest.utils.js';
import {group} from 'k6';
import {randomItem} from 'https://jslib.k6.io/k6-utils/1.2.0/index.js';

// Choose which scenario to run via environment variable
// Example: k6 run -e SCENARIO=load src/tests/balanced.test.js
const SCENARIO = __ENV.SCENARIO || 'load';

export const options = {
    scenarios: {
        grpc_test: {
            ...SCENARIOS[SCENARIO],
            exec: 'testGrpc',
            startTime: '0s',
        },
        rest_test: {
            ...SCENARIOS[SCENARIO],
            exec: 'testRest',
            startTime: '0s',
        },
    },
    thresholds: THRESHOLDS,
};

// Initialize clients
const grpcClient = new GrpcClient(CONFIG.GRPC_ADDR, 'skin_analyzer.proto');
const restClient = new RestClient(CONFIG.REST_ADDR);

export function setup() {
    console.log(`Running ${SCENARIO} scenario with balanced gRPC and REST tests`);
    console.log(`Loaded ${TEST_DATASET.length} test images.`);
    console.log(`Chunk size: ${CONFIG.CHUNK_SIZE} bytes`);
    return {
        startTime: Date.now(),
    };
}

export function testGrpc() {
    group('gRPC Analysis', () => {
        grpcClient.connect(CONFIG.TIMEOUT);

        const randomTestCase = randomItem(TEST_DATASET);

        grpcClient.analyzeSkin(
            randomTestCase,
            CONFIG.METADATA,
            CONFIG.CHUNK_SIZE,
            () => grpcClient.close()
        );
    });
}

export function testRest() {
    group('REST Analysis', () => {
        const randomTestCase = randomItem(TEST_DATASET);
        restClient.analyzeSkin(
            randomTestCase,
            CONFIG.METADATA,
            CONFIG.TIMEOUT
        );
    });
}

export function teardown(data) {
    const duration = (Date.now() - data.startTime) / 1000;
    console.log(`Test completed in ${duration.toFixed(2)} seconds`);
}