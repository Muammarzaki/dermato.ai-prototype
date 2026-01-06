// src/tests/balanced.test.js
import {CONFIG, SCENARIOS, THRESHOLDS} from '../config/config.js';
import {GrpcClient} from '../utils/grpc.utils.js';
import {RestClient} from '../utils/rest.utils.js';
import {group, sleep} from 'k6';
import {randomIntBetween} from 'https://jslib.k6.io/k6-utils/1.2.0/index.js';

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
const grpcClient = new GrpcClient(CONFIG.GRPC_ADDR, 'citra.proto');
const restClient = new RestClient(CONFIG.REST_ADDR);

export function setup() {
    console.log(`Running ${SCENARIO} scenario with balanced gRPC and REST tests`);
    console.log(`Image size: ${CONFIG.IMAGE_DATA.byteLength} bytes`);
    console.log(`Chunk size: ${CONFIG.CHUNK_SIZE} bytes`);
    return {
        startTime: Date.now(),
    };
}

export function testGrpc() {
    group('gRPC Analysis', () => {
        grpcClient.connect(CONFIG.TIMEOUT);

        grpcClient.analyzeSkin(
            CONFIG.IMAGE_DATA,
            CONFIG.METADATA,
            CONFIG.CHUNK_SIZE,
            () => grpcClient.close()
        );
        // Random sleep between requests (0.5-2 seconds)
        sleep(randomIntBetween(0.5, 2));
    });
}

export function testRest() {
    group('REST Analysis', () => {
        restClient.analyzeSkin(
            CONFIG.IMAGE_DATA,
            CONFIG.METADATA,
            CONFIG.TIMEOUT
        );

        // Random sleep between requests (0.5-2 seconds)
        sleep(randomIntBetween(0.5, 2));
    });
}

export function teardown(data) {
    const duration = (Date.now() - data.startTime) / 1000;
    console.log(`Test completed in ${duration.toFixed(2)} seconds`);
}