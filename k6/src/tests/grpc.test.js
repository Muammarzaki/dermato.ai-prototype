// src/tests/grpc.test.js

import {CONFIG, SCENARIOS, THRESHOLDS} from '../config/config.js';
import {GrpcClient} from '../utils/grpc.utils.js';
import {group, sleep} from 'k6';
import {randomIntBetween} from 'https://jslib.k6.io/k6-utils/1.2.0/index.js';

const SCENARIO = __ENV.SCENARIO || 'load';

export const options = {
    scenarios: {
        grpc_test: {
            ...SCENARIOS[SCENARIO],
            exec: 'testGrpc',
        },
    },
    thresholds: {
        'grpc_req_duration': ['p(95)<5000', 'p(99)<10000'],
        'http_req_failed': ['rate<0.05'],
        'checks': ['rate>0.95'],
    },
};

const grpcClient = new GrpcClient(CONFIG.GRPC_ADDR, 'citra.proto');

export function setup() {
    console.log(`Running ${SCENARIO} scenario - gRPC only`);
    console.log(`gRPC Address: ${CONFIG.GRPC_ADDR}`);
    console.log(`Image size: ${CONFIG.IMAGE_DATA.byteLength} bytes`);
    console.log(`Chunk size: ${CONFIG.CHUNK_SIZE} bytes`);
    return {startTime: Date.now()};
}

export function testGrpc() {
    group('gRPC Skin Analysis', () => {
        grpcClient.connect(CONFIG.TIMEOUT);

        const result = grpcClient.analyzeSkin(
            CONFIG.IMAGE_DATA,
            CONFIG.METADATA,
            CONFIG.CHUNK_SIZE,
            () => grpcClient.close()
        );

        // Realistic delay between requests
        sleep(randomIntBetween(1, 3));
    });
}

export function teardown(data) {
    const duration = (Date.now() - data.startTime) / 1000;
    console.log(`Test completed in ${duration.toFixed(2)} seconds`);
}