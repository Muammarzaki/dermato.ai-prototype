// src/tests/comparison.test.js
// This test runs gRPC and REST side-by-side for direct comparison
import {CONFIG, TEST_DATASET, THRESHOLDS} from '../config/config.js';
import {GrpcClient} from '../utils/grpc.utils.js';
import {RestClient} from '../utils/rest.utils.js';
import {group} from 'k6';
import {randomItem} from 'https://jslib.k6.io/k6-utils/1.2.0/index.js';

export const options = {
    scenarios: {
        grpc_comparison: {
            executor: 'constant-vus',
            vus: 10,
            duration: '5m',
            exec: 'testGrpc',
            tags: {protocol: 'grpc'},
        },
        rest_comparison: {
            executor: 'constant-vus',
            vus: 10,
            duration: '5m',
            exec: 'testRest',
            tags: {protocol: 'rest'},
        },
    },
    thresholds: THRESHOLDS,
};

const grpcClient = new GrpcClient(CONFIG.GRPC_ADDR, 'skin_analyzer.proto');
const restClient = new RestClient(CONFIG.REST_ADDR);

export function testGrpc() {
    group('gRPC Protocol', () => {
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
    group('REST Protocol', () => {
        const randomTestCase = randomItem(TEST_DATASET);
        restClient.analyzeSkin(
            randomTestCase,
            CONFIG.METADATA,
            CONFIG.TIMEOUT
        );
    });
}