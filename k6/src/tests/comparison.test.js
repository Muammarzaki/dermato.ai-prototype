// src/tests/comparison.test.js
// This test runs gRPC and REST side-by-side for direct comparison
import { CONFIG, THRESHOLDS } from '../config/config.js';
import { GrpcClient } from '../utils/grpc.utils.js';
import { RestClient } from '../utils/rest.utils.js';
import { group } from 'k6';

export const options = {
    scenarios: {
        // Same load for both protocols to compare performance
        grpc_comparison: {
            executor: 'constant-vus',
            vus: 10,
            duration: '5m',
            exec: 'testGrpc',
            tags: { protocol: 'grpc' },
        },
        rest_comparison: {
            executor: 'constant-vus',
            vus: 10,
            duration: '5m',
            exec: 'testRest',
            tags: { protocol: 'rest' },
        },
    },
    thresholds: THRESHOLDS,
};

const grpcClient = new GrpcClient(CONFIG.GRPC_ADDR, 'citra.proto');
const restClient = new RestClient(CONFIG.REST_ADDR);

export function testGrpc() {
    group('gRPC Protocol', () => {
        grpcClient.connect(CONFIG.TIMEOUT);
        grpcClient.analyzeSkin(
            CONFIG.IMAGE_DATA,
            CONFIG.METADATA,
            CONFIG.CHUNK_SIZE
        );
        grpcClient.close();
    });
}

export function testRest() {
    group('REST Protocol', () => {
        restClient.analyzeSkin(
            CONFIG.IMAGE_DATA,
            CONFIG.METADATA,
            CONFIG.TIMEOUT
        );
    });
}