// src/tests/warmup.test.js

import {CONFIG, TEST_DATASET} from '../config/config.js';
import {GrpcClient} from '../utils/grpc.utils.js';
import {RestClient} from '../utils/rest.utils.js';
import {group} from 'k6';
import {randomItem} from 'https://jslib.k6.io/k6-utils/1.2.0/index.js';

export const options = {
    scenarios: {
        grpc_warmup: {
            executor: 'constant-vus',
            vus: 5,
            duration: '1m',
            exec: 'warmupGrpc',
            tags: {test_type: 'warmup', protocol: 'grpc'},
        },
        rest_warmup: {
            executor: 'constant-vus',
            vus: 5,
            duration: '1m',
            exec: 'warmupRest',
            tags: {test_type: 'warmup', protocol: 'rest'},
        },
    },
    thresholds: {
        checks: ['rate>0.80'],
    },
};

export function setup() {
    console.log('Warmup started (gRPC + REST)');
    return {startTime: Date.now()};
}

const grpcClient = new GrpcClient(CONFIG.GRPC_ADDR, 'skin_analyzer.proto');
const restClient = new RestClient(CONFIG.REST_ADDR);

export function warmupGrpc() {
    grpcClient.connect(CONFIG.TIMEOUT);

    group('gRPC Warmup', () => {
        const randomTestCase = randomItem(TEST_DATASET);

        grpcClient.analyzeSkin(
            randomTestCase,
            CONFIG.METADATA,
            CONFIG.CHUNK_SIZE,
            () => grpcClient.close()
        );
    });
}

export function warmupRest() {
    group('REST Warmup', () => {
        const randomTestCase = randomItem(TEST_DATASET);
        restClient.analyzeSkin(
            randomTestCase,
            CONFIG.METADATA,
            CONFIG.TIMEOUT
        );
    });
}

export function teardown(data) {
    const duration = ((Date.now() - data.startTime) / 1000).toFixed(1);
    console.log(`✅ Warmup finished (${duration}s)`);
}
