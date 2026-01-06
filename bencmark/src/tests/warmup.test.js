// src/tests/warmup.test.js

import { CONFIG } from '../config/config.js';
import { GrpcClient } from '../utils/grpc.utils.js';
import { RestClient } from '../utils/rest.utils.js';
import { group, sleep } from 'k6';

export const options = {
    scenarios: {
        grpc_warmup: {
            executor: 'constant-vus',
            vus: 5,
            duration: '1m',
            exec: 'warmupGrpc',
            tags: { test_type: 'warmup', protocol: 'grpc' },
        },
        rest_warmup: {
            executor: 'constant-vus',
            vus: 5,
            duration: '1m',
            exec: 'warmupRest',
            tags: { test_type: 'warmup', protocol: 'rest' },
        },
    },
    thresholds: {
        checks: ['rate>0.80'],
    },
};

export function setup() {
    console.log('🔥 Warmup started (gRPC + REST)');
    return { startTime: Date.now() };
}

const grpcClient = new GrpcClient(CONFIG.GRPC_ADDR, 'citra.proto');
const restClient = new RestClient(CONFIG.REST_ADDR);

export function warmupGrpc() {
    group('gRPC Warmup', () => {
        try {
            if (grpcClient.connect(CONFIG.TIMEOUT)) {
                grpcClient.analyzeSkin(
                    CONFIG.IMAGE_DATA,
                    CONFIG.METADATA,
                    CONFIG.CHUNK_SIZE,
                    () => grpcClient.close()
                );
            }
        } catch (_) {
            // silent – warmup only
        }

        sleep(2);
    });
}

export function warmupRest() {
    group('REST Warmup', () => {
        try {
            restClient.analyzeSkin(
                CONFIG.IMAGE_DATA,
                CONFIG.METADATA,
                CONFIG.TIMEOUT
            );
        } catch (_) {
            // silent – warmup only
        }

        sleep(2);
    });
}

export function teardown(data) {
    const duration = ((Date.now() - data.startTime) / 1000).toFixed(1);
    console.log(`✅ Warmup finished (${duration}s)`);
}
