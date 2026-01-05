// src/tests/warmup.test.js
// Warmup test untuk memastikan server siap sebelum test utama
import {CONFIG} from '../config/config.js';
import {GrpcClient} from '../utils/grpc.utils.js';
import {RestClient} from '../utils/rest.utils.js';
import {group, sleep} from 'k6';

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
    // No strict thresholds for warmup
    thresholds: {
        'checks': ['rate>0.80'],  // Lower threshold for warmup
    },
};

export function setup() {
    console.log('\n' + '━'.repeat(70));
    console.log('🔥 WARMUP TEST - Preparing Server for Load Testing');
    console.log('━'.repeat(70));
    console.log('Purpose:');
    console.log('  1. Initialize server connections');
    console.log('  2. Load necessary resources into memory');
    console.log('  3. Warm up caches and connection pools');
    console.log('  4. Establish baseline performance');
    console.log('━'.repeat(70));
    console.log(`Duration: 1 minute with 5 VUs per protocol`);
    console.log(`After warmup, wait 30 seconds before running actual test`);
    console.log('━'.repeat(70) + '\n');

    return {
        startTime: Date.now(),
    };
}

export function warmupGrpc() {
    group('gRPC Warmup', () => {
        const client = new GrpcClient(CONFIG.GRPC_ADDR, 'citra.proto');

        // Simple warmup - don't fail on errors
        try {
            const connected = client.connect(CONFIG.TIMEOUT);
            if (connected) {
                client.analyzeSkin(
                    CONFIG.IMAGE_DATA,
                    CONFIG.METADATA,
                    CONFIG.CHUNK_SIZE
                );
            }
        } catch (error) {
            // Ignore errors during warmup
            console.log(`Warmup gRPC request error (expected): ${error.message}`);
        } finally {
            client.close();
        }

        sleep(2);
    });
}

export function warmupRest() {
    group('REST Warmup', () => {
        const client = new RestClient(CONFIG.REST_ADDR);

        // Simple warmup - don't fail on errors
        try {
            client.analyzeSkin(
                CONFIG.IMAGE_DATA,
                CONFIG.METADATA,
                CONFIG.TIMEOUT
            );
        } catch (error) {
            // Ignore errors during warmup
            console.log(`Warmup REST request error (expected): ${error.message}`);
        }

        sleep(2);
    });
}

export function teardown(data) {
    const duration = (Date.now() - data.startTime) / 1000;

    console.log('\n' + '━'.repeat(70));
    console.log('✅ WARMUP COMPLETED');
    console.log('━'.repeat(70));
    console.log(`Duration: ${duration.toFixed(2)} seconds`);
    console.log('\n⏰ NEXT STEPS:');
    console.log('  1. Wait 30 seconds for server to stabilize');
    console.log('  2. Run your actual load test:');
    console.log('     k6 run -e SCENARIO=load src/tests/balanced.test.js');
    console.log('━'.repeat(70) + '\n');
}