// src/tests/detailed.analysis.test.js
// Test untuk analisis mendalam dengan logging detail
import {CONFIG} from '../config/config.js';
import {GrpcClient} from '../utils/grpc.utils.js';
import {RestClient} from '../utils/rest.utils.js';
import {group, sleep} from 'k6';

export const options = {
    scenarios: {
        grpc_detailed: {
            executor: 'constant-vus',
            vus: 5,
            duration: '2m',
            exec: 'testGrpcDetailed',
            tags: {protocol: 'grpc'},
        },
        rest_detailed: {
            executor: 'constant-vus',
            vus: 5,
            duration: '2m',
            exec: 'testRestDetailed',
            tags: {protocol: 'rest'},
        },
    },
    thresholds: {
        'grpc_backend_processing_time': ['p(50)<5000', 'p(95)<8000'],
        'rest_backend_processing_time': ['p(50)<5000', 'p(95)<8000'],
        'checks': ['rate>0.95'],
    },
};

export function setup() {
    console.log('\n' + '='.repeat(80));
    console.log('DETAILED ANALYSIS TEST - Request Lifecycle Breakdown');
    console.log('='.repeat(80));
    console.log(`This test logs detailed timing for every 10th request`);
    console.log(`to help analyze and debug performance differences.`);
    console.log('='.repeat(80) + '\n');

    return {
        startTime: Date.now(),
        grpcRequestCount: 0,
        restRequestCount: 0,
    };
}

export function testGrpcDetailed(data) {
    const requestNum = __ITER;
    const shouldLog = requestNum % 10 === 0; // Log every 10th request

    group('gRPC Detailed Analysis', () => {
        const timestamps = {
            start: Date.now(),
            connected: null,
            streamStarted: null,
            dataSent: null,
            responseReceived: null,
            closed: null,
        };

        const client = new GrpcClient(CONFIG.GRPC_ADDR, 'citra.proto');

        // Step 1: Connect
        const connectStart = Date.now();
        const connected = client.connect(CONFIG.TIMEOUT);
        timestamps.connected = Date.now();
        const connectionTime = timestamps.connected - connectStart;

        if (!connected) {
            console.error(`[gRPC-${requestNum}] Connection failed`);
            return;
        }

        timestamps.streamStarted = Date.now();

        // Step 2-6: Stream and process
        const result = client.analyzeSkin(
            CONFIG.IMAGE_DATA,
            CONFIG.METADATA,
            CONFIG.CHUNK_SIZE,
            () => {
                timestamps.responseReceived = Date.now();
                client.close()
                timestamps.closed = Date.now();
            }
        );

        // Calculate timings
        const timings = {
            connection: connectionTime,
            stream: timestamps.responseReceived - timestamps.streamStarted,
            total: timestamps.closed - timestamps.start,
        };

        // Log details for analysis
        if (shouldLog && result && result.success) {
            console.log(`\n[gRPC Request #${requestNum}] Lifecycle Breakdown:`);
            console.log(`  ├─ Connection Time:    ${timings.connection.toFixed(2)}ms`);
            console.log(`  ├─ Stream Time:        ${timings.stream.toFixed(2)}ms`);
            console.log(`  │   ├─ Data Transfer:  ~${(timings.stream * 0.3).toFixed(2)}ms (estimated)`);
            console.log(`  │   └─ Backend Proc:   ~${(timings.stream * 0.7).toFixed(2)}ms (estimated)`);
            console.log(`  └─ Total Time:         ${timings.total.toFixed(2)}ms`);
            console.log(`  Chunks Sent: ${result.metrics.chunksCount}`);
            console.log(`  Bytes Sent:  ${result.metrics.totalBytesSent}`);
        }

        sleep(1);
    });
}

export function testRestDetailed(data) {
    const requestNum = __ITER;
    const shouldLog = requestNum % 10 === 0; // Log every 10th request

    group('REST Detailed Analysis', () => {
        const timestamps = {
            start: Date.now(),
            requestSent: null,
            responseReceived: null,
        };

        const client = new RestClient(CONFIG.REST_ADDR);

        timestamps.requestSent = Date.now();

        const result = client.analyzeSkin(
            CONFIG.IMAGE_DATA,
            CONFIG.METADATA,
            CONFIG.TIMEOUT
        );

        timestamps.responseReceived = Date.now();

        // Log details for analysis
        if (shouldLog && result && result.success) {
            const t = result.timings;

            console.log(`\n[REST Request #${requestNum}] Lifecycle Breakdown:`);
            console.log(`  ├─ Connection Phase:     ${(t.connecting + (t.tls_handshaking || 0)).toFixed(2)}ms`);
            console.log(`  │   ├─ DNS/Blocked:      ${t.blocked.toFixed(2)}ms`);
            console.log(`  │   ├─ TCP Connect:      ${t.connecting.toFixed(2)}ms`);
            console.log(`  │   └─ TLS Handshake:    ${(t.tls_handshaking || 0).toFixed(2)}ms`);
            console.log(`  ├─ Upload Time:          ${t.sending.toFixed(2)}ms`);
            console.log(`  ├─ Backend Processing:   ${t.waiting.toFixed(2)}ms ← KEY METRIC`);
            console.log(`  ├─ Download Time:        ${t.receiving.toFixed(2)}ms`);
            console.log(`  └─ Total Time:           ${t.duration.toFixed(2)}ms`);
            console.log(`  Data Sent:     ${result.metrics.dataSent} bytes`);
            console.log(`  Data Received: ${result.metrics.dataReceived} bytes`);
            console.log(`  HTTP Status:   ${result.status}`);
        } else if (shouldLog && !result.success) {
            console.error(`\n[REST Request #${requestNum}] FAILED:`);
            console.error(`  Status: ${result.status || 'N/A'}`);
            console.error(`  Error:  ${result.error || 'Unknown'}`);
            if (result.metrics) {
                console.error(`  Total Time: ${result.metrics.totalDuration}ms`);
            }
        }

        sleep(1);
    });
}

export function teardown(data) {
    const totalDuration = (Date.now() - data.startTime) / 1000;

    console.log('\n' + '='.repeat(80));
    console.log('DETAILED ANALYSIS COMPLETED');
    console.log('='.repeat(80));
    console.log(`Total Test Duration: ${totalDuration.toFixed(2)} seconds`);
    console.log('\nCheck the summary metrics above for:');
    console.log('  1. Backend Processing Time comparison (should be similar!)');
    console.log('  2. Connection/Transfer overhead differences');
    console.log('  3. Overall performance patterns');
    console.log('\nUse this data to understand WHERE differences come from.');
    console.log('='.repeat(80) + '\n');
}