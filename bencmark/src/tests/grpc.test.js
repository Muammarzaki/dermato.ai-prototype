import grpc from 'k6/net/grpc';
import { check, group } from 'k6';
import { sleep } from 'k6';
import { b64encode } from 'k6/encoding';
import { Trend, Counter, Rate, Gauge } from 'k6/metrics';
import { CONFIG, SCENARIOS, TEST_DATASET } from '../config/config.js';
import { calcBytes } from '../utils/bytes.js';

// ─── Scenario ────────────────────────────────────────────────────────────────
const SCENARIO = __ENV.SCENARIO || 'load';

export const options = {
    scenarios: {
        execution_benchmark: SCENARIOS[SCENARIO],
    },
};

// ─── Metrics ─────────────────────────────────────────────────────────────────
const mDuration      = new Trend('grpc_req_duration',    true);
const mWaiting       = new Trend('grpc_req_waiting',     true);
const mSending       = new Trend('grpc_req_sending',     true);
const mConnecting    = new Trend('grpc_req_connecting',  true);
const mStreamDur     = new Trend('grpc_stream_duration', true);
const mDataSent      = new Counter('grpc_data_sent');
const mDataReceived  = new Counter('grpc_data_received');
const mFailed        = new Counter('grpc_req_failed');
const mSuccessRate   = new Rate('grpc_req_success_rate');
const mActiveStreams  = new Gauge('grpc_active_streams');

// ─── gRPC Client — init stage, satu koneksi per VU (HTTP/2 multiplexing) ─────
//
// grpc.Client() dipanggil di scope global sehingga objek dibuat sekali
// per VU di init stage, BUKAN tiap iterasi. connect() dipanggil sekali
// sebelum loop utama dimulai (di setup atau di awal default func dgn flag).
//
const client = new grpc.Client();
client.load(['../../../protobuf'], 'skin_analyzer.proto');

// Flag koneksi per VU — karena scope global di k6 di-reset tiap VU baru
let connected = false;

// ─── Helpers ─────────────────────────────────────────────────────────────────

function ensureConnected() {
    if (connected) return;

    const t0 = Date.now();
    client.connect(CONFIG.GRPC_ADDR, { plaintext: true, timeout: CONFIG.TIMEOUT });
    mConnecting.add(Date.now() - t0);
    connected = true;
}

function analyzeSkin(testCase) {
    const reqStart    = Date.now();
    const streamStart = Date.now();

    let bytesSent     = 0;
    let bytesReceived = 0;
    let sendingStart  = 0;
    let sendingEnd    = 0;
    let responseStart = 0;
    let hasError      = false;
    let responseReceived = false;
    let activeStreams  = 0;

    activeStreams++;
    mActiveStreams.add(activeStreams);

    const stream = new grpc.Stream(
        client,
        'skin_analyzer.SkinAnalysisService/AnalyzeSkin',
        { tags: { protocol: 'grpc', scenario: SCENARIO } }
    );

    stream.on('data', (res) => {
        if (!responseReceived) {
            responseStart    = Date.now();
            responseReceived = true;
        }

        bytesReceived += calcBytes(res);

        const ok = check(res, {
            'gRPC: response exists':      (r) => r !== null,
            'gRPC: has analysisId':       (r) => typeof r.analysisId === 'string',
            'gRPC: has results':          (r) => Array.isArray(r.results),
            'gRPC: confidence valid':     (r) => r.results?.[0]?.confidence >= 0 && r.results?.[0]?.confidence <= 1,
            'gRPC: has label':            (r) => typeof r?.results?.[0]?.label === 'string',
            'gRPC: has description':      (r) => typeof r?.results?.[0]?.description === 'string',
            'gRPC: has recommendation':   (r) => typeof r?.results?.[0]?.recommendation === 'string',
            'gRPC: has correct label':    (r) => r.results[0].label === testCase.expected_label,
        });

        if (!ok) hasError = true;
    });

    stream.on('error', (err) => {
        hasError = true;
        mFailed.add(1);
        console.error(`gRPC Error [${err.code}]: ${err.message}`);

        // Koneksi mungkin putus — reset flag agar VU reconnect di iterasi berikutnya
        connected = false;
    });

    stream.on('end', () => {
        mDuration.add(Date.now() - reqStart);
        mStreamDur.add(Date.now() - streamStart);

        if (sendingStart && sendingEnd) {
            mSending.add(sendingEnd - sendingStart);
        }
        if (responseStart && sendingEnd) {
            mWaiting.add(responseStart - sendingEnd);
        }

        mDataSent.add(bytesSent);
        mDataReceived.add(bytesReceived);
        mSuccessRate.add(!hasError);

        activeStreams--;
        mActiveStreams.add(activeStreams);
    });

    // ── Kirim data ────────────────────────────────────────────────────────────
    try {
        sendingStart = Date.now();

        // Frame 1: metadata
        const metaMsg = {
            info: {
                user_id:       CONFIG.METADATA.user_id,
                image_type:    CONFIG.METADATA.image_type,
                client_sha256: testCase.hash_base64,
                metadata:      CONFIG.METADATA.meta_tags,
            },
        };
        stream.write(metaMsg);
        bytesSent += calcBytes(metaMsg);

        // Frame 2..N: chunks gambar
        let offset = 0;
        while (offset < testCase.data.byteLength) {
            const end   = Math.min(offset + CONFIG.CHUNK_SIZE, testCase.data.byteLength);
            const chunk = testCase.data.slice(offset, end);

            stream.write({ chunk: b64encode(chunk) });
            bytesSent += end - offset;
            offset     = end;
        }

        sendingEnd = Date.now();
        stream.end();

    } catch (e) {
        hasError  = true;
        connected = false; // asumsi koneksi rusak
        mFailed.add(1);
        console.error(`gRPC Send Error: ${e.message}`);
        try { stream.end(); } catch (_) {}
    }
}

// ─── Lifecycle ───────────────────────────────────────────────────────────────

export function setup() {
    console.log(`Scenario  : ${SCENARIO}`);
    console.log(`gRPC Addr : ${CONFIG.GRPC_ADDR}`);
    console.log(`Images    : ${TEST_DATASET.length}`);
    console.log(`Chunk     : ${CONFIG.CHUNK_SIZE} bytes`);
    return { startTime: Date.now() };
}

export default function () {
    ensureConnected();

    group('gRPC Skin Analysis', () => {
        const testCase = TEST_DATASET[__ITER % TEST_DATASET.length];
        analyzeSkin(testCase);
    });

    sleep(Math.random() * 2 + 1);
}

export function teardown(data) {
    const duration = ((Date.now() - data.startTime) / 1000).toFixed(2);
    console.log(`Test selesai dalam ${duration} detik`);
}