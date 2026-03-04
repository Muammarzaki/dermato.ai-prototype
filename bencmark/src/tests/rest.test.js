import http from 'k6/http';
import { check, group } from 'k6';
import { sleep } from 'k6';
import { Trend, Counter, Rate, Gauge } from 'k6/metrics';
import { CONFIG, SCENARIOS, TEST_DATASET } from '../config/config.js';
import { calcBytes } from '../utils/bytes.js';

// ─── Scenario ────────────────────────────────────────────────────────────────
const SCENARIO = __ENV.SCENARIO || 'load';

export const options = {
    noConnectionReuse: false, // Biarkan HTTP/1.1 keep-alive aktif (default k6)
    scenarios: {
        execution_benchmark: SCENARIOS[SCENARIO],
    },
};

// ─── Metrics ─────────────────────────────────────────────────────────────────
const mDuration       = new Trend('rest_req_duration',       true);
const mWaiting        = new Trend('rest_req_waiting',        true);
const mSending        = new Trend('rest_req_sending',        true);
const mReceiving      = new Trend('rest_req_receiving',      true);
const mConnecting     = new Trend('rest_req_connecting',     true);
const mBlocked        = new Trend('rest_req_blocked',        true);
const mTLS            = new Trend('rest_req_tls_handshaking',true);
const mDataSent       = new Counter('rest_data_sent');
const mDataReceived   = new Counter('rest_data_received');
const mFailed         = new Counter('rest_req_failed');
const mSuccessRate    = new Rate('rest_req_success_rate');
const mActiveRequests = new Gauge('rest_active_requests');

// ─── State per VU ────────────────────────────────────────────────────────────
let active = 0;

// ─── Core function ───────────────────────────────────────────────────────────

function analyzeSkin(testCase) {
    let hasError = false;

    active++;
    mActiveRequests.add(active);

    const userIdSafe  = String(CONFIG.METADATA.user_id || '');
    const sha256Safe  = String(testCase.hash_hex || '');
    const metaString  = JSON.stringify(CONFIG.METADATA.meta_tags || {});

    const formData = {
        file:          http.file(testCase.data, testCase.filename, 'image/jpeg'),
        user_id:       userIdSafe,
        client_sha256: sha256Safe,
        metadata:      metaString,
    };

    const requestSize =
        testCase.data.byteLength +
        calcBytes(userIdSafe) +
        calcBytes(sha256Safe) +
        calcBytes(metaString);

    try {
        const res = http.post(
            `${CONFIG.REST_ADDR}/analyze-skin`,
            formData,
            {
                timeout: CONFIG.TIMEOUT,
                tags:    { protocol: 'rest', scenario: SCENARIO },
            }
        );

        // ── Timings ───────────────────────────────────────────────────────────
        const t = res.timings || {};
        mDuration.add(t.duration   ?? 0);
        if (t.blocked)         mBlocked.add(t.blocked);
        if (t.connecting)      mConnecting.add(t.connecting);
        if (t.tls_handshaking) mTLS.add(t.tls_handshaking);
        if (t.sending)         mSending.add(t.sending);
        if (t.waiting)         mWaiting.add(t.waiting);
        if (t.receiving)       mReceiving.add(t.receiving);

        mDataSent.add(requestSize);
        mDataReceived.add(calcBytes(res.body));

        // ── Status check ──────────────────────────────────────────────────────
        if (res.status < 200 || res.status >= 300) {
            hasError = true;
            mFailed.add(1);
            console.error(`REST Error: Status ${res.status} | Body: ${res.body}`);
        }

        // ── Parse body ────────────────────────────────────────────────────────
        let body = null;
        try {
            body = JSON.parse(res.body);
        } catch (e) {
            if (!hasError) console.error(`REST JSON Parse Error: ${e.message}`);
        }

        // ── Assertions ────────────────────────────────────────────────────────
        check(res, {
            'REST: status is 200': (r) => r.status === 200,
        });

        check(body, {
            'REST: response exists':    (r) => r !== null,
            'REST: has analysis_id':    (r) => typeof r?.analysis_id === 'string',
            'REST: has server_sha256':  (r) => typeof r?.server_sha256 === 'string',
            'REST: has results':        (r) => Array.isArray(r?.results) && r.results.length > 0,
            'REST: confidence valid':   (r) => r?.results?.[0]?.confidence >= 0 && r?.results?.[0]?.confidence <= 1,
            'REST: has label':          (r) => typeof r?.results?.[0]?.label === 'string',
            'REST: has description':    (r) => typeof r?.results?.[0]?.description === 'string',
            'REST: has recommendation': (r) => typeof r?.results?.[0]?.recommendation === 'string',
            'REST: has correct label':  (r) => r?.results?.[0]?.label === testCase.expected_label,
        });

        mSuccessRate.add(!hasError);
        return body;

    } catch (e) {
        mFailed.add(1);
        mSuccessRate.add(false);
        console.error(`REST Request Error: ${e.message}`);
        return null;

    } finally {
        active--;
        mActiveRequests.add(active);
    }
}

// ─── Lifecycle ───────────────────────────────────────────────────────────────

export function setup() {
    console.log(`Scenario  : ${SCENARIO}`);
    console.log(`REST Addr : ${CONFIG.REST_ADDR}`);
    console.log(`Images    : ${TEST_DATASET.length}`);
    return { startTime: Date.now() };
}

export default function () {
    group('REST Skin Analysis', () => {
        const testCase = TEST_DATASET[__ITER % TEST_DATASET.length];
        analyzeSkin(testCase);
    });

    sleep(Math.random() * 2 + 1);
}

export function teardown(data) {
    const duration = ((Date.now() - data.startTime) / 1000).toFixed(2);
    console.log(`Test selesai dalam ${duration} detik`);
}