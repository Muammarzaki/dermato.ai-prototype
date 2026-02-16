import http from 'k6/http';
import {check} from 'k6';
import {Trend, Counter, Rate, Gauge} from 'k6/metrics';
import {calcBytes} from './bytes.js';

const restReqDuration = new Trend('rest_req_duration', true);
const restReqWaiting = new Trend('rest_req_waiting', true);
const restReqSending = new Trend('rest_req_sending', true);
const restReqReceiving = new Trend('rest_req_receiving', true);
const restReqConnecting = new Trend('rest_req_connecting', true);
const restReqBlocked = new Trend('rest_req_blocked', true);
const restReqTLS = new Trend('rest_req_tls_handshaking', true);

const restDataSent = new Counter('rest_data_sent');
const restDataReceived = new Counter('rest_data_received');

const restReqFailed = new Counter('rest_req_failed');
const restReqSucceeded = new Rate('rest_req_success_rate');
const restActiveRequests = new Gauge('rest_active_requests');

export class RestClient {
    constructor(baseUrl) {
        this.baseUrl = baseUrl;
        this.active = 0;
    }

    analyzeSkin(testCase, metadata, timeout = '30s') {
        const start = Date.now();
        let hasError = false;

        this.active++;
        restActiveRequests.add(this.active);

        const userIdSafe = String(metadata.user_id || "");
        const sha256_HEX_Safe = String(testCase.hash_hex || "");
        const metaString = JSON.stringify(metadata.meta_tags || {});

        const formData = {
            file: http.file(testCase.data, 'sample.jpg', 'image/jpeg'),
            user_id: userIdSafe,
            client_sha256: sha256_HEX_Safe,
            metadata: metaString,
        };

        const requestSize =
            calcBytes(testCase.buffer) +
            calcBytes(userIdSafe) +
            calcBytes(sha256_HEX_Safe) +
            calcBytes(metaString);

        try {
            const res = http.post(
                `${this.baseUrl}/analyze-skin`,
                formData,
                {
                    timeout: timeout,
                    tags: {protocol: 'rest'},
                }
            );

            restReqDuration.add(Date.now() - start);

            const t = res.timings || {};
            if (t.blocked) restReqBlocked.add(t.blocked);
            if (t.connecting) restReqConnecting.add(t.connecting);
            if (t.tls_handshaking) restReqTLS.add(t.tls_handshaking);
            if (t.sending) restReqSending.add(t.sending);
            if (t.waiting) restReqWaiting.add(t.waiting);
            if (t.receiving) restReqReceiving.add(t.receiving);

            restDataSent.add(requestSize);
            restDataReceived.add(calcBytes(res.body));

            const ok = res.status >= 200 && res.status < 300;
            if (!ok) {
                hasError = true;
                restReqFailed.add(1);
                console.error(`REST Error: Status ${res.status}, Body: ${res.body}`);
            }

            let body = null;
            try {
                body = JSON.parse(res.body);
            } catch (e) {
                if (!ok) {
                    console.error(`REST JSON Parse Error: ${e.message}`);
                }
            }

            check(res, {
                'REST: status is 200': (r) => r.status === 200,
            });

            check(body, {
                'REST: response exists': (r) => r !== null,
                'REST: has analysis_id': (r) => typeof r?.analysis_id === 'string',
                'REST: has server_sha256': (r) => typeof r?.server_sha256 === 'string',
                'REST: has results': (r) => Array.isArray(r?.results) && r.results.length > 0,
                'REST: confidence valid': (r) =>
                    r?.results?.[0]?.confidence >= 0 &&
                    r?.results?.[0]?.confidence <= 1,
                'REST: has label': (r) => typeof r?.results?.[0]?.label === 'string',
                'REST: has description': (r) => typeof r?.results?.[0]?.description === 'string',
                'REST: has recommendation': (r) => typeof r?.results?.[0]?.recommendation === 'string',
                'REST: has correct label': (r) => r.results[0].label === testCase.expected_label,
            });

            restReqSucceeded.add(!hasError);
            return body;

        } catch (e) {
            hasError = true;
            restReqFailed.add(1);
            restReqSucceeded.add(false);
            console.error(`REST Request Error: ${e.message}`);
            return null;

        } finally {
            this.active--;
            restActiveRequests.add(this.active);
        }
    }
}