// src/utils/rest.utils.js
import http from 'k6/http';
import {check} from 'k6';
import {Trend, Counter, Rate, Gauge} from 'k6/metrics';
import {calcBytes} from './bytes.js';

// ================= METRICS =================
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

// ================= CLIENT =================
export class RestClient {
    constructor(baseUrl) {
        this.baseUrl = baseUrl;
        this.active = 0;
    }

    analyzeSkin(imageData, metadata, timeout = '30s') {
        const start = Date.now();
        let hasError = false;

        this.active++;
        restActiveRequests.add(this.active);

        const formData = {
            file: http.file(imageData, 'sample.jpg', 'image/jpeg'),
            user_id: metadata.user_id,
            metadata: JSON.stringify(metadata.meta_tags),
        };

        const requestSize =
            calcBytes(imageData) +
            calcBytes(metadata);

        try {
            const res = http.post(
                `${this.baseUrl}/analyze-skin`,
                formData,
                {
                    timeout, tags: {protocol: 'rest'}, headers: {
                        'Connection': 'close'
                    }
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
            }

            let body = null;
            try {
                body = JSON.parse(res.body);
            } catch {
                hasError = true;
            }

            check(body, {
                'REST: response exists': (r) => r !== null,
                'REST: has analysis_id': (r) => typeof r.analysis_id === 'string',
                'REST: has results': (r) => Array.isArray(r.results),
            });

            restReqSucceeded.add(!hasError);
            return body;

        } catch (e) {
            hasError = true;
            restReqFailed.add(1);
            restReqSucceeded.add(false);
            console.error(`REST Error: ${e}`);
            return null;

        } finally {
            this.active--;
            restActiveRequests.add(this.active);
        }
    }
}
