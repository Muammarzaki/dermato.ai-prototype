// src/utils/rest.utils.js
import http from 'k6/http';
import { check } from 'k6';
import { Trend } from 'k6/metrics';

// Custom metrics
const restDuration = new Trend('rest_req_duration', true);

export class RestClient {
    constructor(baseUrl) {
        this.baseUrl = baseUrl;
    }

    analyzeSkin(imageData, metadata, timeout = '30s') {
        const startTime = Date.now();

        const formData = {
            file: http.file(imageData, 'sample.jpg', 'image/jpeg'),
            user_id: metadata.user_id,
            metadata: JSON.stringify(metadata.meta_tags),
        };

        const params = {
            timeout: timeout,
            tags: { protocol: 'rest' },
            headers: {
                'Accept': 'application/json',
            },
        };

        try {
            const response = http.post(
                `${this.baseUrl}/analyze-skin`,
                formData,
                params
            );

            const duration = Date.now() - startTime;
            restDuration.add(duration);

            check(response, {
                'REST: status is 200': (r) => r.status === 200,
                'REST: status is 2xx': (r) => r.status >= 200 && r.status < 300,
                'REST: has response body': (r) => r.body && r.body.length > 0,
                'REST: response time < 10s': (r) => r.timings.duration < 10000,
            });

            let result = null;
            if (response.status === 200) {
                try {
                    result = JSON.parse(response.body);
                } catch (e) {
                    console.error(`Failed to parse response: ${e.message}`);
                }
            }

            return {
                success: response.status === 200,
                status: response.status,
                data: result,
                duration: duration,
            };

        } catch (error) {
            console.error(`REST Request Error: ${error.message}`);
            check(null, {
                'REST: request successful': () => false,
            });
            return {
                success: false,
                error: error.message,
            };
        }
    }
}