// src/utils/rest.utils.js
import http from 'k6/http';
import {check} from 'k6';
import {Trend} from 'k6/metrics';

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
            tags: {protocol: 'rest'},
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

            console.log(`REST Response: ${JSON.stringify(response.body)}`);

            const body = JSON.parse(response.body);
            check(body, {
                'REST: response exists': (r) => r !== null,

                'REST: has analysis_id': (r) =>
                    typeof r.analysis_id === 'string' && r.analysis_id.length > 0,

                'REST: has results': (r) =>
                    Array.isArray(r.results) && r.results.length > 0,

                'REST: has label': (r) =>
                    typeof r.results[0].label === 'string',

                'REST: confidence valid': (r) =>
                    typeof r.results[0].confidence === 'number' &&
                    r.results[0].confidence >= 0 &&
                    r.results[0].confidence <= 1,
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
            console.error(`REST Request Error: ${error}`);
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