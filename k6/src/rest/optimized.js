import http from 'k6/http';
import {check} from 'k6';
import {TEST_CONFIG} from '../utils/config.js';

export const options = {
    scenarios: {
        rest_throughput: {
            executor: 'constant-vus',
            vus: 10,
            duration: '30s',
        },
    },
};

export default function () {
    const data = {
        user_id: TEST_CONFIG.METADATA.user_id,
        metadata: TEST_CONFIG.METADATA.meta_tags,
        file: http.file(TEST_CONFIG.IMAGE_DATA, 'sample.jpg', 'image/jpeg'),
    };


    const res = http.post(`${TEST_CONFIG.REST_ADDR}/analyze-skin`, data);

    check(res, {
        'status is 200': (r) => r.status === 200,
    });
}