import http from 'k6/http';
import {check} from "k6";

const imageBin = open('../test-images/sample.jpg', 'b');

export const options = {
    scenarios: {
        single_user: {
            executor: 'per-vu-iterations',
            vus: 1,
            iterations: 20,
            maxDuration: '5m',
        },
        concurrent_users: {
            executor: 'constant-vus',
            vus: 20,
            duration: '2m',
        },
        ramp_up: {
            executor: 'ramping-vus',
            stages: [
                {duration: '1m', target: 5},
                {duration: '1m', target: 10},
                {duration: '1m', target: 20},
                {duration: '1m', target: 30},
            ],
        },
    }
}

export default function () {
    const url = 'http://localhost:8088/analyze-skin';

    const data = {
        file: http.file(imageBin, 'sample.jpg', 'image/jpeg'),
        user_id: 'user-rest-test-k6',
    };

    const res = http.post(url, data, {
        timeout: '10s',
        tags: {protocol: 'rest'},
    });

    check(res, {
        'status is 200': (r) => r.status === 200,
    });

}
