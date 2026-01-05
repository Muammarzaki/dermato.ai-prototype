// src/tests/rest.test.js

import { CONFIG, SCENARIOS, THRESHOLDS } from '../config/config.js';
import { RestClient } from '../utils/rest.utils.js';
import { group, sleep } from 'k6';
import { randomIntBetween } from 'https://jslib.k6.io/k6-utils/1.2.0/index.js';

const SCENARIO = __ENV.SCENARIO || 'load';

export const options = {
    scenarios: {
        rest_test: {
            ...SCENARIOS[SCENARIO],
            exec: 'testRest',
        },
    },
    thresholds: {
        'rest_req_duration': ['p(95)<5000', 'p(99)<10000'],
        'http_req_failed': ['rate<0.05'],
        'checks': ['rate>0.95'],
    },
};

const restClient = new RestClient(CONFIG.REST_ADDR);

export function setup() {
    console.log(`Running ${SCENARIO} scenario - REST only`);
    console.log(`REST Address: ${CONFIG.REST_ADDR}`);
    console.log(`Image size: ${CONFIG.IMAGE_DATA.byteLength} bytes`);
    return { startTime: Date.now() };
}

export function testRest() {
    group('REST Skin Analysis', () => {
        const result = restClient.analyzeSkin(
            CONFIG.IMAGE_DATA,
            CONFIG.METADATA,
            CONFIG.TIMEOUT
        );

        // Realistic delay between requests
        sleep(randomIntBetween(1, 3));
    });
}

export function teardown(data) {
    const duration = (Date.now() - data.startTime) / 1000;
    console.log(`Test completed in ${duration.toFixed(2)} seconds`);
}