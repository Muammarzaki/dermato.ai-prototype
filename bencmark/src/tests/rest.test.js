import {CONFIG, SCENARIOS, TEST_DATASET} from '../config/config.js';
import {RestClient} from '../utils/rest.utils.js';
import {group, sleep} from 'k6';
import {randomItem} from 'https://jslib.k6.io/k6-utils/1.2.0/index.js';

const SCENARIO = __ENV.SCENARIO || 'load';

export const options = {
    noConnectionReuse: true,
    scenarios: {
        execution_benchmark: SCENARIOS[SCENARIO],
    },
};

const restClient = new RestClient(CONFIG.REST_ADDR);

export function setup() {
    console.log(`Running ${SCENARIO} scenario - REST only`);
    console.log(`REST Address: ${CONFIG.REST_ADDR}`);
    console.log(`Loaded ${TEST_DATASET.length} test images`);
    console.log(`Chunk size: ${CONFIG.CHUNK_SIZE} bytes`);
    return {startTime: Date.now()};
}

export default function () {
    group('REST Skin Analysis', () => {
        const index = __ITER % TEST_DATASET.length;
        const testCase = TEST_DATASET[index];
        restClient.analyzeSkin(
            testCase,
            CONFIG.METADATA,
            CONFIG.TIMEOUT
        );
    });

    sleep(1);
}

export function teardown(data) {
    const duration = (Date.now() - data.startTime) / 1000;
    console.log(`Test completed in ${duration.toFixed(2)} seconds`);
}