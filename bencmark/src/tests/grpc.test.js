import {CONFIG, SCENARIOS, TEST_DATASET} from '../config/config.js';
import {GrpcClient} from '../utils/grpc.utils.js';
import {group, sleep} from 'k6';
import {randomItem} from 'https://jslib.k6.io/k6-utils/1.2.0/index.js';

const SCENARIO = __ENV.SCENARIO || 'load';

export const options = {
    ...SCENARIOS[SCENARIO],
};

const grpcClient = new GrpcClient(CONFIG.GRPC_ADDR, 'skin_analyzer.proto');

export function setup() {
    console.log(`Running ${SCENARIO} scenario - gRPC only`);
    console.log(`gRPC Address: ${CONFIG.GRPC_ADDR}`);
    console.log(`Loaded ${TEST_DATASET.length} test images`);
    console.log(`Chunk size: ${CONFIG.CHUNK_SIZE} bytes`);
    return {startTime: Date.now()};
}

export default function () {
    grpcClient.connect(CONFIG.TIMEOUT);

    group('gRPC Skin Analysis', () => {
        const randomTestCase = randomItem(TEST_DATASET);

        grpcClient.analyzeSkin(
            randomTestCase,
            CONFIG.METADATA,
            CONFIG.CHUNK_SIZE
        );
    });

    sleep(1);
}

export function teardown(data) {
    const duration = (Date.now() - data.startTime) / 1000;
    console.log(`Test completed in ${duration.toFixed(2)} seconds`);

    grpcClient.close();
}