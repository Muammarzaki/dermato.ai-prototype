import crypto from "k6/crypto";
// src/config/config.js
// const imgAcne = open('../../test-images/acne-closed-comedo-27.jpg', 'b');
// const imgBCC = open('../../test-images/basal-cell-carcinoma-lid-3.jpg', 'b');
// const imgBullous = open('../../test-images/benign-familial-chronic-pemphigus-11.jpg', 'b');
// const imgCacarAir = open('../../test-images/13_VI-chickenpox (22).jpg', 'b');
// const imgEczema = open('../../test-images/03EczemaExcoriated.jpg', 'b');
// const imgMelanoma = open('../../test-images/malignant-melanoma-17.jpg', 'b');
// const imgNevi = open('../../test-images/atypical-nevi-44.jpg', 'b');
// const imgUrticaria = open('../../test-images/dermagraphism-27.jpg', 'b');
const imgTaiLalat1mb = open('../../test-images/tahi_lalat_1.5mb.jpg', 'b');
const imgTaiLalat2mb = open('../../test-images/tahi_lalat_2mb.jpg', 'b');
const imgTaiLalat2_5mb = open('../../test-images/tahi_lalat_2.5mb.jpg', 'b');
const imgTaiLalat3_7mb = open('../../test-images/tahi_lalat_3.7mb.jpg', 'b');
const imgTaiLalat4mb = open('../../test-images/tahi_lalat_4mb.jpg', 'b');
// const imgTaiLalat22mb = open('../../test-images/tahi_lalat_22mb.jpg', 'b');
// const imgTaiLalat18mb = open('../../test-images/tahi_lalat_18mb.jpg', 'b');

function createTestCase(filename, expectedLabel, buffer) {
    return {
        filename: filename,
        expected_label: expectedLabel,
        data: buffer,
        hash_hex: crypto.sha256(buffer, 'hex'),      // Dipakai REST API
        hash_base64: crypto.sha256(buffer, 'base64') // Dipakai gRPC API
    };
}

export const TEST_DATASET = [
    // createTestCase('acne-closed-comedo-27.jpg', 'Acne', imgAcne),
    // createTestCase('basal-cell-carcinoma-lid-3.jpg', 'Basal Cell Carcinoma', imgBCC),
    // createTestCase('benign-familial-chronic-pemphigus-11.jpg', 'Bullous Disease', imgBullous),
    // createTestCase('13_VI-chickenpox (22).jpg', 'Cacar Air', imgCacarAir),
    // createTestCase('03EczemaExcoriated.jpg', 'Eczema', imgEczema),
    // createTestCase('malignant-melanoma-17.jpg', 'Skin Cancer', imgMelanoma),
    // createTestCase('atypical-nevi-44.jpg', 'Skin Cancer', imgNevi),
    // createTestCase('dermagraphism-27.jpg', 'Bullous Disease', imgUrticaria),
    createTestCase("tahi_lalat_1.5mb.jpg", "Eczema", imgTaiLalat1mb),
    createTestCase("tahi_lalat_2mb.jpg", "Cacar Air", imgTaiLalat2mb),
    createTestCase("tahi_lalat_2_5mb.jpg", "Cacar Air", imgTaiLalat2_5mb),
    createTestCase("tahi_lalat_3_7mb.jpg", "Cacar Air", imgTaiLalat3_7mb),
    createTestCase("tahi_lalat_4mb.jpg", "Cacar Air", imgTaiLalat4mb),
    // createTestCase("tahi_lalat_18mb.jpg", "Cacar Air", imgTaiLalat18mb),
    // createTestCase("tahi_lalat_22mb.jpg", "Eczema", imgTaiLalat22mb),
];

export const CONFIG = {
    GRPC_ADDR: __ENV.GRPC_ADDR || '127.0.0.1:8008',
    REST_ADDR: __ENV.REST_ADDR || 'http://127.0.0.1:8088',

    CHUNK_SIZE: 64 * 1024, // 64KB chunks for optimal performance
    TIMEOUT: '30s',

    METADATA: {
        user_id: 'user-k6-test',
        image_type: 'image/jpeg',
        meta_tags: {
            source: 'k6-load-test',
            environment: 'testing'
        }
    }
};

export const SCENARIOS = {
    smoke: {
        executor: 'constant-vus',
        vus: 1,
        duration: '1m',
        gracefulStop: '10s',
    },

    load: {
        executor: 'ramping-vus',
        startVUs: 0,
        stages: [
            {duration: '2m', target: 10},  // Ramp up
            {duration: '5m', target: 10},  // Stay at load
            {duration: '2m', target: 0},   // Ramp down
        ],
        gracefulStop: '10s',
    },

    stress: {
        executor: 'ramping-vus',
        startVUs: 0,
        stages: [
            {duration: '2m', target: 20},  // Ramp to normal load
            {duration: '5m', target: 20},  // Stay at normal load
            {duration: '2m', target: 40},  // Ramp to stress load
            {duration: '5m', target: 40},  // Stay at stress load
            {duration: '2m', target: 0},   // Ramp down
        ],
        gracefulStop: '10s',
    },

    spike: {
        executor: 'ramping-vus',
        startVUs: 0,
        stages: [
            {duration: '1m', target: 10},  // Normal load
            {duration: '30s', target: 50}, // Spike!
            {duration: '3m', target: 50},  // Maintain spike
            {duration: '1m', target: 10},  // Recovery
            {duration: '1m', target: 0},   // Ramp down
        ],
        gracefulStop: '10s',
    },

    soak: {
        executor: 'constant-vus',
        vus: 15,
        duration: '30m',
        gracefulStop: '10s',
    },
};
