import grpc from 'k6/net/grpc';
import { b64encode } from 'k6/encoding';
import {TEST_CONFIG} from '../utils/config.js';

const client = new grpc.Client();

client.load(['../../protobuf'], 'citra.proto');

export const options = {
    // Skenario pengujian: Constant throughput untuk melihat batas maksimal
    scenarios: {
        grpc_throughput: {
            executor: 'constant-vus',
            vus: 10, // Mulai dengan 10 user simultan
            duration: '30s',
        },
    },
};

export default () => {
    client.connect(TEST_CONFIG.GRPC_ADDR, {
        plaintext: true,
        timeout: '10s'
    });

    const stream = new grpc.Stream(client, 'dermatoai.SkinAnalysisService/AnalyzeSkin');

    stream.on('error', (err) => {
        if (err && err.message && !err.message.includes('canceled')) {
            console.error('Stream Error: ' + JSON.stringify(err));
        }
    });

    // 1. Kirim Metadata (ImageInfo)
    stream.write({
        info: {
            user_id: TEST_CONFIG.METADATA.user_id,
            image_type: TEST_CONFIG.METADATA.image_type,
            metadata: { "source": "k6" }
        }
    });

    // 2. Kirim Chunk (Binary -> Base64)
    const totalBytes = TEST_CONFIG.IMAGE_DATA.byteLength;
    let offset = 0;

    while (offset < totalBytes) {
        const end = Math.min(offset + TEST_CONFIG.CHUNK_SIZE, totalBytes);

        // Ambil potongan ArrayBuffer
        const chunkBuffer = TEST_CONFIG.IMAGE_DATA.slice(offset, end);

        // Encode ke Base64 String
        // k6 akan otomatis mendekode ini kembali ke bytes saat dikirim via gRPC
        const chunkBase64 = b64encode(chunkBuffer);

        stream.write({
            chunk: chunkBase64
        });

        offset = end;
    }

    stream.end();
    client.close();
};