import grpc from "k6/net/grpc"
import encoding from 'k6/encoding'
import {check} from "k6";

const client = new grpc.Client();
client.load(['../../protobuf'], 'citra.proto');

const imageBin = open('../test-images/sample.jpg', 'b')

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

export default () => {
    client.connect('127.0.0.1:8008', {
        plaintext: true,
        timeout: "10s"
    });

    const stream = new grpc.Stream(
        client, 'dermatoai.SkinAnalysisService/AnalyzeSkin', {
            tags: {
                protocol: 'grpc',
            }
        }
    );

    stream.on('data', (res) => {
        check(res, {
            'status is 200': (r) => r.message !== null,
        })
    });

    stream.on('error', (err) => {
        console.error('gRPC ERROR');
        console.error('code:', err.code);
        console.error('message:', err.message);
        client.close();
    });

    stream.on('end', () => {
        client.close();
    });

    stream.write({
        info: {
            user_id: 'user-grpc-test-k6',
            image_type: 'jpg',
        },
    });

    const chunkSize = 128 * 1024;
    for (let i = 0; i < imageBin.byteLength; i += chunkSize) {
        stream.write({
            chunk: encoding.b64encode(
                imageBin.slice(i, i + chunkSize)
            ),
        });
    }

    stream.end();
}