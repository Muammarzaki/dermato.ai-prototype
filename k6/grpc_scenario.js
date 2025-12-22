import grpc from "k6/net/grpc"
import encoding from 'k6/encoding'
import {check} from "k6";

const client = new grpc.Client();
client.load(['../protobuf'], 'citra.proto');

const imageBin = open('./test-images/sample.jpg', 'b')

export const options = {
    vus: 100,
    duration: '30s'
}

export default () => {
    client.connect('127.0.0.1:8008', {plaintext: true});

    const stream = new grpc.Stream(
        client, 'dermatoai.SkinAnalysisService/AnalyzeSkin'
    );


    stream.on('data', (res) => {
        check(res, {
            'status is 200': (r) => r !== null,
        })
        client.close();
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