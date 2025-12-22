import grpc from "k6/net/grpc";
import {check} from "k6";
import encoding from "k6/encoding";

export function grpc_fetch(chunkSize, client, imageBin,) {

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

    for (let i = 0; i < imageBin.byteLength; i += chunkSize) {
        stream.write({
            chunk: encoding.b64encode(
                imageBin.slice(i, i + chunkSize)
            ),
        });
    }

    stream.end();
}