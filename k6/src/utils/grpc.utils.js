// src/utils/grpc.utils.js
import grpc from 'k6/net/grpc';
import {check} from 'k6';
import {b64encode} from 'k6/encoding';
import {Trend} from 'k6/metrics';

// Custom metrics
const grpcDuration = new Trend('grpc_req_duration', true);
const chunksSent = new Trend('grpc_chunks_sent', true);

export class GrpcClient {
    constructor(address, protoPath) {
        this.client = new grpc.Client();
        this.client.load(['../../../protobuf'], protoPath);
        this.address = address;
    }

    connect(timeout = '30s') {
        this.client.connect(this.address, {
            plaintext: true,
            timeout: timeout,
        });
    }

    close() {
        this.client.close();
    }

    analyzeSkin(imageData, metadata, chunkSize = 64 * 1024, onClose) {
        const startTime = Date.now();
        let chunksCount = 0;
        let responseReceived = false;
        let analysisResult = null;

        const stream = new grpc.Stream(
            this.client,
            'dermatoai.SkinAnalysisService/AnalyzeSkin',
            {tags: {protocol: 'grpc'}}
        );

        // Handle response data
        stream.on('data', (response) => {
            responseReceived = true;
            analysisResult = response;

            check(response, {
                'gRPC: response received': (r) => r !== null,
                'gRPC: has valid message': (r) => r !== undefined,
            });
        });

        // Handle errors
        stream.on('error', (err) => {
            if (err && err.message && !err.message.includes('canceled')) {
                console.error(`gRPC Error [${err.code}]: ${err.message}`);
            }
            check(null, {
                'gRPC: no errors': () => false,
            });
        });

        // Handle stream end
        stream.on('end', () => {
            const duration = Date.now() - startTime;
            grpcDuration.add(duration);
            chunksSent.add(chunksCount);
            onClose()
        });

        try {
            // Send metadata
            stream.write({
                info: {
                    user_id: metadata.user_id,
                    image_type: metadata.image_type,
                    metadata: metadata.meta_tags,
                }
            });

            // Send image chunks
            const totalBytes = imageData.byteLength;
            let offset = 0;

            while (offset < totalBytes) {
                const end = Math.min(offset + chunkSize, totalBytes);
                const chunkBuffer = imageData.slice(offset, end);
                const chunkBase64 = b64encode(chunkBuffer);

                stream.write({
                    chunk: chunkBase64
                });

                chunksCount++;
                offset = end;
            }

            stream.end();

            check(null, {
                'gRPC: all chunks sent': () => chunksCount > 0,
                'gRPC: response received': () => responseReceived,
            });

            return analysisResult;

        } catch (error) {
            console.error(`gRPC Request Error: ${error.message}`);
            check(null, {
                'gRPC: request successful': () => false,
            });
            return null;
        }
    }
}