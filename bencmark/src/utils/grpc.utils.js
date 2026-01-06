// src/utils/grpc.utils.js
import grpc from 'k6/net/grpc';
import { check } from 'k6';
import { b64encode } from 'k6/encoding';
import { Trend, Counter, Rate, Gauge } from 'k6/metrics';
import { calcBytes } from './bytes.js';

// ===================== METRICS =====================
const grpcReqDuration = new Trend('grpc_req_duration', true);
const grpcWaitingTime = new Trend('grpc_req_waiting', true);
const grpcSendingTime = new Trend('grpc_req_sending', true);
const grpcConnectingTime = new Trend('grpc_req_connecting', true);

const grpcReqSending = new Counter('grpc_data_sent');
const grpcReqReceived = new Counter('grpc_data_received');
const grpcChunkSize = new Trend('grpc_chunk_size', true);
const grpcChunksPerRequest = new Trend('grpc_chunks_per_request', true);

const grpcStreamDuration = new Trend('grpc_stream_duration', true);
const grpcMessagesPerStream = new Trend('grpc_messages_per_stream', true);

const grpcReqFailed = new Counter('grpc_req_failed');
const grpcReqSucceeded = new Rate('grpc_req_success_rate');
const grpcActiveStreams = new Gauge('grpc_active_streams');

// ===================== CLIENT =====================
export class GrpcClient {
    constructor(address, protoPath) {
        this.client = new grpc.Client();
        this.client.load(['../../../protobuf'], protoPath);
        this.address = address;
        this.activeStreamCount = 0;
    }

    connect(timeout = '30s') {
        const start = Date.now();
        this.client.connect(this.address, { plaintext: true, timeout });
        grpcConnectingTime.add(Date.now() - start);
    }

    close() {
        this.client.close();
    }

    analyzeSkin(imageData, metadata, chunkSize = 64 * 1024, onClose) {
        const requestStart = Date.now();
        const streamStart = Date.now();

        let chunks = 0;
        let bytesSent = 0;
        let bytesReceived = 0;
        let sendingStart = 0;
        let sendingEnd = 0;
        let responseStart = 0;
        let hasError = false;
        let responseReceived = false;

        this.activeStreamCount++;
        grpcActiveStreams.add(this.activeStreamCount);

        const stream = new grpc.Stream(
            this.client,
            'dermatoai.SkinAnalysisService/AnalyzeSkin',
            { tags: { protocol: 'grpc' } }
        );

        // ================= RESPONSE =================
        stream.on('data', (res) => {
            if (!responseReceived) {
                responseStart = Date.now();
                responseReceived = true;
            }

            bytesReceived += calcBytes(res);

            const ok = check(res, {
                'gRPC: response exists': (r) => r !== null,
                'gRPC: has analysisId': (r) => typeof r.analysisId === 'string',
                'gRPC: has results': (r) => Array.isArray(r.results),
                'gRPC: confidence valid': (r) =>
                    r.results?.[0]?.confidence >= 0 &&
                    r.results?.[0]?.confidence <= 1,
            });

            if (!ok) hasError = true;
        });

        // ================= ERROR =================
        stream.on('error', (err) => {
            hasError = true;
            grpcReqFailed.add(1);
            console.error(`gRPC Error [${err.code}]: ${err.message}`);
        });

        // ================= END =================
        stream.on('end', () => {
            grpcReqDuration.add(Date.now() - requestStart);
            grpcStreamDuration.add(Date.now() - streamStart);

            if (sendingStart && sendingEnd) {
                grpcSendingTime.add(sendingEnd - sendingStart);
            }

            if (responseStart && sendingEnd) {
                grpcWaitingTime.add(responseStart - sendingEnd);
            }

            grpcReqSending.add(bytesSent);
            grpcReqReceived.add(bytesReceived);
            grpcChunksPerRequest.add(chunks);
            grpcMessagesPerStream.add(chunks + 1);
            grpcReqSucceeded.add(!hasError);

            this.activeStreamCount--;
            grpcActiveStreams.add(this.activeStreamCount);

            if (onClose) onClose();
        });

        // ================= SEND =================
        try {
            sendingStart = Date.now();

            const metaMsg = {
                info: {
                    user_id: metadata.user_id,
                    image_type: metadata.image_type,
                    metadata: metadata.meta_tags,
                },
            };

            stream.write(metaMsg);
            bytesSent += calcBytes(metaMsg);

            let offset = 0;
            while (offset < imageData.byteLength) {
                const end = Math.min(offset + chunkSize, imageData.byteLength);
                const chunk = imageData.slice(offset, end);

                stream.write({ chunk: b64encode(chunk) });

                const size = end - offset;
                bytesSent += size;
                grpcChunkSize.add(size);
                chunks++;

                offset = end;
            }

            sendingEnd = Date.now();
            stream.end();

        } catch (e) {
            hasError = true;
            grpcReqFailed.add(1);
            console.error(`gRPC Send Error: ${e.message}`);
            try { stream.end(); } catch (_) {}
        }
    }
}
