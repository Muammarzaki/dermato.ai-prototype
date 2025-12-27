import grpc from "k6/net/grpc"
import {grpc_fetch} from "./utils/grpc_utils.js";

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

    grpc_fetch(1024 * 1024, client, imageBin);
}
