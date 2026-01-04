import {grpc_fetch} from "../utils/grpc_utils.js";
import grpc from "k6/net/grpc";

const client = new grpc.Client();
client.load(['../../protobuf'], 'citra.proto');

const imageBin = open('../test-images/sample.jpg', 'b')

export const options = {
    scenarios: {
        concurrent_users: {
            executor: 'constant-vus',
            vus: 20,
            duration: '2m',
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
