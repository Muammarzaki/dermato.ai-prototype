import http from 'k6/http';
import {check} from "k6";

const imageBin = open('./test-images/sample.jpg', 'b');

export const options = {
    vus: 100,
    duration: '30s'
}
export default function () {
    // Endpoint REST (Default port di main.go untuk REST adalah 8088)
    const url = 'http://localhost:8088/analyze-skin';

    // Siapkan Payload Multipart
    const data = {
        file: http.file(imageBin, 'sample.jpg', 'image/jpeg'),
        user_id: 'user-rest-test-k6',
    };

    // Kirim Request POST
    const res = http.post(url, data);

    // Validasi
    check(res, {
        'status is 200': (r) => r.status === 200,
        'response time < 1s': (r) => r.timings.duration < 1000,
    });

}