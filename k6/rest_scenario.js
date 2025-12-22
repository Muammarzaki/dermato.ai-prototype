import http from 'k6/http';
import {check} from "k6";

const imageBin = open('./test-images/sample.jpg', 'b');

export const options = {
    vus: 100,
    duration: '30s'
}
export default function () {
    const url = 'http://localhost:8088/analyze-skin';

    const data = {
        file: http.file(imageBin, 'sample.jpg', 'image/jpeg'),
        user_id: 'user-rest-test-k6',
    };

    const res = http.post(url, data);

    check(res, {
        'status is 200': (r) => r.status === 200,
    });

}