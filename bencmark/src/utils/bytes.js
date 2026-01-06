export function calcBytes(data) {
    if (data === null || data === undefined) return 0;

    // ArrayBuffer / TypedArray
    if (data instanceof ArrayBuffer) {
        return data.byteLength;
    }

    // String
    if (typeof data === 'string') {
        return data.length;
    }

    // Object / JSON
    if (typeof data === 'object') {
        try {
            return JSON.stringify(data).length;
        } catch (_) {
            return 0;
        }
    }

    return 0;
}