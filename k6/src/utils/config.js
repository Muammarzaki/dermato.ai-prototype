// Membaca file gambar sekali saja ke RAM saat script dimulai (Init Context)
// Pastikan path ini benar relatif terhadap tempat Anda menjalankan k6
const imgData = open('../../test-images/sample.jpg', 'b');

export const TEST_CONFIG = {
    // Alamat Server
    GRPC_ADDR: '127.0.0.1:8008',
    REST_ADDR: 'http://127.0.0.1:8008', 

    // Data Gambar (Binary)
    IMAGE_DATA: imgData,

    // Ukuran Chunk untuk gRPC (Sangat PENTING untuk performa)
    // 64KB seringkali menjadi performa terbaik untuk throughput LAN
    CHUNK_SIZE: 64 * 1024,

    // Metadata dummy
    METADATA: {
        user_id: "user-test-k6",
        image_type: "image/jpeg",
        meta_tags: JSON.stringify({ source: "k6-load-test" })
    }
};