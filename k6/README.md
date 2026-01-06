# K6 Load Testing Suite

Balanced load testing for gRPC and REST API endpoints with multiple test scenarios.

## 📁 Project Structure

```
k6/
├── src/
│   ├── config/
│   │   └── config.js                 # Centralized configuration
│   ├── utils/
│   │   ├── grpc.utils.js             # gRPC client with detailed metrics
│   │   └── rest.utils.js             # REST client with detailed metrics
│   ├── tests/
│   │   ├── warmup.test.js            # Warmup server before testing
│   │   ├── balanced.test.js          # Balanced gRPC & REST tests
│   │   └── comparison.test.js        # Direct protocol comparison
├── test-images/
│   └── sample.jpg                    # Test image
├── protobuf/
│   └── citra.proto                   # Protocol buffer definitions
├── results/                          # Test results (auto-created)
├── run-tests.sh                      # Automated test runner
├── README.md                         # This file
├── LIFECYCLE.md                      # Request lifecycle documentation
├── TROUBLESHOOTING.md                # Troubleshooting guide
├── REFACTOR_SUMMARY.md               # Refactoring details
└── package.json
```

**Smoke Test** (1 VU for 1 minute):

```bash
k6 run -e SCENARIO=smoke src/tests/balanced.test.js
```

**Load Test** (Ramp to 10 VUs):

```bash
k6 run -e SCENARIO=load src/tests/balanced.test.js
```

**Stress Test** (Ramp to 40 VUs):

```bash
k6 run -e SCENARIO=stress src/tests/balanced.test.js
```

**Spike Test** (Sudden spike to 50 VUs):

```bash
k6 run -e SCENARIO=spike src/tests/balanced.test.js
```

**Soak Test** (15 VUs for 30 minutes):

```bash
k6 run -e SCENARIO=soak src/tests/balanced.test.js
```

#### 3. Direct Protocol Comparison

Runs both protocols with identical load (10 VUs for 5 minutes):

```bash
k6 run src/tests/comparison.test.js
```

#### 4. Custom Configuration

Override default settings:

```bash
k6 run \
  -e SCENARIO=stress \
  -e GRPC_ADDR=production.example.com:8008 \
  -e REST_ADDR=https://production.example.com \
  src/tests/balanced.test.js
```

## 📊 Test Scenarios Explained

### Smoke Test

- **Purpose**: Verify basic functionality
- **Load**: 1 VU for 1 minute
- **Use When**: After deployment, before running larger tests

### Load Test

- **Purpose**: Test under normal conditions
- **Load**: Ramp 0→10 VUs, hold 5min, ramp down
- **Use When**: Regular performance validation

### Stress Test

- **Purpose**: Find breaking point
- **Load**: Ramp 0→20→40 VUs, hold each stage
- **Use When**: Capacity planning

### Spike Test

- **Purpose**: Test sudden traffic increase
- **Load**: Quick spike from 10→50 VUs
- **Use When**: Preparing for marketing campaigns, flash sales

### Soak Test

- **Purpose**: Detect memory leaks and degradation
- **Load**: Constant 15 VUs for 30 minutes
- **Use When**: Before production deployment

## 📈 Metrics & Thresholds

### Default Thresholds

- 95th percentile response time < 5 seconds
- 99th percentile response time < 10 seconds
- Error rate < 5%
- Success rate > 95%
- Data transfer rate > 1MB/s

### Custom Metrics

- `grpc_req_duration`: gRPC request duration
- `grpc_chunks_sent`: Number of chunks sent per request
- `rest_req_duration`: REST request duration

### Understanding the Results

Saat membandingkan gRPC vs REST, perhatikan:

1. **Backend Processing Time** - Ini harus MIRIP karena komputasi sama
    - Jika berbeda jauh, ada masalah di salah satu implementasi
    - Metrik ini paling penting untuk fairness

2. **Connection Overhead**
    - gRPC: Persistent connection, overhead rendah setelah koneksi pertama
    - REST: Per-request connection, overhead lebih tinggi di HTTP/1.1

3. **Data Transfer**
    - gRPC: Streaming chunks, Protocol Buffers binary
    - REST: Multipart form data, base64 encoding overhead

4. **Total Request Time** = Connection + Upload + Backend + Download
    - Ini yang dirasakan end user
    - gRPC biasanya lebih baik untuk requests berulang
    - REST lebih sederhana untuk debugging

### View Results

```bash
# Output results to JSON for detailed analysis
k6 run --out json=results.json src/tests/balanced.test.js

# Output to InfluxDB + Grafana
k6 run --out influxdb=http://localhost:8086/k6 src/tests/balanced.test.js

# Summary output only
k6 run --summary-export=summary.json src/tests/comparison.test.js
```

### Interpreting Comparison Results

```
Contoh output yang FAIR:
✓ grpc_backend_processing_time...avg=2.3s  p(95)=3.8s
✓ rest_backend_processing_time...avg=2.4s  p(95)=3.9s
^ Backend time HARUS mirip! Jika beda > 20%, check implementasi

✓ grpc_connection_time...........avg=50ms   p(95)=100ms
✓ rest_connection_time...........avg=200ms  p(95)=500ms
^ Connection overhead gRPC lebih rendah (expected)

✓ grpc_total_request_time........avg=3.2s   p(95)=5.1s
✓ rest_total_request_time........avg=3.8s   p(95)=6.2s
^ Total time dari user perspective
```

## 🔧 Configuration

Edit `src/config/config.js` to customize:

- Server endpoints
- Chunk size for gRPC streaming
- Test timeouts
- Metadata
- Scenario parameters
- Performance thresholds

## 📝 Example Output

```
scenarios: (100.00%) 2 scenarios, 20 max VUs, 5m30s max duration

✓ gRPC: response received
✓ gRPC: all chunks sent
✓ REST: status is 200
✓ REST: response time < 10s

checks.........................: 98.50% ✓ 1970  ✗ 30
grpc_req_duration..............: avg=2.3s   min=1.1s med=2.1s max=4.5s p(95)=3.8s
rest_req_duration..............: avg=1.8s   min=0.9s med=1.6s max=3.2s p(95)=2.9s
http_req_duration..............: avg=1.8s   min=0.9s med=1.6s max=3.2s
data_received..................: 125 MB  2.1 MB/s
data_sent......................: 89 MB   1.5 MB/s
```

## 🎯 Best Practices

1. **Start Small**: Begin with smoke tests before running stress tests
2. **Monitor Server**: Watch server metrics (CPU, memory, disk I/O) during tests
3. **Gradual Scaling**: Use ramp-up periods to avoid overwhelming the system
4. **Realistic Delays**: Include sleep() to simulate real user behavior
5. **Version Control**: Commit baseline test results for comparison
6. **CI/CD Integration**: Run smoke tests on every deployment

## 🤖 Automated Test Runner

Gunakan shell script untuk menjalankan test suite dengan mudah:

```bash
# Make script executable (first time only)
chmod +x run-tests.sh

# Run full test suite
./run-tests.sh full-suite

# Run specific tests
./run-tests.sh smoke
./run-tests.sh balanced load
./run-tests.sh comparison
./run-tests.sh detailed

# See all options
./run-tests.sh help
```

Script ini akan:

- ✅ Check k6 installation
- ✅ Verify servers are running
- ✅ Run warmup automatically
- ✅ Save results with timestamps
- ✅ Generate comparison reports

## 🐛 Troubleshooting

Untuk troubleshooting lengkap, lihat [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

### Quick Fixes

**Connection Refused:**

```bash
# Check if services are running
netstat -an | grep 8008
netstat -an | grep 8088

# Check server process
ps aux | grep your-server
```

**Malformed HTTP Response:**

```
Error: malformed HTTP response "\x00\x00\x06\x04..."
```

→ Server di port 8008 hanya gRPC! Update config untuk gunakan port berbeda untuk REST.

**High Error Rate:**

```bash
# Start with minimal load
./run-tests.sh smoke

# Check server resources
top -p $(pgrep -f your-server)
```

**Backend Time Berbeda:**
→ Check implementasi backend - harus menggunakan code yang sama!

## 📚 Additional Resources

- [k6 Documentation](https://k6.io/docs/)
- [gRPC Performance Best Practices](https://grpc.io/docs/guides/performance/)
- [Load Testing Best Practices](https://k6.io/docs/testing-guides/test-types/)