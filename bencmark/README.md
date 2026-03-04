# Locust Benchmark — Dermato.ai

Port dari k6 benchmark ke Locust, mendukung REST dan gRPC client-streaming.

## Struktur

```
locust/
├── locustfile.py               # Entry point utama
├── requirements.txt
├── protobuf/
│   └── skin_analyzer.proto     # Salin dari direktori protobuf project
├── test-images/                # Salin dari direktori test-images project
│   ├── tahi_lalat_1.5mb.jpg
│   └── ...
└── src/
    ├── config/
    │   └── config.py           # Konfigurasi terpusat
    ├── tasks/
    │   ├── rest_task.py        # Logic request REST
    │   └── grpc_task.py        # Logic request gRPC
    └── utils/
        └── grpc_client.py      # Channel + stub factory
```

## Install

### Windows (Git Bash / WSL)

If you're on Windows with Git Bash, the virtual environment symlinks might break. Use one of these methods:

**Method 1: Use Python directly (recommended for Windows)**
```bash
# Find your Python installation (e.g., C:\Python312\python.exe)
# Then install dependencies:
python -m pip install -r requirements.txt
```

**Method 2: Create venv with PowerShell**
```powershell
python -m venv locust-env
.\locust-env\Scripts\activate
pip install -r requirements.txt
```

**Method 3: Use WSL or Linux**
```bash
python3 -m venv locust-env
source locust-env/bin/activate
pip install -r requirements.txt
```

### Linux / macOS

```bash
python3 -m venv locust-env
source locust-env/bin/activate
pip install -r requirements.txt
```

## Jalankan

### Web UI (development)
```bash
locust -f locustfile.py
# buka http://localhost:8089
```

### Headless — REST saja
```bash
SCENARIO=load \
REST_ADDR=http://10.128.0.2:8088 \
locust -f locustfile.py RestUser --headless
```

### Headless — gRPC saja
```bash
SCENARIO=stress \
GRPC_ADDR=10.128.0.2:8008 \
locust -f locustfile.py GrpcUser --headless
```

### Headless — Kedua protokol sekaligus
```bash
SCENARIO=spike \
GRPC_ADDR=10.128.0.2:8008 \
REST_ADDR=http://10.128.0.2:8088 \
locust -f locustfile.py RestUser GrpcUser --headless
```

### Simpan hasil ke CSV
```bash
locust -f locustfile.py RestUser --headless \
  --csv=results/run_$(date +%Y%m%d_%H%M%S)
```

### Distributed (master + workers)
```bash
# Master
locust -f locustfile.py --master

# Worker (jalankan di tiap node)
locust -f locustfile.py --worker --master-host=<IP_MASTER>
```

## Scenarios

Pilih via env var `SCENARIO`:

| Scenario | Deskripsi                              |
|----------|----------------------------------------|
| `smoke`  | 1 VU, 1 menit — sanity check          |
| `load`   | Ramp 0→10→0, ~9 menit                 |
| `stress` | Ramp 0→20→40→0, ~16 menit             |
| `spike`  | Normal → spike 50 VU → recovery        |
| `soak`   | 15 VU konstan selama 30 menit          |

## Users

| Class       | Protokol         | Keterangan                    |
|-------------|------------------|-------------------------------|
| `RestUser`  | REST HTTP/1.1    | Multipart form-data upload    |
| `GrpcUser`  | gRPC HTTP/2      | Client-streaming, persistent  |
| `MixedUser` | REST + gRPC 50/50| Untuk perbandingan side-by-side |