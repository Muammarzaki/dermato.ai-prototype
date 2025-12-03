# Build stage
FROM golang:1.21-bullseye AS builder

WORKDIR /app

# Copy go mod and sum files
COPY go.mod go.sum ./

# Download dependencies
RUN go mod download

# Copy the source code
COPY . .

# Build the application
RUN CGO_ENABLED=1 GOOS=linux go build -o main .

# Runtime stage
FROM debian:bullseye-slim

WORKDIR /app

# Install required dependencies for ONNX runtime
RUN apt-get update && apt-get install -y \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Download and install ONNX Runtime
RUN wget https://github.com/microsoft/onnxruntime/releases/download/v1.15.1/onnxruntime-linux-x64-1.15.1.tgz \
    && tar -xzf onnxruntime-linux-x64-1.15.1.tgz \
    && cp onnxruntime-linux-x64-1.15.1/lib/libonnxruntime.so* /usr/lib/ \
    && rm -rf onnxruntime-linux-x64-1.15.1*

# Copy the compiled binary from builder
COPY --from=builder /app/main .

# Copy any additional required files (models, configs, etc.)
COPY ../models/model_dermatoai94.onnx ./models/model.onnx
COPY ../models/model_classes.json ./models/classes.json

# Expose the ports your application uses
EXPOSE 8008 8088

# Set any environment variables if needed
ENV ENVIRONMENT=production

# Run the application
CMD ["./main"]