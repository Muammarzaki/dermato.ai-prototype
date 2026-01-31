# Build stage
FROM golang:1.21-bookworm AS builder

WORKDIR /app

# Copy the source code
COPY . .

# Download dependencies
RUN go mod download

# Build the application
RUN CGO_ENABLED=1 GOOS=linux go build -o main .

# Runtime stage
FROM debian:bookworm-slim

WORKDIR /app

# Install required dependencies for ONNX runtime
RUN apt-get update && apt-get install -y \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-linux-x64-1.23.2.tgz \
    && tar -xzf onnxruntime-linux-x64-1.23.2.tgz \
    && cp onnxruntime-linux-x64-1.23.2/lib/libonnxruntime.so* /usr/lib/ \
    && ln -s /usr/lib/libonnxruntime.so /usr/lib/onnxruntime.so \
    && rm -rf onnxruntime-linux-x64-1.23.2*

# Copy the compiled binary from builder
COPY --from=builder /app/main .

# Copy any additional required files (models, configs, etc.)
RUN mkdir -p models
COPY --from=builder /app/models/model_dermatoai94.onnx ./models/model.onnx
COPY --from=builder /app/models/model_classes.json ./models/classes.json

# Expose the ports your application uses
EXPOSE 8008 8088

# Set any environment variables if needed
ENV ENVIRONMENT=production
ENV LD_LIBRARY_PATH=/usr/lib:/usr/local/lib

# Run the application
CMD ["./main"]