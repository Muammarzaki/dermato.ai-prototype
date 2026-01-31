# Variables
PROJECT_NAME := skin-analyzer-service
PROTO_DIR := internal/pb
CMD_DIR := cmd
BUILD_DIR := build
MODELS_DIR := models

# Go variables
GO := go
GOFLAGS := -v
GOOS := linux
GOARCH := amd64
CGO_ENABLED := 1

# Docker variables
DOCKER := docker
DOCKER_COMPOSE := docker-compose
GRPC_IMAGE := skin-analyzer-grpc
REST_IMAGE := skin-analyzer-rest

# Protobuf variables
PROTOC := protoc
PROTOC_GEN_GO := $(shell go env GOPATH)/bin/protoc-gen-go
PROTOC_GEN_GO_GRPC := $(shell go env GOPATH)/bin/protoc-gen-go-grpc

# Colors for output
COLOR_RESET := \033[0m
COLOR_BOLD := \033[1m
COLOR_GREEN := \033[32m
COLOR_YELLOW := \033[33m
COLOR_BLUE := \033[34m

.PHONY: all help build run clean test proto docker-build docker-up docker-down deps install-tools

# Default target
all: help

## help: Display this help message
help:
	@echo "$(COLOR_BOLD)$(PROJECT_NAME) Makefile Commands:$(COLOR_RESET)"
	@echo ""
	@grep -E '^## ' $(MAKEFILE_LIST) | sed 's/## /  $(COLOR_GREEN)/' | sed 's/:/ $(COLOR_RESET)-/'
	@echo ""

## deps: Download Go dependencies
deps:
	@echo "$(COLOR_BLUE)Downloading dependencies...$(COLOR_RESET)"
	$(GO) mod download
	$(GO) mod tidy
	@echo "$(COLOR_GREEN)✓ Dependencies downloaded$(COLOR_RESET)"

## install-tools: Install required tools (protoc, protoc-gen-go, etc.)
install-tools:
	@echo "$(COLOR_BLUE)Installing required tools...$(COLOR_RESET)"
	$(GO) install google.golang.org/protobuf/cmd/protoc-gen-go@latest
	$(GO) install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest
	@echo "$(COLOR_GREEN)✓ Tools installed$(COLOR_RESET)"

## proto: Generate protobuf files
proto:
	@echo "$(COLOR_BLUE)Generating protobuf files...$(COLOR_RESET)"
	@if [ ! -f "$(PROTOC_GEN_GO)" ]; then \
		echo "$(COLOR_YELLOW)protoc-gen-go not found, installing...$(COLOR_RESET)"; \
		$(GO) install google.golang.org/protobuf/cmd/protoc-gen-go@latest; \
	fi
	@if [ ! -f "$(PROTOC_GEN_GO_GRPC)" ]; then \
		echo "$(COLOR_YELLOW)protoc-gen-go-grpc not found, installing...$(COLOR_RESET)"; \
		$(GO) install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest; \
	fi
	$(PROTOC) --go_out=. --go_opt=paths=source_relative \
		--go-grpc_out=. --go-grpc_opt=paths=source_relative \
		$(PROTO_DIR)/skin_analyzer.proto
	@echo "$(COLOR_GREEN)✓ Protobuf files generated$(COLOR_RESET)"

## build: Build the application binary
build: deps
	@echo "$(COLOR_BLUE)Building application...$(COLOR_RESET)"
	@mkdir -p $(BUILD_DIR)
	CGO_ENABLED=$(CGO_ENABLED) GOOS=$(GOOS) GOARCH=$(GOARCH) \
		$(GO) build $(GOFLAGS) -o $(BUILD_DIR)/main ./$(CMD_DIR)/main.go
	@echo "$(COLOR_GREEN)✓ Build complete: $(BUILD_DIR)/main$(COLOR_RESET)"

## build-local: Build for local environment
build-local:
	@echo "$(COLOR_BLUE)Building for local environment...$(COLOR_RESET)"
	@mkdir -p $(BUILD_DIR)
	CGO_ENABLED=$(CGO_ENABLED) $(GO) build $(GOFLAGS) -o $(BUILD_DIR)/main ./$(CMD_DIR)/main.go
	@echo "$(COLOR_GREEN)✓ Local build complete$(COLOR_RESET)"

## run: Run the application locally (gRPC mode)
run: build-local
	@echo "$(COLOR_BLUE)Running application in gRPC mode...$(COLOR_RESET)"
	./$(BUILD_DIR)/main

## run-rest: Run the application in REST mode
run-rest: build-local
	@echo "$(COLOR_BLUE)Running application in REST mode...$(COLOR_RESET)"
	REST_MODE=true ./$(BUILD_DIR)/main

## test: Run tests
test:
	@echo "$(COLOR_BLUE)Running tests...$(COLOR_RESET)"
	$(GO) test -v -race -coverprofile=coverage.out ./...
	@echo "$(COLOR_GREEN)✓ Tests complete$(COLOR_RESET)"

## test-coverage: Run tests with coverage report
test-coverage: test
	@echo "$(COLOR_BLUE)Generating coverage report...$(COLOR_RESET)"
	$(GO) tool cover -html=coverage.out -o coverage.html
	@echo "$(COLOR_GREEN)✓ Coverage report generated: coverage.html$(COLOR_RESET)"

## lint: Run linter
lint:
	@echo "$(COLOR_BLUE)Running linter...$(COLOR_RESET)"
	@if command -v golangci-lint >/dev/null 2>&1; then \
		golangci-lint run ./...; \
		echo "$(COLOR_GREEN)✓ Linting complete$(COLOR_RESET)"; \
	else \
		echo "$(COLOR_YELLOW)golangci-lint not installed. Install with: go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest$(COLOR_RESET)"; \
	fi

## fmt: Format Go code
fmt:
	@echo "$(COLOR_BLUE)Formatting code...$(COLOR_RESET)"
	$(GO) fmt ./...
	@echo "$(COLOR_GREEN)✓ Code formatted$(COLOR_RESET)"

## vet: Run go vet
vet:
	@echo "$(COLOR_BLUE)Running go vet...$(COLOR_RESET)"
	$(GO) vet ./...
	@echo "$(COLOR_GREEN)✓ Vet complete$(COLOR_RESET)"

## clean: Clean build artifacts
clean:
	@echo "$(COLOR_BLUE)Cleaning build artifacts...$(COLOR_RESET)"
	rm -rf $(BUILD_DIR)
	rm -f coverage.out coverage.html
	$(GO) clean
	@echo "$(COLOR_GREEN)✓ Clean complete$(COLOR_RESET)"

## docker-build: Build Docker images
docker-build:
	@echo "$(COLOR_BLUE)Building Docker images...$(COLOR_RESET)"
	$(DOCKER_COMPOSE) build
	@echo "$(COLOR_GREEN)✓ Docker images built$(COLOR_RESET)"

## docker-build-grpc: Build gRPC Docker image only
docker-build-grpc:
	@echo "$(COLOR_BLUE)Building gRPC Docker image...$(COLOR_RESET)"
	$(DOCKER_COMPOSE) build $(GRPC_IMAGE)
	@echo "$(COLOR_GREEN)✓ gRPC Docker image built$(COLOR_RESET)"

## docker-build-rest: Build REST Docker image only
docker-build-rest:
	@echo "$(COLOR_BLUE)Building REST Docker image...$(COLOR_RESET)"
	$(DOCKER_COMPOSE) build $(REST_IMAGE)
	@echo "$(COLOR_GREEN)✓ REST Docker image built$(COLOR_RESET)"

## docker-up: Start Docker containers
docker-up:
	@echo "$(COLOR_BLUE)Starting Docker containers...$(COLOR_RESET)"
	$(DOCKER_COMPOSE) up -d
	@echo "$(COLOR_GREEN)✓ Containers started$(COLOR_RESET)"
	@echo "$(COLOR_YELLOW)gRPC server: localhost:8008$(COLOR_RESET)"
	@echo "$(COLOR_YELLOW)REST server: localhost:8088$(COLOR_RESET)"

## docker-up-grpc: Start gRPC container only
docker-up-grpc:
	@echo "$(COLOR_BLUE)Starting gRPC container...$(COLOR_RESET)"
	$(DOCKER_COMPOSE) up -d $(GRPC_IMAGE)
	@echo "$(COLOR_GREEN)✓ gRPC container started on port 8008$(COLOR_RESET)"

## docker-up-rest: Start REST container only
docker-up-rest:
	@echo "$(COLOR_BLUE)Starting REST container...$(COLOR_RESET)"
	$(DOCKER_COMPOSE) up -d $(REST_IMAGE)
	@echo "$(COLOR_GREEN)✓ REST container started on port 8088$(COLOR_RESET)"

## docker-down: Stop Docker containers
docker-down:
	@echo "$(COLOR_BLUE)Stopping Docker containers...$(COLOR_RESET)"
	$(DOCKER_COMPOSE) down
	@echo "$(COLOR_GREEN)✓ Containers stopped$(COLOR_RESET)"

## docker-logs: Show Docker container logs
docker-logs:
	$(DOCKER_COMPOSE) logs -f

## docker-logs-grpc: Show gRPC container logs
docker-logs-grpc:
	$(DOCKER_COMPOSE) logs -f $(GRPC_IMAGE)

## docker-logs-rest: Show REST container logs
docker-logs-rest:
	$(DOCKER_COMPOSE) logs -f $(REST_IMAGE)

## docker-restart: Restart Docker containers
docker-restart: docker-down docker-up

## docker-clean: Remove Docker containers, images, and volumes
docker-clean:
	@echo "$(COLOR_BLUE)Cleaning Docker resources...$(COLOR_RESET)"
	$(DOCKER_COMPOSE) down -v --rmi all
	@echo "$(COLOR_GREEN)✓ Docker resources cleaned$(COLOR_RESET)"

## dev: Development mode - rebuild and restart
dev: clean build-local docker-down docker-build docker-up docker-logs

## check: Run all checks (fmt, vet, lint, test)
check: fmt vet lint test
	@echo "$(COLOR_GREEN)✓ All checks passed$(COLOR_RESET)"

## models-check: Check if model files exist
models-check:
	@echo "$(COLOR_BLUE)Checking model files...$(COLOR_RESET)"
	@if [ -f "$(MODELS_DIR)/model_dermatoai94.onnx" ]; then \
		echo "$(COLOR_GREEN)✓ ONNX model found$(COLOR_RESET)"; \
	else \
		echo "$(COLOR_YELLOW)⚠ ONNX model not found at $(MODELS_DIR)/model_dermatoai94.onnx$(COLOR_RESET)"; \
	fi
	@if [ -f "$(MODELS_DIR)/model_classes.json" ]; then \
		echo "$(COLOR_GREEN)✓ Classes file found$(COLOR_RESET)"; \
	else \
		echo "$(COLOR_YELLOW)⚠ Classes file not found at $(MODELS_DIR)/model_classes.json$(COLOR_RESET)"; \
	fi

## db-migrate: Run database migrations (if using migrations)
db-migrate:
	@echo "$(COLOR_BLUE)Running database migrations...$(COLOR_RESET)"
	@echo "$(COLOR_YELLOW)Migrations are handled automatically by GORM AutoMigrate$(COLOR_RESET)"

## env-example: Create .env.example file
env-example:
	@echo "$(COLOR_BLUE)Creating .env.example...$(COLOR_RESET)"
	@cat > .env.example <<EOF
			# Application Configuration
			ENVIRONMENT=development
			REST_MODE=false

			# ONNX Model Configuration
			ONNX_MODEL_PATH=./models/model.onnx
			CLASS_DICTIONARY_PATH=./models/classes.json

			# Database Configuration
			DB_HOST=localhost
			DB_USER=postgres
			DB_PASSWORD=your_password
			DB_NAME=skin_analyzer_db
			DB_PORT=5432
			EOF
	@echo "$(COLOR_GREEN)✓ .env.example created$(COLOR_RESET)"

## info: Display project information
info:
	@echo "$(COLOR_BOLD)Project Information:$(COLOR_RESET)"
	@echo "  Name: $(PROJECT_NAME)"
	@echo "  Go Version: $(shell $(GO) version)"
	@echo "  GOOS: $(GOOS)"
	@echo "  GOARCH: $(GOARCH)"
	@echo ""
	@echo "$(COLOR_BOLD)Directories:$(COLOR_RESET)"
	@echo "  Proto: $(PROTO_DIR)"
	@echo "  Command: $(CMD_DIR)"
	@echo "  Build: $(BUILD_DIR)"
	@echo "  Models: $(MODELS_DIR)"