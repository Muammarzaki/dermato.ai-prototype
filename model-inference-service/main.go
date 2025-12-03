package main

import (
	"log"
	"model-inference-service/api"
	"model-inference-service/model"
	"model-inference-service/service"
	"net"
	"os"

	pb "model-inference-service/gen"

	"github.com/gofiber/fiber/v2"
	"github.com/joho/godotenv"
	"google.golang.org/grpc"
)

func main() {
	err := godotenv.Load()
	if err != nil {
		log.Fatal("Error to load .env file")
	}

	modelPath := os.Getenv("ONNX_MODEL_PATH")
	if modelPath == "" {
		modelPath = "./models/model.onnx"
	}

	onnxModel, err := model.NewONNXModel(modelPath)
	if err != nil {
		log.Fatalf("Failed to load ONNX model: %v", err)
	}
	defer func(onnxModel *model.ONNXModel) {
		err := onnxModel.Close()
		if err != nil {
			log.Fatalf("Failed to close ONNX model: %v", err)
		}
	}(onnxModel)

	inferenceService := service.NewInferenceService(onnxModel)

	skinAnalysisServer := api.NewSkinAnalysisServer(inferenceService)
	restHandler := api.HandleFileUpload(inferenceService)

	restsServer := fiber.New()
	grpcServer := grpc.NewServer()

	pb.RegisterSkinAnalysisServiceServer(grpcServer, skinAnalysisServer)

	restsServer.Post("/analyze-skin", restHandler)

	go func() {
		lis, err := net.Listen("tcp", ":8008")
		if err != nil {
			log.Fatalf("Failed to listen: %v", err)
		}
		log.Printf("Starting gRPC server on :8008")
		if err := grpcServer.Serve(lis); err != nil {
			log.Fatalf("Failed to serve gRPC: %v", err)
		}
	}()

	go func() {
		log.Printf("Starting Fiber server on :8088")
		if err := restsServer.Listen(":8088"); err != nil {
			log.Fatalf("Failed to serve Fiber: %v", err)
		}
	}()

	select {}
}
