package main

import (
	"log"
	"model-inference-service/api"
	"net"

	"github.com/gofiber/fiber/v2"
	"google.golang.org/grpc"

	pb "model-inference-service/gen"
)

func main() {
	restsServer := fiber.New()
	grpcServer := grpc.NewServer()

	pb.RegisterSkinAnalysisServiceServer(grpcServer, &api.SkinAnalysisServer{})

	restsServer.Post("/analyze-skin", api.HandleFileUpload)

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
