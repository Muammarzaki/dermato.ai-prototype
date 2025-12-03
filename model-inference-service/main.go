package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"model-inference-service/api"
	"model-inference-service/data"
	"model-inference-service/event"
	"model-inference-service/model"
	"model-inference-service/service"
	"net"
	"os"
	"time"

	pb "model-inference-service/gen"

	"github.com/gofiber/fiber/v2"
	"github.com/google/uuid"
	"github.com/joho/godotenv"
	"google.golang.org/grpc"
	"gorm.io/driver/postgres"
	"gorm.io/gorm"
)

type Config struct {
	ModelPath     string
	ClassDictPath string
	DBConfig      DBConfig
}

type DBConfig struct {
	Host     string
	User     string
	Password string
	Name     string
	Port     string
}

func loadConfig() (*Config, error) {
	if err := godotenv.Load(); err != nil {
		return nil, fmt.Errorf("error loading .env file: %v", err)
	}

	modelPath := os.Getenv("ONNX_MODEL_PATH")
	if modelPath == "" {
		modelPath = "./models/model.onnx"
	}

	classDictPath := os.Getenv("CLASS_DICTIONARY_PATH")
	if classDictPath == "" {
		classDictPath = "./models/classes.json"
	}

	return &Config{
		ModelPath:     modelPath,
		ClassDictPath: classDictPath,
		DBConfig: DBConfig{
			Host:     os.Getenv("DB_HOST"),
			User:     os.Getenv("DB_USER"),
			Password: os.Getenv("DB_PASSWORD"),
			Name:     os.Getenv("DB_NAME"),
			Port:     os.Getenv("DB_PORT"),
		},
	}, nil
}

func loadClassDictionary(path string) ([]string, error) {
	classesFile, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read class dictionary: %v", err)
	}

	var classDict []string
	if err := json.Unmarshal(classesFile, &classDict); err != nil {
		return nil, fmt.Errorf("failed to parse class dictionary: %v", err)
	}

	return classDict, nil
}

func initDB(config DBConfig) (*gorm.DB, error) {
	dsn := fmt.Sprintf("host=%s user=%s password=%s dbname=%s port=%s sslmode=disable",
		config.Host, config.User, config.Password, config.Name, config.Port)

	db, err := gorm.Open(postgres.Open(dsn), &gorm.Config{})
	if err != nil {
		return nil, fmt.Errorf("failed to connect to database: %v", err)
	}

	if err := db.AutoMigrate(&data.Chronic{}); err != nil {
		return nil, fmt.Errorf("failed to migrate database: %v", err)
	}

	return db, nil
}

func startChronicEventProcessor(repository *data.ChronicRepository, events chan event.Event) {
	go func() {
		for ev := range events {
			err := repository.Create(context.Background(), &data.Chronic{
				ID:        uuid.New(),
				Body:      ev.Body,
				Status:    ev.Status,
				CreatedAt: time.Now(),
			})
			if err != nil {
				log.Printf("failed to save chronic event: %v", err)
			}
		}
	}()
}

func startServers(inferenceService *service.InferenceService, events chan event.Event) {
	// Initialize servers
	restsServer := fiber.New()
	grpcServer := grpc.NewServer()

	// Setup handlers and routes
	skinAnalysisServer := api.NewSkinAnalysisServer(inferenceService, events)
	restHandler := api.HandleFileUpload(inferenceService, events)

	pb.RegisterSkinAnalysisServiceServer(grpcServer, skinAnalysisServer)
	restsServer.Post("/analyze-skin", restHandler)

	// Start gRPC server
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

	// Start a REST server
	go func() {
		log.Printf("Starting Fiber server on :8088")
		if err := restsServer.Listen(":8088"); err != nil {
			log.Fatalf("Failed to serve Fiber: %v", err)
		}
	}()
}

func main() {
	config, err := loadConfig()
	if err != nil {
		log.Fatal(err)
	}

	classDict, err := loadClassDictionary(config.ClassDictPath)
	if err != nil {
		log.Fatal(err)
	}

	db, err := initDB(config.DBConfig)
	if err != nil {
		log.Fatal(err)
	}

	onnxModel, err := model.NewONNXModel(config.ModelPath)
	if err != nil {
		log.Fatalf("Failed to load ONNX model: %v", err)
	}
	defer func() {
		if err := onnxModel.Close(); err != nil {
			log.Printf("Failed to close ONNX model: %v", err)
		}
	}()

	repository := data.NewChronicRepository(db)
	chronicEvents := make(chan event.Event)
	startChronicEventProcessor(repository, chronicEvents)

	inferenceService := service.NewInferenceService(onnxModel, &classDict)
	startServers(inferenceService, chronicEvents)

	select {}
}
