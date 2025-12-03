package main

import (
	"context"
	"database/sql"
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
	"os/signal"
	"syscall"
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
	RestMode      bool
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

	restMode := os.Getenv("REST_MODE") == "true"

	return &Config{
		ModelPath:     modelPath,
		ClassDictPath: classDictPath,
		RestMode:      restMode,
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

func startChronicEventProcessor(ctx context.Context, repository *data.ChronicRepository, events chan event.Event) {
	go func() {
		defer close(events)
		for {
			select {
			case <-ctx.Done():
				log.Println("Stopping chronic event processor")
				return
			case ev, ok := <-events:
				if !ok {
					return
				}
				err := repository.Create(ctx, &data.Chronic{
					ID:        uuid.New(),
					Body:      ev.Body,
					Status:    ev.Status,
					CreatedAt: time.Now(),
				})
				if err != nil {
					log.Printf("failed to save chronic event: %v", err)
				}
			}
		}
	}()
}

func startServers(ctx context.Context, inferenceService *service.InferenceService, events chan event.Event, mode bool) error {
	errChan := make(chan error, 1)

	if !mode {
		grpcServer := grpc.NewServer()
		pb.RegisterSkinAnalysisServiceServer(grpcServer, api.NewSkinAnalysisServer(inferenceService, events))

		lis, err := net.Listen("tcp", ":8008")
		if err != nil {
			return fmt.Errorf("failed to listen: %v", err)
		}

		go func() {
			log.Printf("Starting gRPC server on :8008")
			if err := grpcServer.Serve(lis); err != nil {
				errChan <- fmt.Errorf("failed to serve gRPC: %v", err)
			}
		}()

		go func() {
			<-ctx.Done()
			grpcServer.GracefulStop()
		}()

	} else {
		app := fiber.New()
		app.Post("/analyze-skin", api.HandleFileUpload(inferenceService, events))

		go func() {
			log.Printf("Starting Fiber server on :8088")
			if err := app.Listen(":8088"); err != nil {
				errChan <- fmt.Errorf("failed to serve Fiber: %v", err)
			}
		}()

		go func() {
			<-ctx.Done()
			if err := app.Shutdown(); err != nil {
				log.Printf("Error shutting down Fiber server: %v", err)
			}
		}()
	}

	select {
	case err := <-errChan:
		return err
	case <-ctx.Done():
		return nil
	}
}

func main() {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		sig := <-sigChan
		log.Printf("Received signal %v, initiating shutdown", sig)
		cancel()
	}()

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

	sqlDB, err := db.DB()
	if err != nil {
		log.Fatal(err)
	}
	defer func(sqlDB *sql.DB) {
		err := sqlDB.Close()
		if err != nil {
			log.Printf("Failed to close database connection: %v", err)
		}
	}(sqlDB)

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
	chronicEvents := make(chan event.Event, 100)
	startChronicEventProcessor(ctx, repository, chronicEvents)

	inferenceService := service.NewInferenceService(onnxModel, classDict)

	if err := startServers(ctx, inferenceService, chronicEvents, config.RestMode); err != nil {
		log.Fatal(err)
	}

	<-ctx.Done()
	log.Println("Shutting down gracefully...")
}
