package transport

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"mime/multipart"
	"skin-analyzer-service/internal/event"
	"skin-analyzer-service/internal/service"
	"time"

	"github.com/gofiber/fiber/v2"
	"github.com/google/uuid"
)

type FileUploadRequest struct {
	UserID       string            `json:"user_id"`
	ImageType    string            `json:"image_type"`
	ClientSha256 string            `json:"client_sha256"`
	Metadata     map[string]string `json:"metadata"`
}

type AnalysisResult struct {
	Label          string  `json:"label"`
	Confidence     float32 `json:"confidence"`
	Description    string  `json:"description"`
	Recommendation string  `json:"recommendation"`
}

type FileUploadResponse struct {
	AnalysisID        string           `json:"analysis_id"`
	AnalysisTimestamp time.Time        `json:"analysis_timestamp"`
	ServerSha256      string           `json:"server_sha256"`
	Results           []AnalysisResult `json:"results"`
}

func emitRestEvent(ch chan<- event.Event, evtStatus, body string) {
	select {
	case ch <- event.Event{Status: evtStatus, Body: body}:
	default:
		log.Printf("Warning: chronicEvent channel full, dropping REST event: %s", body)
	}
}

func HandleFileUpload(inferenceService service.Inference, chronicEvent chan event.Event) fiber.Handler {
	return func(c *fiber.Ctx) error {
		file, err := c.FormFile("file")
		if err != nil {
			emitRestEvent(chronicEvent, "error", "Failed to get file: "+err.Error())
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
				"error":   "Failed to get file",
				"details": err.Error(),
			})
		}

		if file.Size == 0 {
			emitRestEvent(chronicEvent, "error", "Empty file uploaded")
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
				"error": "Empty file uploaded",
			})
		}

		metadata := make(map[string]string)
		if metadataStr := c.FormValue("metadata"); metadataStr != "" {
			if err := json.Unmarshal([]byte(metadataStr), &metadata); err != nil {
				emitRestEvent(chronicEvent, "error", "Invalid metadata format: "+err.Error())
				return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
					"error":   "Invalid metadata format",
					"details": err.Error(),
				})
			}
		}

		uploadRequest := FileUploadRequest{
			UserID:       c.FormValue("user_id"),
			ImageType:    file.Header.Get("Content-Type"),
			Metadata:     metadata,
			ClientSha256: c.FormValue("client_sha256"),
		}

		if uploadRequest.UserID == "" {
			emitRestEvent(chronicEvent, "error", "User ID is required")
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
				"error": "User ID is required",
			})
		}

		fileContent, err := file.Open()
		if err != nil {
			emitRestEvent(chronicEvent, "error", "Failed to open file: "+err.Error())
			return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
				"error":   "Failed to open file",
				"details": err.Error(),
			})
		}
		defer func(fileContent multipart.File) {
			_ = fileContent.Close()
		}(fileContent)

		buffer := make([]byte, file.Size)
		if _, err := io.ReadFull(fileContent, buffer); err != nil {
			emitRestEvent(chronicEvent, "error", "Failed to read file: "+err.Error())
			return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
				"error":   "Failed to read file",
				"details": err.Error(),
			})
		}

		serverChecksum := sha256.Sum256(buffer)

		if uploadRequest.ClientSha256 == "" {
			emitRestEvent(chronicEvent, "error", "client checksum is required for data integrity validation")
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
				"error": "client checksum is required for data integrity validation",
			})
		}

		clientChecksum, err := hex.DecodeString(uploadRequest.ClientSha256)
		if err != nil {
			emitRestEvent(chronicEvent, "error", "invalid checksum format: "+err.Error())
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
				"error":   "invalid checksum format",
				"details": err.Error(),
			})
		}

		if len(clientChecksum) != sha256.Size {
			emitRestEvent(chronicEvent, "error", "invalid sha256 checksum length")
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
				"error": "invalid sha256 checksum length",
			})
		}

		if !bytes.Equal(serverChecksum[:], clientChecksum) {
			message := fmt.Sprintf(
				"checksum mismatch: client=%x server=%x",
				clientChecksum,
				serverChecksum,
			)
			emitRestEvent(chronicEvent, "error", message)
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
				"error":   message,
				"details": "Client and server checksums do not match",
			})
		}

		// Wait for permission to process the image to prevent RAM spikes.
		// Uses the shared package-level semaphore defined in grpc.go.
		imageProcessingSemaphore <- struct{}{}
		preprocessedInput, err := service.Preprocessing(&buffer)
		<-imageProcessingSemaphore // Release the token immediately after decoding

		if err != nil {
			emitRestEvent(chronicEvent, "error", "Failed to preprocess image: "+err.Error())
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
				"error":   "Failed to preprocess image",
				"details": err.Error(),
			})
		}

		predictionResults, err := inferenceService.GetTopKPredictions(preprocessedInput, 1)
		if err != nil {
			emitRestEvent(chronicEvent, "error", "Inference failed: "+err.Error())
			return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
				"error":   "Inference failed",
				"details": err.Error(),
			})
		}

		var analysisResults []AnalysisResult

		for _, predictionResult := range predictionResults {
			analysisResults = append(analysisResults, AnalysisResult{
				Label:          predictionResult.ClassName,
				Confidence:     predictionResult.Confidence,
				Description:    inferenceService.GetDescription(predictionResult.ClassIndex),
				Recommendation: inferenceService.GetRecommendation(predictionResult.ClassIndex),
			})
		}

		response := FileUploadResponse{
			AnalysisID:        uuid.New().String(),
			AnalysisTimestamp: time.Now(),
			ServerSha256:      hex.EncodeToString(serverChecksum[:]),
			Results:           analysisResults,
		}

		responseJSON, err := json.Marshal(response)
		if err != nil {
			emitRestEvent(chronicEvent, "error", "Failed to marshal response: "+err.Error())
			return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
				"error":   "Internal server error",
				"details": err.Error(),
			})
		}

		emitRestEvent(chronicEvent, "success", string(responseJSON))

		return c.JSON(response)
	}
}
