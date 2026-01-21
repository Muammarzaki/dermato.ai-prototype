package api

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"skin-analyzer-service/event"
	"skin-analyzer-service/service"
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

func HandleFileUpload(inferenceService *service.InferenceService, chronicEvent chan event.Event) fiber.Handler {
	return func(c *fiber.Ctx) error {
		file, err := c.FormFile("file")
		if err != nil {
			chronicEvent <- event.Event{
				Status: "error",
				Body:   "Failed to get file: " + err.Error(),
			}
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
				"error":   "Failed to get file",
				"details": err.Error(),
			})
		}

		if file.Size == 0 {
			chronicEvent <- event.Event{
				Status: "error",
				Body:   "Empty file uploaded",
			}
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
				"error": "Empty file uploaded",
			})
		}

		metadata := make(map[string]string)
		if metadataStr := c.FormValue("metadata"); metadataStr != "" {
			if err := json.Unmarshal([]byte(metadataStr), &metadata); err != nil {
				chronicEvent <- event.Event{
					Status: "error",
					Body:   "Invalid metadata format: " + err.Error(),
				}
				return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
					"error":   "Invalid metadata format",
					"details": err.Error(),
				})
			}
		}

		uploadRequest := FileUploadRequest{
			UserID:    c.FormValue("user_id"),
			ImageType: file.Header.Get("Content-Type"),
			Metadata:  metadata,
		}

		if uploadRequest.UserID == "" {
			chronicEvent <- event.Event{
				Status: "error",
				Body:   "User ID is required",
			}
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
				"error": "User ID is required",
			})
		}

		fileContent, err := file.Open()
		if err != nil {
			chronicEvent <- event.Event{
				Status: "error",
				Body:   "Failed to open file: " + err.Error(),
			}
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
			chronicEvent <- event.Event{
				Status: "error",
				Body:   "Failed to read file: " + err.Error(),
			}
			return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
				"error":   "Failed to read file",
				"details": err.Error(),
			})
		}

		serverChecksum := sha256.Sum256(buffer)

		clientChecksumHex := c.FormValue("client_sha256")
		if clientChecksumHex == "" {
			chronicEvent <- event.Event{
				Status: "error",
				Body:   "client checksum is required for data integrity validation",
			}
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
				"error": "client checksum is required for data integrity validation",
			})
		}

		clientChecksum, err := hex.DecodeString(clientChecksumHex)
		if err != nil {
			chronicEvent <- event.Event{
				Status: "error",
				Body:   "invalid checksum format: " + err.Error(),
			}
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
				"error":   "invalid checksum format",
				"details": err.Error(),
			})
		}

		if len(clientChecksum) != sha256.Size {
			chronicEvent <- event.Event{
				Status: "error",
				Body:   "invalid sha256 checksum length",
			}
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
			chronicEvent <- event.Event{
				Status: "error",
				Body:   message,
			}
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
				"error":   message,
				"details": "Client and server checksums do not match",
			})
		}

		preprocessedInput, err := service.Preprocessing(&buffer)
		if err != nil {
			chronicEvent <- event.Event{
				Status: "error",
				Body:   "Failed to preprocess image: " + err.Error(),
			}
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
				"error":   "Failed to preprocess image",
				"details": err.Error(),
			})
		}

		predictionResults, err := inferenceService.GetTopKPredictions(preprocessedInput, 1)
		if err != nil {
			chronicEvent <- event.Event{
				Status: "error",
				Body:   "Inference failed: " + err.Error(),
			}
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
			chronicEvent <- event.Event{
				Status: "error",
				Body:   "Failed to marshal response: " + err.Error(),
			}
			return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
				"error":   "Internal server error",
				"details": err.Error(),
			})
		}

		chronicEvent <- event.Event{
			Status: "success",
			Body:   string(responseJSON),
		}

		return c.JSON(response)
	}
}
