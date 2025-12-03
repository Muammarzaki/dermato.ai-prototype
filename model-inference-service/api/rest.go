package api

import (
	"encoding/json"
	"io"
	"model-inference-service/event"
	"model-inference-service/service"
	"time"

	"github.com/gofiber/fiber/v2"
	"github.com/google/uuid"
)

type FileUploadRequest struct {
	UserID    string            `json:"user_id"`
	ImageType string            `json:"image_type"`
	Metadata  map[string]string `json:"metadata"`
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
	Results           []AnalysisResult `json:"results"`
}

func HandleFileUpload(inferenceService *service.InferenceService, event chan event.Event) fiber.Handler {
	return func(c *fiber.Ctx) error {
		file, err := c.FormFile("file")
		if err != nil {
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
				"error": "Failed to get file",
			})
		}

		metadata := make(map[string]string)
		if metadataStr := c.FormValue("metadata"); metadataStr != "" {
			if err := json.Unmarshal([]byte(metadataStr), &metadata); err != nil {
				return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
					"error": "Invalid metadata format",
				})
			}
		}

		_ = FileUploadRequest{
			UserID:    c.FormValue("user_id"),
			ImageType: file.Header.Get("Content-Type"),
			Metadata:  metadata,
		}

		fileContent, err := file.Open()
		if err != nil {
			return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
				"error": "Failed to open file",
			})
		}
		defer fileContent.Close()

		buffer := make([]byte, file.Size)
		if _, err := io.ReadFull(fileContent, buffer); err != nil {
			return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
				"error": "Failed to read file",
			})
		}

		// TODO: Preprocess image buffer ke float32 array
		// preprocessedInput := preprocessImage(buffer)

		// Sekarang bisa gunakan inferenceService yang di-capture dari closure!
		// predictions, err := inferenceService.Infer(preprocessedInput)
		// if err != nil {
		//     return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
		//         "error": "Inference failed",
		//     })
		// }

		response := FileUploadResponse{
			AnalysisID:        uuid.New().String(),
			AnalysisTimestamp: time.Now(),
			Results:           []AnalysisResult{},
		}

		return c.JSON(response)
	}
}
