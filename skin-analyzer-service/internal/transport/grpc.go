package transport

import (
	"bytes"
	"crypto/sha256"
	"fmt"
	"io"
	"log"
	"skin-analyzer-service/internal/event"
	"skin-analyzer-service/internal/pb"
	"skin-analyzer-service/internal/service"
	"strconv"
	"time"

	"github.com/google/uuid"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/timestamppb"
)

// imageProcessingSemaphore limits concurrent image decodes across the transport package
// This prevents OOM (Out of Memory) crashes during heavy benchmark loads.
var imageProcessingSemaphore = make(chan struct{}, 4)

type SkinAnalysisServer struct {
	citra.UnimplementedSkinAnalysisServiceServer
	inferenceService service.Inference
	chronicEvent     chan event.Event
}

func NewSkinAnalysisServer(inferenceService service.Inference, event chan event.Event) *SkinAnalysisServer {
	return &SkinAnalysisServer{
		inferenceService: inferenceService,
		chronicEvent:     event,
	}
}

// emitEvent safely sends events without blocking the goroutine if the channel is full
func (s *SkinAnalysisServer) emitEvent(evtStatus, body string) {
	select {
	case s.chronicEvent <- event.Event{Status: evtStatus, Body: body}:
		// Event successfully sent
	default:
		// Channel is full, drop the event to prevent goroutine memory leaks
		log.Printf("Warning: chronicEvent channel full, dropping gRPC event: %s", body)
	}
}

func (s *SkinAnalysisServer) AnalyzeSkin(stream citra.SkinAnalysisService_AnalyzeSkinServer) error {
	var imageBuffer bytes.Buffer
	var imageInfo *citra.ImageInfo

	for {
		req, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			s.emitEvent("error", "Error receiving stream: "+err.Error())
			return status.Errorf(codes.Internal, "failed to receive stream: %v", err)
		}

		switch payload := req.RequestPayload.(type) {
		case *citra.AnalyzeSkinRequest_Info:
			imageInfo = payload.Info
			if sizeStr, ok := imageInfo.Metadata["file_size"]; ok {
				if size, err := strconv.ParseInt(sizeStr, 10, 64); err == nil {
					imageBuffer.Grow(int(size))
				}
			}
		case *citra.AnalyzeSkinRequest_Chunk:
			imageBuffer.Write(payload.Chunk)
		}
	}

	if imageBuffer.Len() == 0 {
		s.emitEvent("error", "Empty image data received")
		return status.Error(codes.InvalidArgument, "empty image data")
	}
	finalBytes := imageBuffer.Bytes()
	serverChecksum := sha256.Sum256(finalBytes)

	if imageInfo == nil || len(imageInfo.ClientSha256) == 0 {
		s.emitEvent("error", "client checksum is required for data integrity validation")
		return status.Error(
			codes.InvalidArgument,
			"client checksum is required for data integrity validation",
		)
	}

	if len(imageInfo.ClientSha256) != sha256.Size {
		s.emitEvent("error", "invalid sha256 checksum length")
		return status.Error(
			codes.InvalidArgument,
			"invalid sha256 checksum length",
		)
	}

	if !bytes.Equal(serverChecksum[:], imageInfo.ClientSha256) {
		message := fmt.Sprintf(
			"checksum mismatch: client=%x server=%x",
			imageInfo.ClientSha256,
			serverChecksum,
		)
		s.emitEvent("error", message)
		return status.Error(codes.DataLoss, message)
	}

	// Wait for permission to process the image to prevent RAM spikes
	imageProcessingSemaphore <- struct{}{}
	preprocessedInput, err := service.Preprocessing(&finalBytes)
	<-imageProcessingSemaphore // Release the token immediately after decoding

	if err != nil {
		s.emitEvent("error", fmt.Sprintf("Failed to preprocess image: %v", err))
		return status.Errorf(codes.InvalidArgument, "failed to preprocess image: %v", err)
	}

	predictionResults, err := s.inferenceService.GetTopKPredictions(preprocessedInput, 1)
	if err != nil {
		s.emitEvent("error", fmt.Sprintf("Prediction failed: %v", err))
		return status.Errorf(codes.Internal, "failed to predict: %v", err)
	}

	var analysisResults []*citra.AnalysisResult

	for _, predictionResult := range predictionResults {
		analysisResults = append(analysisResults, &citra.AnalysisResult{
			Label:          predictionResult.ClassName,
			Confidence:     predictionResult.Confidence,
			Description:    s.inferenceService.GetDescription(predictionResult.ClassIndex),
			Recommendation: s.inferenceService.GetRecommendation(predictionResult.ClassIndex),
		})
	}

	response := &citra.AnalyzeSkinResponse{
		AnalysisId:        uuid.New().String(),
		AnalysisTimestamp: timestamppb.New(time.Now()),
		ServerSha256:      serverChecksum[:],
		Results:           analysisResults,
	}

	s.emitEvent("success", fmt.Sprintf("Analysis completed successfully for ID: %s", response.AnalysisId))

	if err := stream.SendAndClose(response); err != nil {
		s.emitEvent("error", fmt.Sprintf("Failed to send response: %v", err))
		return status.Errorf(codes.Internal, "failed to send response: %v", err)
	}

	return nil
}
