package api

import (
	"bytes"
	"fmt"
	"io"
	"skin-analyzer-service/event"
	"skin-analyzer-service/service"
	"time"

	pb "skin-analyzer-service/gen"

	"github.com/google/uuid"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/timestamppb"
)

type SkinAnalysisServer struct {
	pb.UnimplementedSkinAnalysisServiceServer
	inferenceService *service.InferenceService
	chronicEvent     chan event.Event
}

func NewSkinAnalysisServer(inferenceService *service.InferenceService, event chan event.Event) *SkinAnalysisServer {
	return &SkinAnalysisServer{
		inferenceService: inferenceService,
		chronicEvent:     event,
	}
}

func (s *SkinAnalysisServer) AnalyzeSkin(stream pb.SkinAnalysisService_AnalyzeSkinServer) error {
	var imageBuffer bytes.Buffer
	var _ *pb.ImageInfo

	for {
		req, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			s.chronicEvent <- event.Event{
				Status: "error",
				Body:   "Error receiving stream: " + err.Error(),
			}
			return status.Errorf(codes.Internal, "failed to receive stream: %v", err)
		}

		switch payload := req.RequestPayload.(type) {
		case *pb.AnalyzeSkinRequest_Info:
			_ = payload.Info
		case *pb.AnalyzeSkinRequest_Chunk:
			imageBuffer.Write(payload.Chunk)
		}
	}

	if imageBuffer.Len() == 0 {
		s.chronicEvent <- event.Event{
			Status: "error",
			Body:   "Empty image data received",
		}
		return status.Error(codes.InvalidArgument, "empty image data")
	}
	finalBytes := imageBuffer.Bytes()
	preprocessedInput, err := service.Preprocessing(&finalBytes)
	if err != nil {
		s.chronicEvent <- event.Event{
			Status: "error",
			Body:   fmt.Sprintf("Failed to preprocess image: %v", err),
		}
		return status.Errorf(codes.InvalidArgument, "failed to preprocess image: %v", err)
	}

	predictionResults, err := s.inferenceService.GetTopKPredictions(preprocessedInput, 1)
	if err != nil {
		s.chronicEvent <- event.Event{
			Status: "error",
			Body:   fmt.Sprintf("Prediction failed: %v", err),
		}
		return status.Errorf(codes.Internal, "failed to predict: %v", err)
	}

	var analysisResults []*pb.AnalysisResult

	for _, predictionResult := range predictionResults {
		analysisResults = append(analysisResults, &pb.AnalysisResult{
			Label:          predictionResult.ClassName,
			Confidence:     predictionResult.Confidence,
			Description:    s.inferenceService.GetDescription(predictionResult.ClassIndex),
			Recommendation: s.inferenceService.GetRecommendation(predictionResult.ClassIndex),
		})
	}

	response := &pb.AnalyzeSkinResponse{
		AnalysisId:        uuid.New().String(),
		AnalysisTimestamp: timestamppb.New(time.Now()),
		Results:           analysisResults,
	}

	s.chronicEvent <- event.Event{
		Status: "success",
		Body:   fmt.Sprintf("Analysis completed successfully for ID: %s", response.AnalysisId),
	}

	if err := stream.SendAndClose(response); err != nil {
		s.chronicEvent <- event.Event{
			Status: "error",
			Body:   fmt.Sprintf("Failed to send response: %v", err),
		}
		return status.Errorf(codes.Internal, "failed to send response: %v", err)
	}

	return nil
}
