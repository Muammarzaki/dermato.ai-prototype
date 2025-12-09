package api

import (
	"fmt"
	"io"
	"model-inference-service/event"
	"model-inference-service/service"
	"time"

	pb "model-inference-service/gen"

	"github.com/google/uuid"
	"google.golang.org/protobuf/types/known/timestamppb"
)

type SkinAnalysisServer struct {
	pb.UnimplementedSkinAnalysisServiceServer
	inferenceService *service.InferenceService
	event            chan event.Event
}

func NewSkinAnalysisServer(inferenceService *service.InferenceService, event chan event.Event) *SkinAnalysisServer {
	return &SkinAnalysisServer{
		inferenceService: inferenceService,
		event:            event,
	}
}

func (s *SkinAnalysisServer) AnalyzeSkin(stream pb.SkinAnalysisService_AnalyzeSkinServer) error {
	var imageData []byte
	var _ *pb.ImageInfo

	for {
		req, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}

		switch payload := req.RequestPayload.(type) {
		case *pb.AnalyzeSkinRequest_Info:
			_ = payload.Info
		case *pb.AnalyzeSkinRequest_Chunk:
			imageData = append(imageData, payload.Chunk...)
		}
	}

	// TODO: Preprocess image buffer ke float32 array
	preprocessedInput, err := service.Preprocessing(&imageData)
	if err != nil {
		return fmt.Errorf("failed to preprocess image: %w", err)
	}

	predictionResults, err := s.inferenceService.GetTopKPredictions(preprocessedInput, 1)
	if err != nil {
		return fmt.Errorf("failed to predict: %w", err)
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

	return stream.SendAndClose(response)
}
