package api

import (
	"io"
	"model-inference-service/service"
	"time"

	pb "model-inference-service/gen"

	"github.com/google/uuid"
	"google.golang.org/protobuf/types/known/timestamppb"
)

type SkinAnalysisServer struct {
	pb.UnimplementedSkinAnalysisServiceServer
	inferenceService *service.InferenceService
}

func NewSkinAnalysisServer(inferenceService *service.InferenceService) *SkinAnalysisServer {
	return &SkinAnalysisServer{
		inferenceService: inferenceService,
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

	// There should be image processing and model inference
	// For now returning mock response
	response := &pb.AnalyzeSkinResponse{
		AnalysisId:        uuid.New().String(),
		AnalysisTimestamp: timestamppb.New(time.Now()),
		Results: []*pb.AnalysisResult{
			{
				Label:          "normal",
				Confidence:     0.95,
				Description:    "Skin appears normal",
				Recommendation: "Continue with regular skin care routine",
			},
		},
	}

	return stream.SendAndClose(response)
}
