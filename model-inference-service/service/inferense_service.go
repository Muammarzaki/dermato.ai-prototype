package service

import (
	"fmt"
	"model-inference-service/model"
	"sync"
)

type InferenceService struct {
	model     *model.ONNXModel
	classDict []string
	mu        sync.Mutex
}

func NewInferenceService(m *model.ONNXModel, c []string) *InferenceService {
	return &InferenceService{
		model:     m,
		classDict: c,
	}
}

func (s *InferenceService) Predict(input []float32) ([]float32, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.model.Predict(input)
}

func (s *InferenceService) PredictClass(input []float32) (int, float32, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.model.PredictClass(input)
}

func (s *InferenceService) GetTopKPredictions(input []float32, k int) ([]PredictionResult, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	indices, probs, err := s.model.GetTopKPredictions(input, k)
	if err != nil {
		return nil, err
	}

	results := make([]PredictionResult, len(indices))
	for i := range indices {
		className, err := s.GetClassName(indices[i])
		if err != nil {
			return nil, err
		}
		results[i] = PredictionResult{
			ClassIndex: indices[i],
			ClassName:  className,
			Confidence: probs[i],
		}
	}

	return results, nil
}

type PredictionResult struct {
	ClassIndex int     `json:"class_index"`
	ClassName  string  `json:"class_name"`
	Confidence float32 `json:"confidence"`
}

func (s *InferenceService) GetClassName(classIndex int) (string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.classDict == nil {
		return "", fmt.Errorf("class dictionary is nil")
	}

	if classIndex >= 0 && classIndex < len(s.classDict) {
		return (s.classDict)[classIndex], nil
	}

	return "", fmt.Errorf("unknown class index: %d", classIndex)
}

func (s *InferenceService) ValidateInput(input []float32) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	expectedSize := s.model.GetExpectedInputSize()
	if len(input) != expectedSize {
		return fmt.Errorf("invalid input size: expected %d, got %d", expectedSize, len(input))
	}
	return nil
}
