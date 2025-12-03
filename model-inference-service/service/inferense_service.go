package service

import (
	"fmt"
	"model-inference-service/model"
)

type InferenceService struct {
	model *model.ONNXModel
}

func NewInferenceService(m *model.ONNXModel) *InferenceService {
	return &InferenceService{
		model: m,
	}
}

func (s *InferenceService) Infer(input []float32) ([]float32, error) {
	return s.model.Predict(input)
}

func (s *InferenceService) PredictClass(input []float32) (int, float32, error) {
	return s.model.PredictClass(input)
}

func (s *InferenceService) GetTopKPredictions(input []float32, k int) ([]PredictionResult, error) {
	indices, probs, err := s.model.GetTopKPredictions(input, k)
	if err != nil {
		return nil, err
	}

	results := make([]PredictionResult, len(indices))
	for i := range indices {
		className, err := GetClassName(indices[i])
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

func GetClassName(classIndex int) (string, error) {
	classNames := []string{
		"Normal",
		"Acne",
		"Chickenpox",
		"Eczema",
		"Psoriasis",
		"Rosacea",
		"Vitiligo",
		"Melanoma",
	}

	if classIndex >= 0 && classIndex < len(classNames) {
		return classNames[classIndex], nil
	}

	return "", fmt.Errorf("unknown class index: %d", classIndex)
}

func (s *InferenceService) ValidateInput(input []float32) error {
	expectedSize := s.model.GetExpectedInputSize()
	if len(input) != expectedSize {
		return fmt.Errorf("invalid input size: expected %d, got %d", expectedSize, len(input))
	}
	return nil
}
