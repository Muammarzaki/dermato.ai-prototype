package service

import (
	"model-inference-service/model"
)

type InferenceService struct {
	Model *model.ONNXModel
}

func NewInferenceService(m *model.ONNXModel) *InferenceService {
	return &InferenceService{Model: m}
}

func (s *InferenceService) Infer(input []float32) ([]float32, error) {
	return s.Model.Predict(input)
}
