package service

type Inference interface {
	GetTopKPredictions(input []float32, k int) ([]PredictionResult, error)
	GetDescription(classIndex int) string
	GetRecommendation(classIndex int) string
}
