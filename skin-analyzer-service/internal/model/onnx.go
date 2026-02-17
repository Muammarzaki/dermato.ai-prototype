package model

type ONNXModel interface {
	GetTopKPredictions(input []float32, k int) ([]int, []float32, error)
	Close() error
}
