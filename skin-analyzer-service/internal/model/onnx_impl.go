package model

import (
	"fmt"

	ort "github.com/yalue/onnxruntime_go"
)

// ONNXModelImpl represents a wrapper for ONNX Runtime model operations
// Designed for image classification with 8 classes
type ONNXModelImpl struct {
	session      *ort.AdvancedSession
	inputTensor  *ort.Tensor[float32]
	outputTensor *ort.Tensor[float32]
	inputShape   []int64
	outputShape  []int64
}

// NewONNXModel creates a new instance of ONNX model
// This is specifically configured for your TensorFlow.js converted model:
// - Input: "input_6" with shape [1, 180, 180, 3]
// - Output: "dense_11" with shape [1, 8]
//
// Parameters:
//   - path: path to the .onnx model file
//
// Returns:
//   - *ONNXModelImpl: pointer to the created ONNX model
//   - error: error if any occurs during initialization
func NewONNXModel(path string) (*ONNXModelImpl, error) {
	if err := ort.InitializeEnvironment(); err != nil {
		return nil, fmt.Errorf("failed to initialize ONNX runtime: %w", err)
	}

	inputNodeNames := []string{"input_6"}
	outputNodeNames := []string{"dense_11"}

	inputShape := []int64{1, 180, 180, 3} // NHWC
	outputShape := []int64{1, 8}

	options, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("failed to create session options: %w", err)
	}
	defer func(options *ort.SessionOptions) {
		_ = options.Destroy()
	}(options)

	totalInputElements := int64(1)
	for _, d := range inputShape {
		totalInputElements *= d
	}

	totalOutputElements := int64(1)
	for _, d := range outputShape {
		totalOutputElements *= d
	}

	inputTensor, err := ort.NewTensor(inputShape, make([]float32, totalInputElements))
	if err != nil {
		return nil, fmt.Errorf("failed to create input tensor: %w", err)
	}

	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		_ = inputTensor.Destroy()
		return nil, fmt.Errorf("failed to create output tensor: %w", err)
	}

	session, err := ort.NewAdvancedSession(
		path,
		inputNodeNames,
		outputNodeNames,
		[]ort.ArbitraryTensor{inputTensor},
		[]ort.ArbitraryTensor{outputTensor},
		options,
	)

	if err != nil {
		_ = inputTensor.Destroy()
		_ = outputTensor.Destroy()
		return nil, fmt.Errorf(
			"failed to create session (check input/output node names): %w",
			err,
		)
	}

	return &ONNXModelImpl{
		session:      session,
		inputTensor:  inputTensor,
		outputTensor: outputTensor,
		inputShape:   inputShape,
		outputShape:  outputShape,
	}, nil
}

// Predict performs inference with the given input image data
// Input should be a flattened array of size 97,200 (1*180*180*3)
// in format [batch, height, width, channels]
//
// Parameters:
//   - input: preprocessed image data as float32 slice (size: 97,200)
//     Values should be normalized (typically 0-1 or -1 to 1)
//
// Returns:
//   - []float32: prediction probabilities for 8 classes (size: 8)
//   - error: error if any occurs during inference
func (m *ONNXModelImpl) Predict(input []float32) ([]float32, error) {
	// Validate input size
	inputData := m.inputTensor.GetData()
	expectedSize := len(inputData)

	if len(input) != expectedSize {
		return nil, fmt.Errorf("input size mismatch: expected %d (1*180*180*3), got %d", expectedSize, len(input))
	}

	// Copy input data to tensor
	copy(inputData, input)

	// Run inference
	err := m.session.Run()
	if err != nil {
		return nil, fmt.Errorf("failed to run inference: %w", err)
	}

	// Get output (8 class probabilities)
	outputData := m.outputTensor.GetData()
	result := make([]float32, len(outputData))
	copy(result, outputData)

	return result, nil
}

// GetTopKPredictions returns top K predictions with their indices and probabilities
//
// Parameters:
//   - input: preprocessed image data as float32 slice
//   - k: number of top predictions to return (max 8)
//
// Returns:
//   - []int: class indices sorted by probability
//   - []float32: corresponding probabilities
//   - error: error if any occurs during inference
func (m *ONNXModelImpl) GetTopKPredictions(input []float32, k int) ([]int, []float32, error) {
	if k > 8 {
		k = 8
	}
	if k < 1 {
		k = 1
	}

	probabilities, err := m.Predict(input)
	if err != nil {
		return nil, nil, err
	}

	type pred struct {
		idx  int
		prob float32
	}

	preds := make([]pred, len(probabilities))
	for i, p := range probabilities {
		preds[i] = pred{idx: i, prob: p}
	}

	for i := 0; i < k; i++ {
		maxIdx := i
		for j := i + 1; j < len(preds); j++ {
			if preds[j].prob > preds[maxIdx].prob {
				maxIdx = j
			}
		}
		preds[i], preds[maxIdx] = preds[maxIdx], preds[i]
	}

	topIndices := make([]int, k)
	topProbs := make([]float32, k)
	for i := 0; i < k; i++ {
		topIndices[i] = preds[i].idx
		topProbs[i] = preds[i].prob
	}

	return topIndices, topProbs, nil
}

// Close cleans up the resources used by the model
//
// Returns:
//   - error: error if any occurs during cleanup
func (m *ONNXModelImpl) Close() error {
	if m.inputTensor != nil {
		_ = m.inputTensor.Destroy()
	}
	if m.outputTensor != nil {
		_ = m.outputTensor.Destroy()
	}
	if m.session != nil {
		_ = m.session.Destroy()
	}
	return ort.DestroyEnvironment()
}
