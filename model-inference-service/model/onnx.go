package model

import (
	"fmt"

	ort "github.com/yalue/onnxruntime_go"
)

// ONNXModel represents a wrapper for ONNX Runtime model operations
// Designed for image classification with 8 classes (from TensorFlow.js converted model)
type ONNXModel struct {
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
//   - *ONNXModel: pointer to the created ONNX model
//   - error: error if any occurs during initialization
func NewONNXModel(path string) (*ONNXModel, error) {
	// Initialize ONNX Runtime environment
	if err := ort.InitializeEnvironment(); err != nil {
		return nil, fmt.Errorf("failed to initialize ONNX runtime: %w", err)
	}

	// Expected node names for TensorFlow.js ONNX conversion
	inputNodeNames := []string{"input_6"}   // adjust if needed
	outputNodeNames := []string{"dense_11"} // adjust if needed

	// Expected shapes
	inputShape := []int64{1, 180, 180, 3} // NHWC
	outputShape := []int64{1, 8}

	// Session options
	options, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("failed to create session options: %w", err)
	}
	defer options.Destroy()

	// Calculate tensor sizes
	totalInputElements := int64(1)
	for _, d := range inputShape {
		totalInputElements *= d
	}

	totalOutputElements := int64(1)
	for _, d := range outputShape {
		totalOutputElements *= d
	}

	// Create tensors
	inputTensor, err := ort.NewTensor(inputShape, make([]float32, totalInputElements))
	if err != nil {
		return nil, fmt.Errorf("failed to create input tensor: %w", err)
	}

	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		inputTensor.Destroy()
		return nil, fmt.Errorf("failed to create output tensor: %w", err)
	}

	// Prepare session
	session, err := ort.NewAdvancedSession(
		path,
		inputNodeNames,
		outputNodeNames,
		[]ort.ArbitraryTensor{inputTensor},
		[]ort.ArbitraryTensor{outputTensor},
		options,
	)
	if err != nil {
		inputTensor.Destroy()
		outputTensor.Destroy()
		return nil, fmt.Errorf(
			"failed to create session (check input/output node names): %w",
			err,
		)
	}

	return &ONNXModel{
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
func (m *ONNXModel) Predict(input []float32) ([]float32, error) {
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

// PredictClass performs inference and returns the predicted class and confidence
//
// Parameters:
//   - input: preprocessed image data as float32 slice
//
// Returns:
//   - int: predicted class index (0-7)
//   - float32: confidence score (0-1)
//   - error: error if any occurs during inference
func (m *ONNXModel) PredictClass(input []float32) (int, float32, error) {
	// Get all class probabilities
	probabilities, err := m.Predict(input)
	if err != nil {
		return -1, 0, err
	}

	// Find the class with the highest probability
	maxIdx := 0
	maxProb := probabilities[0]

	for i := 1; i < len(probabilities); i++ {
		if probabilities[i] > maxProb {
			maxProb = probabilities[i]
			maxIdx = i
		}
	}

	return maxIdx, maxProb, nil
}

// PredictWithShape performs inference and returns results with shape information
//
// Parameters:
//   - input: preprocessed image data as float32 slice
//
// Returns:
//   - []float32: prediction probabilities
//   - []int64: shape of the output [1, 8]
//   - error: error if any occurs during inference
func (m *ONNXModel) PredictWithShape(input []float32) ([]float32, []int64, error) {
	result, err := m.Predict(input)
	if err != nil {
		return nil, nil, err
	}

	shape := m.outputTensor.GetShape()
	return result, shape, nil
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
func (m *ONNXModel) GetTopKPredictions(input []float32, k int) ([]int, []float32, error) {
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

	// Create pairs of (indexes, probability)
	type pred struct {
		idx  int
		prob float32
	}

	preds := make([]pred, len(probabilities))
	for i, p := range probabilities {
		preds[i] = pred{idx: i, prob: p}
	}

	// Simple selection sort for top K
	for i := 0; i < k; i++ {
		maxIdx := i
		for j := i + 1; j < len(preds); j++ {
			if preds[j].prob > preds[maxIdx].prob {
				maxIdx = j
			}
		}
		preds[i], preds[maxIdx] = preds[maxIdx], preds[i]
	}

	// Extract top K
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
func (m *ONNXModel) Close() error {
	if m.inputTensor != nil {
		m.inputTensor.Destroy()
	}
	if m.outputTensor != nil {
		m.outputTensor.Destroy()
	}
	if m.session != nil {
		m.session.Destroy()
	}

	return ort.DestroyEnvironment()
}

// GetInputShape returns the shape of the input tensor [1, 180, 180, 3]
//
// Returns:
//   - []int64: shape of the input tensor
func (m *ONNXModel) GetInputShape() []int64 {
	return m.inputShape
}

// GetOutputShape returns the shape of the output tensor [1, 8]
//
// Returns:
//   - []int64: shape of the output tensor
func (m *ONNXModel) GetOutputShape() []int64 {
	return m.outputShape
}

// GetExpectedInputSize returns the expected total number of input elements (97,200)
//
// Returns:
//   - int: total number of input elements
func (m *ONNXModel) GetExpectedInputSize() int {
	size := 1
	for _, dim := range m.inputShape {
		size *= int(dim)
	}
	return size
}

// GetNumClasses returns the number of output classes (8)
//
// Returns:
//   - int: number of classes
func (m *ONNXModel) GetNumClasses() int {
	return int(m.outputShape[1])
}
