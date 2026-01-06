package service

import (
	"bytes"
	"fmt"
	"image"
	"log"
	"skin-analyzer-service/model"
	"sync"

	_ "image/jpeg"
	_ "image/png"

	"golang.org/x/image/draw"
)

type InferenceService struct {
	model     *model.ONNXModel
	classDict []DiseaseClass
	mu        sync.Mutex
}

func NewInferenceService(m *model.ONNXModel, c []DiseaseClass) *InferenceService {
	log.Println("Initializing InferenceService")
	return &InferenceService{
		model:     m,
		classDict: c,
	}
}

func (s *InferenceService) Predict(input []float32) ([]float32, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	log.Println("Making prediction with input of length:", len(input))
	return s.model.Predict(input)
}

func (s *InferenceService) PredictClass(input []float32) (int, float32, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	log.Println("Predicting class with input of length:", len(input))
	return s.model.PredictClass(input)
}

func (s *InferenceService) GetTopKPredictions(input []float32, k int) ([]PredictionResult, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	log.Printf("Getting top %d predictions", k)
	indices, probs, err := s.model.GetTopKPredictions(input, k)
	if err != nil {
		log.Printf("Error getting top K predictions: %v", err)
		return nil, err
	}

	results := make([]PredictionResult, len(indices))
	for i := range indices {
		classLabel, err := s.GetClassName(indices[i])
		if err != nil {
			log.Printf("Error getting class name for index %d: %v", indices[i], err)
			return nil, err
		}
		results[i] = PredictionResult{
			ClassIndex: indices[i],
			ClassName:  classLabel.Name,
			Confidence: probs[i],
		}
	}

	log.Printf("Found %d predictions", len(results))
	return results, nil
}

type PredictionResult struct {
	ClassIndex int     `json:"class_index"`
	ClassName  string  `json:"class_name"`
	Confidence float32 `json:"confidence"`
}

func (s *InferenceService) GetClassName(classIndex int) (DiseaseClass, error) {

	log.Printf("Getting class name for index: %d", classIndex)
	if s.classDict == nil {
		return DiseaseClass{}, fmt.Errorf("class dictionary is nil")
	}

	if classIndex >= 0 && classIndex < len(s.classDict) {
		return (s.classDict)[classIndex], nil
	}

	return DiseaseClass{}, fmt.Errorf("unknown class index: %d", classIndex)
}

func (s *InferenceService) ValidateInput(input []float32) error {

	expectedSize := s.model.GetExpectedInputSize()
	log.Printf("Validating input size: expected %d, got %d", expectedSize, len(input))
	if len(input) != expectedSize {
		return fmt.Errorf("invalid input size: expected %d, got %d", expectedSize, len(input))
	}
	return nil
}

func Preprocessing(buffer *[]byte) ([]float32, error) {
	log.Println("Starting image preprocessing")
	img, _, err := image.Decode(bytes.NewReader(*buffer))
	if err != nil {
		log.Printf("Error decoding image: %v", err)
		return []float32{}, fmt.Errorf("error decoding image: %v", err)
	}
	log.Println("Image decoded successfully")
	return ImageToNHWCTensor(img), nil
}

type DiseaseClass struct {
	Name           string `json:"name"`
	Description    string `json:"description"`
	Recommendation string `json:"recommendation"`
}

func (s *InferenceService) GetDescription(index int) string {
	log.Printf("Getting description for index: %d", index)
	return s.classDict[index].Description
}
func (s *InferenceService) GetRecommendation(index int) string {
	log.Printf("Getting recommendation for index: %d", index)
	return s.classDict[index].Recommendation
}

func ResizeImage(img image.Image, targetSize int) image.Image {
	log.Printf("Resizing image to %dx%d", targetSize, targetSize)
	resized := image.NewRGBA(image.Rect(0, 0, targetSize, targetSize))

	draw.CatmullRom.Scale(
		resized,
		resized.Bounds(),
		img,
		img.Bounds(),
		draw.Src,
		nil,
	)

	return resized
}

func ImageToNHWCTensor(img image.Image) []float32 {
	log.Println("Converting image to NHWC tensor")
	final := ResizeImage(img, 180)

	data := make([]float32, 1*180*180*3)
	idx := 0

	for y := 0; y < 180; y++ {
		for x := 0; x < 180; x++ {
			r, g, b, _ := final.At(x, y).RGBA()

			data[idx] = float32(r>>8) / 255.0
			data[idx+1] = float32(g>>8) / 255.0
			data[idx+2] = float32(b>>8) / 255.0

			idx += 3
		}
	}

	log.Println("Image conversion completed")
	return data
}
