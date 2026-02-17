package transport_test

import (
	"context"
	"crypto/sha256"
	"net"
	"os"
	"skin-analyzer-service/internal/service"
	"testing"

	"skin-analyzer-service/internal/event"
	citra "skin-analyzer-service/internal/pb"
	"skin-analyzer-service/internal/transport"

	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/suite"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
	"google.golang.org/grpc/test/bufconn"
)

type MockInference struct {
	mock.Mock
}

func (m *MockInference) GetTopKPredictions(input []float32, k int) ([]service.PredictionResult, error) {
	args := m.Called(input, k)
	return args.Get(0).([]service.PredictionResult), args.Error(1)
}

func (m *MockInference) GetDescription(classIndex int) string {
	args := m.Called(classIndex)
	return args.String(0)
}

func (m *MockInference) GetRecommendation(classIndex int) string {
	args := m.Called(classIndex)
	return args.String(0)
}

type AnalyzeSkinTestSuite struct {
	suite.Suite
	lis           *bufconn.Listener
	server        *grpc.Server
	client        citra.SkinAnalysisServiceClient
	conn          *grpc.ClientConn
	mockInference *MockInference
}

func (suite *AnalyzeSkinTestSuite) SetupTest() {
	suite.lis = bufconn.Listen(1024 * 1024)
	suite.server = grpc.NewServer()

	suite.mockInference = new(MockInference)

	dummyEventChan := make(chan event.Event, 100)

	skinServer := transport.NewSkinAnalysisServer(suite.mockInference, dummyEventChan)
	citra.RegisterSkinAnalysisServiceServer(suite.server, skinServer)

	go func() {
		if err := suite.server.Serve(suite.lis); err != nil {
			panic(err)
		}
	}()

	ctx := context.Background()
	conn, err := grpc.DialContext(ctx, "bufnet",
		grpc.WithContextDialer(func(context.Context, string) (net.Conn, error) {
			return suite.lis.Dial()
		}),
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	suite.Require().NoError(err) // Assertion standar testify

	suite.conn = conn
	suite.client = citra.NewSkinAnalysisServiceClient(conn)
}

func (suite *AnalyzeSkinTestSuite) TearDownTest() {
	suite.conn.Close()
	suite.server.Stop()
	suite.lis.Close()
}

func (suite *AnalyzeSkinTestSuite) TestAnalyzeSkin_MissingChecksum() {
	ctx := context.Background()
	stream, err := suite.client.AnalyzeSkin(ctx)
	suite.Require().NoError(err)

	// Mengirim info tanpa ClientSha256
	err = stream.Send(&citra.AnalyzeSkinRequest{
		RequestPayload: &citra.AnalyzeSkinRequest_Info{
			Info: &citra.ImageInfo{
				UserId:    "tester_1",
				ImageType: "jpg",
			},
		},
	})
	suite.Require().NoError(err)

	stream.Send(&citra.AnalyzeSkinRequest{
		RequestPayload: &citra.AnalyzeSkinRequest_Chunk{Chunk: []byte("dummy_data")},
	})

	_, err = stream.CloseAndRecv()

	suite.Error(err)
	st, _ := status.FromError(err)
	suite.Equal(codes.InvalidArgument, st.Code())
	suite.Contains(st.Message(), "client checksum is required")
}

func (suite *AnalyzeSkinTestSuite) TestAnalyzeSkin_ChecksumMismatch() {
	ctx := context.Background()
	stream, err := suite.client.AnalyzeSkin(ctx)
	suite.Require().NoError(err)

	wrongHash := sha256.Sum256([]byte("data_sudah_berubah_di_jalan"))

	stream.Send(&citra.AnalyzeSkinRequest{
		RequestPayload: &citra.AnalyzeSkinRequest_Info{
			Info: &citra.ImageInfo{ClientSha256: wrongHash[:]},
		},
	})
	stream.Send(&citra.AnalyzeSkinRequest{
		RequestPayload: &citra.AnalyzeSkinRequest_Chunk{Chunk: []byte("dummy_data")},
	})

	_, err = stream.CloseAndRecv()

	suite.Error(err)
	st, _ := status.FromError(err)
	suite.Equal(codes.DataLoss, st.Code())
	suite.Contains(st.Message(), "checksum mismatch")
}

func (suite *AnalyzeSkinTestSuite) TestAnalyzeSkin_EmptyData() {
	ctx := context.Background()
	stream, err := suite.client.AnalyzeSkin(ctx)
	suite.Require().NoError(err)

	validHash := sha256.Sum256([]byte{})

	stream.Send(&citra.AnalyzeSkinRequest{
		RequestPayload: &citra.AnalyzeSkinRequest_Info{
			Info: &citra.ImageInfo{ClientSha256: validHash[:]},
		},
	})

	_, err = stream.CloseAndRecv()

	suite.Error(err)
	st, _ := status.FromError(err)
	suite.Equal(codes.InvalidArgument, st.Code())
	suite.Contains(st.Message(), "empty image data")
}

func (suite *AnalyzeSkinTestSuite) TestAnalyzeSkin_ContractValid() {
	ctx := context.Background()
	stream, err := suite.client.AnalyzeSkin(ctx)
	suite.Require().NoError(err)

	dummyData := []byte("ini_bukan_gambar_asli")
	validHash := sha256.Sum256(dummyData)

	stream.Send(&citra.AnalyzeSkinRequest{
		RequestPayload: &citra.AnalyzeSkinRequest_Info{
			Info: &citra.ImageInfo{
				UserId:       "tester",
				ImageType:    "jpg",
				ClientSha256: validHash[:],
			},
		},
	})
	stream.Send(&citra.AnalyzeSkinRequest{
		RequestPayload: &citra.AnalyzeSkinRequest_Chunk{Chunk: dummyData},
	})

	_, err = stream.CloseAndRecv()

	suite.Error(err)
	st, _ := status.FromError(err)
	suite.Equal(codes.InvalidArgument, st.Code())
	suite.Contains(st.Message(), "failed to preprocess image")
}

func (suite *AnalyzeSkinTestSuite) TestAnalyzeSkin_ContractValid_WithRealImage() {
	ctx := context.Background()
	stream, err := suite.client.AnalyzeSkin(ctx)
	suite.Require().NoError(err)

	imagePath := "../../../bencmark/test-images/sample.jpg"
	imgBytes, err := os.ReadFile(imagePath)
	suite.Require().NoError(err, "Gagal membaca file gambar. Pastikan path gambar benar.")

	validHash := sha256.Sum256(imgBytes)

	// 2. Siapkan Skenario Mock (SANGAT PENTING)
	// Karena gambar valid, server akan meminta hasil prediksi ke mockInference.
	// Kita perintahkan mock untuk merespons dengan prediksi "Eczema"
	suite.mockInference.On("GetTopKPredictions", mock.Anything, 1).Return([]service.PredictionResult{
		{ClassIndex: 2, ClassName: "Eczema", Confidence: 0.98},
	}, nil)

	// Server juga akan meminta deskripsi dan rekomendasi berdasarkan index
	suite.mockInference.On("GetDescription", 2).Return("Peradangan pada kulit.")
	suite.mockInference.On("GetRecommendation", 2).Return("Gunakan krim pelembap.")

	// 3. Kirim Header (Info)
	err = stream.Send(&citra.AnalyzeSkinRequest{
		RequestPayload: &citra.AnalyzeSkinRequest_Info{
			Info: &citra.ImageInfo{
				UserId:       "tester_valid",
				ImageType:    "jpg",
				ClientSha256: validHash[:],
			},
		},
	})
	suite.Require().NoError(err)

	// 4. Kirim Payload (Chunk)
	// Karena ukuran sampel gambar umumnya di bawah 4MB (batas default gRPC),
	// kita bisa mengirimnya utuh dalam 1 chunk di pengujian ini.
	err = stream.Send(&citra.AnalyzeSkinRequest{
		RequestPayload: &citra.AnalyzeSkinRequest_Chunk{Chunk: imgBytes},
	})
	suite.Require().NoError(err)

	// 5. Tutup aliran dan tunggu respons
	res, err := stream.CloseAndRecv()

	// 6. ASSERTION (Validasi Happy Path)
	// Kali ini, kita mengharapkan TIDAK ADA error sama sekali
	suite.Require().NoError(err)
	suite.NotNil(res)

	// Pastikan Server merespons dengan ID dan array hasil yang benar
	suite.NotEmpty(res.AnalysisId)
	suite.Len(res.Results, 1)

	// Validasi apakah respons dari gRPC cocok dengan nilai dari Mock kita
	suite.Equal("Eczema", res.Results[0].Label)
	suite.Equal(float32(0.98), res.Results[0].Confidence)
	suite.Equal("Peradangan pada kulit.", res.Results[0].Description)
	suite.Equal("Gunakan krim pelembap.", res.Results[0].Recommendation)
}

func TestAnalyzeSkinSuite(t *testing.T) {
	suite.Run(t, new(AnalyzeSkinTestSuite))
}
