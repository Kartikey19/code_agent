package rag

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// Embedder generates vector embeddings for text
type Embedder interface {
	Embed(text string) ([]float32, error)
	EmbedBatch(texts []string) ([][]float32, error)
	Dimension() int
	Model() string
}

// OllamaEmbedder implements Embedder using Ollama API
type OllamaEmbedder struct {
	baseURL    string
	model      string
	dimensions int
	httpClient *http.Client
}

type ollamaEmbedRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
}

type ollamaEmbedResponse struct {
	Embedding []float32 `json:"embedding"`
}

// NewOllamaEmbedder creates a new Ollama embedder
func NewOllamaEmbedder(model string) *OllamaEmbedder {
	if model == "" {
		model = "nomic-embed-text" // Default model
	}

	dimensions := 768 // nomic-embed-text dimensions
	if model == "mxbai-embed-large" {
		dimensions = 1024
	}

	return &OllamaEmbedder{
		baseURL:    "http://localhost:11434",
		model:      model,
		dimensions: dimensions,
		httpClient: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}

func (e *OllamaEmbedder) Embed(text string) ([]float32, error) {
	reqBody := ollamaEmbedRequest{
		Model:  e.model,
		Prompt: text,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	resp, err := e.httpClient.Post(
		e.baseURL+"/api/embeddings",
		"application/json",
		bytes.NewBuffer(jsonData),
	)
	if err != nil {
		return nil, fmt.Errorf("ollama request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("ollama returned status %d: %s", resp.StatusCode, string(body))
	}

	var result ollamaEmbedResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	if len(result.Embedding) == 0 {
		return nil, fmt.Errorf("empty embedding returned")
	}

	return result.Embedding, nil
}

func (e *OllamaEmbedder) EmbedBatch(texts []string) ([][]float32, error) {
	embeddings := make([][]float32, len(texts))

	// TODO: Implement true batch processing if Ollama supports it
	// For now, process sequentially
	for i, text := range texts {
		embedding, err := e.Embed(text)
		if err != nil {
			return nil, fmt.Errorf("failed to embed text %d: %w", i, err)
		}
		embeddings[i] = embedding
	}

	return embeddings, nil
}

func (e *OllamaEmbedder) Dimension() int {
	return e.dimensions
}

func (e *OllamaEmbedder) Model() string {
	return e.model
}

// MockEmbedder for testing (returns random embeddings)
type MockEmbedder struct {
	dimensions int
}

func NewMockEmbedder(dimensions int) *MockEmbedder {
	return &MockEmbedder{dimensions: dimensions}
}

func (e *MockEmbedder) Embed(text string) ([]float32, error) {
	// Return zero vector for testing
	return make([]float32, e.dimensions), nil
}

func (e *MockEmbedder) EmbedBatch(texts []string) ([][]float32, error) {
	embeddings := make([][]float32, len(texts))
	for i := range texts {
		embeddings[i] = make([]float32, e.dimensions)
	}
	return embeddings, nil
}

func (e *MockEmbedder) Dimension() int {
	return e.dimensions
}

func (e *MockEmbedder) Model() string {
	return "mock"
}
