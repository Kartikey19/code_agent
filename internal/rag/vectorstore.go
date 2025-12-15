package rag

import "math"

// VectorStore stores and searches embeddings
type VectorStore interface {
	Insert(chunk *Chunk, embedding []float32) error
	InsertBatch(chunks []*Chunk, embeddings [][]float32) error
	Search(queryEmbedding []float32, topK int) ([]*SearchResult, error)
	Delete(filePath string) error
	Count() int
	Clear() error
}

// Helper functions

func cosineSimilarity(a, b []float32) float32 {
	if len(a) != len(b) {
		return 0
	}

	var dotProduct, normA, normB float32

	for i := range a {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return float32(dotProduct / (float32(math.Sqrt(float64(normA))) * float32(math.Sqrt(float64(normB)))))
}

func euclideanDistance(a, b []float32) float32 {
	if len(a) != len(b) {
		return math.MaxFloat32
	}

	var sum float32
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}

	return float32(math.Sqrt(float64(sum)))
}
