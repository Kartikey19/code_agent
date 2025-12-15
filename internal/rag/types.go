package rag

import (
	"crypto/sha256"
	"encoding/hex"
)

// Chunk represents a piece of code for RAG indexing
type Chunk struct {
	ID         string `json:"id"`
	FilePath   string `json:"file_path"`
	StartLine  int    `json:"start_line"`
	EndLine    int    `json:"end_line"`
	ChunkType  string `json:"chunk_type"`  // function, class, method, block
	SymbolName string `json:"symbol_name"` // If it's a named symbol
	Language   string `json:"language"`
	Content    string `json:"content"`
	TokenCount int    `json:"token_count"`
	Hash       string `json:"hash"` // Content hash for caching
}

// NewChunk creates a new chunk with auto-generated ID and hash
func NewChunk(filePath, content, chunkType, symbolName, language string, startLine, endLine int) *Chunk {
	hash := computeHash(content)
	return &Chunk{
		ID:         hash[:16], // Use first 16 chars of hash as ID
		FilePath:   filePath,
		StartLine:  startLine,
		EndLine:    endLine,
		ChunkType:  chunkType,
		SymbolName: symbolName,
		Language:   language,
		Content:    content,
		TokenCount: estimateTokens(content),
		Hash:       hash,
	}
}

// Embedding wraps a chunk with its vector
type Embedding struct {
	Chunk  *Chunk
	Vector []float32
}

// SearchResult from vector search
type SearchResult struct {
	Chunk  *Chunk
	Score  float32 // 0-1, higher is better
	Source string  // "indexer" or "rag"
}

// HybridResult combines indexer and RAG results
type HybridResult struct {
	Files       []FileResult
	TotalTokens int
	QueryType   string // "structural", "semantic", "hybrid"
	Sources     struct {
		Indexer int
		RAG     int
	}
}

// FileResult represents a relevant file
type FileResult struct {
	Path       string
	Content    string
	Relevance  float32
	Source     string
	Highlights []LineRange
	Chunks     []*Chunk // If partially matched
}

// LineRange for highlighting
type LineRange struct {
	Start int
	End   int
}

// IndexStats tracks RAG index statistics
type IndexStats struct {
	TotalFiles     int
	TotalChunks    int
	IndexSize      int64
	LastUpdated    string
	EmbeddingModel string
	Dimensions     int
}

// Helper functions

func computeHash(content string) string {
	h := sha256.New()
	h.Write([]byte(content))
	return hex.EncodeToString(h.Sum(nil))
}

func estimateTokens(text string) int {
	// Rough estimate: ~4 characters per token
	return len(text) / 4
}
