package retrieval

import (
	"sort"

	"github.com/yourorg/agent/internal/rag"
)

// ResultMerger combines and ranks results from multiple sources
type ResultMerger struct {
	maxTokens int
}

func NewResultMerger(maxTokens int) *ResultMerger {
	return &ResultMerger{
		maxTokens: maxTokens,
	}
}

// Merge combines results from indexer and RAG
func (m *ResultMerger) Merge(ragResults []*rag.SearchResult, indexerFiles []string) *rag.HybridResult {
	result := &rag.HybridResult{
		Files: make([]rag.FileResult, 0),
	}

	// Track which files we've seen
	fileMap := make(map[string]*rag.FileResult)

	// Add RAG results
	for _, res := range ragResults {
		filePath := res.Chunk.FilePath

		if existing, ok := fileMap[filePath]; ok {
			// File already added, boost score and add chunk
			existing.Relevance = max(existing.Relevance, res.Score) * 1.2 // Boost for multiple matches
			existing.Chunks = append(existing.Chunks, res.Chunk)
			existing.Source = "both" // Found by both sources
		} else {
			// New file
			fileResult := rag.FileResult{
				Path:      filePath,
				Relevance: res.Score,
				Source:    "rag",
				Chunks:    []*rag.Chunk{res.Chunk},
				Highlights: []rag.LineRange{
					{Start: res.Chunk.StartLine, End: res.Chunk.EndLine},
				},
			}
			fileMap[filePath] = &fileResult
			result.Sources.RAG++
		}
	}

	// Add indexer results (exact matches get high score)
	for _, filePath := range indexerFiles {
		if existing, ok := fileMap[filePath]; ok {
			// Boost score for being in both sources
			existing.Relevance = existing.Relevance * 1.3
			existing.Source = "both"
		} else {
			// New file from indexer
			fileResult := rag.FileResult{
				Path:      filePath,
				Relevance: 0.9, // High relevance for exact symbol match
				Source:    "indexer",
				Chunks:    []*rag.Chunk{},
			}
			fileMap[filePath] = &fileResult
			result.Sources.Indexer++
		}
	}

	// Convert map to slice
	for _, fileResult := range fileMap {
		result.Files = append(result.Files, *fileResult)
	}

	// Sort by relevance
	sort.Slice(result.Files, func(i, j int) bool {
		return result.Files[i].Relevance > result.Files[j].Relevance
	})

	// Truncate to token budget
	m.truncateToTokenBudget(result)

	return result
}

func (m *ResultMerger) truncateToTokenBudget(result *rag.HybridResult) {
	totalTokens := 0
	truncated := make([]rag.FileResult, 0)

	for _, fileResult := range result.Files {
		fileTokens := 0

		// Estimate tokens from chunks
		for _, chunk := range fileResult.Chunks {
			fileTokens += chunk.TokenCount
		}

		// If no chunks, estimate from file size (rough estimate)
		if fileTokens == 0 {
			fileTokens = 500 // Default estimate
		}

		if totalTokens+fileTokens > m.maxTokens {
			break
		}

		totalTokens += fileTokens
		truncated = append(truncated, fileResult)
	}

	result.Files = truncated
	result.TotalTokens = totalTokens
}

func max(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}
