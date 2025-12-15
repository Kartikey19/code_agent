package rag

import (
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"time"

	ignore "github.com/sabhiram/go-gitignore"
)

// RAGIndexer manages the RAG indexing lifecycle
type RAGIndexer struct {
	embedder    Embedder
	vectorStore VectorStore
	stats       *IndexStats
}

// NewRAGIndexer creates a new RAG indexer
func NewRAGIndexer(embedder Embedder, vectorStore VectorStore) *RAGIndexer {
	return &RAGIndexer{
		embedder:    embedder,
		vectorStore: vectorStore,
		stats: &IndexStats{
			EmbeddingModel: embedder.Model(),
			Dimensions:     embedder.Dimension(),
		},
	}
}

// IndexProject indexes all code files in a project
func (r *RAGIndexer) IndexProject(projectPath string) error {
	fmt.Printf("Indexing project: %s\n", projectPath)

	var files []string
	var totalChunks int

	// Fresh index each run to avoid duplicates.
	if err := r.vectorStore.Clear(); err != nil {
		return fmt.Errorf("failed to clear vector store: %w", err)
	}

	// Load .gitignore if it exists
	var gitignore *ignore.GitIgnore
	gitignorePath := filepath.Join(projectPath, ".gitignore")
	if _, err := os.Stat(gitignorePath); err == nil {
		gitignore, _ = ignore.CompileIgnoreFile(gitignorePath)
	}

	// Walk the project directory
	err := filepath.WalkDir(projectPath, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}

		// Get relative path for gitignore matching
		relPath, _ := filepath.Rel(projectPath, path)

		// Always skip these critical directories
		if d.IsDir() {
			name := d.Name()
			if name == ".git" || name == ".index" {
				return filepath.SkipDir
			}
		}

		// Check gitignore
		if gitignore != nil && gitignore.MatchesPath(relPath) {
			if d.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}

		// Skip non-directories that aren't code files
		if !d.IsDir() {
			ext := filepath.Ext(path)
			if !isCodeFile(ext) {
				return nil
			}
			files = append(files, path)
		}

		return nil
	})

	if err != nil {
		return fmt.Errorf("failed to walk directory: %w", err)
	}

	fmt.Printf("Found %d code files\n", len(files))

	// Index each file
	for i, filePath := range files {
		if i%10 == 0 {
			fmt.Printf("Progress: %d/%d files (%.1f%%)\n", i, len(files), float64(i)/float64(len(files))*100)
		}

		chunks, err := r.IndexFile(filePath)
		if err != nil {
			fmt.Printf("Warning: failed to index %s: %v\n", filePath, err)
			continue
		}

		totalChunks += len(chunks)
	}

	// Update stats
	r.stats.TotalFiles = len(files)
	r.stats.TotalChunks = totalChunks
	r.stats.LastUpdated = time.Now().Format(time.RFC3339)

	fmt.Printf("\n✓ Indexed %d files, %d chunks\n", len(files), totalChunks)

	return nil
}

// IndexFile indexes a single file
func (r *RAGIndexer) IndexFile(filePath string) ([]*Chunk, error) {
	// Read file content
	content, err := os.ReadFile(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to read file: %w", err)
	}

	// Chunk the file
	chunker := ChunkerFactory(filePath)
	chunks, err := chunker.ChunkFile(filePath, string(content))
	if err != nil {
		return nil, fmt.Errorf("failed to chunk file: %w", err)
	}

	if len(chunks) == 0 {
		return nil, nil
	}

	// Embed chunks in batches
	batchSize := 10
	for i := 0; i < len(chunks); i += batchSize {
		end := i + batchSize
		if end > len(chunks) {
			end = len(chunks)
		}

		batch := chunks[i:end]
		texts := make([]string, len(batch))
		for j, chunk := range batch {
			texts[j] = chunk.Content
		}

		// Generate embeddings
		embeddings, err := r.embedder.EmbedBatch(texts)
		if err != nil {
			return nil, fmt.Errorf("failed to generate embeddings: %w", err)
		}

		// Store in vector store
		if err := r.vectorStore.InsertBatch(batch, embeddings); err != nil {
			return nil, fmt.Errorf("failed to store embeddings: %w", err)
		}
	}

	return chunks, nil
}

// RemoveFile removes a file from the index
func (r *RAGIndexer) RemoveFile(filePath string) error {
	return r.vectorStore.Delete(filePath)
}

// Search performs semantic search
func (r *RAGIndexer) Search(query string, topK int) ([]*SearchResult, error) {
	// Embed the query
	queryEmbedding, err := r.embedder.Embed(query)
	if err != nil {
		return nil, fmt.Errorf("failed to embed query: %w", err)
	}

	// Search vector store
	results, err := r.vectorStore.Search(queryEmbedding, topK)
	if err != nil {
		return nil, fmt.Errorf("search failed: %w", err)
	}

	return results, nil
}

// Stats returns indexing statistics
func (r *RAGIndexer) Stats() *IndexStats {
	r.stats.TotalChunks = r.vectorStore.Count()
	return r.stats
}

// Clear clears the entire index
func (r *RAGIndexer) Clear() error {
	return r.vectorStore.Clear()
}

// Helper functions

func isCodeFile(ext string) bool {
	codeExts := map[string]bool{
		".go":    true,
		".py":    true,
		".js":    true,
		".ts":    true,
		".jsx":   true,
		".tsx":   true,
		".java":  true,
		".c":     true,
		".cpp":   true,
		".h":     true,
		".hpp":   true,
		".rs":    true,
		".rb":    true,
		".php":   true,
		".cs":    true,
		".swift": true,
		".kt":    true,
		".scala": true,
	}

	return codeExts[strings.ToLower(ext)]
}
