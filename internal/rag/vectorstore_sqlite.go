package rag

import (
	"database/sql"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"sync"

	_ "modernc.org/sqlite" // Pure Go SQLite driver
)

// SQLiteVectorStore persists embeddings in a SQLite database.
type SQLiteVectorStore struct {
	db   *sql.DB
	dims int
	mu   sync.RWMutex
}

func NewSQLiteVectorStore(dbPath string, dims int) (*SQLiteVectorStore, error) {
	if err := os.MkdirAll(filepath.Dir(dbPath), 0o755); err != nil {
		return nil, fmt.Errorf("create sqlite directory: %w", err)
	}

	db, err := sql.Open("sqlite", dbPath)
	if err != nil {
		return nil, fmt.Errorf("open sqlite: %w", err)
	}

	store := &SQLiteVectorStore{
		db:   db,
		dims: dims,
	}

	if err := store.initSchema(); err != nil {
		return nil, err
	}

	return store, nil
}

func (s *SQLiteVectorStore) initSchema() error {
	schema := `
CREATE TABLE IF NOT EXISTS chunks (
  id TEXT PRIMARY KEY,
  file_path TEXT NOT NULL,
  start_line INTEGER,
  end_line INTEGER,
  chunk_type TEXT,
  symbol_name TEXT,
  language TEXT,
  content TEXT,
  token_count INTEGER,
  hash TEXT,
  embedding BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_path);
`
	_, err := s.db.Exec(schema)
	if err != nil {
		return fmt.Errorf("init schema: %w", err)
	}
	return nil
}

func (s *SQLiteVectorStore) Insert(chunk *Chunk, embedding []float32) error {
	return s.InsertBatch([]*Chunk{chunk}, [][]float32{embedding})
}

func (s *SQLiteVectorStore) InsertBatch(chunks []*Chunk, embeddings [][]float32) error {
	if len(chunks) != len(embeddings) {
		return fmt.Errorf("chunks and embeddings length mismatch: %d vs %d", len(chunks), len(embeddings))
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	tx, err := s.db.Begin()
	if err != nil {
		return fmt.Errorf("begin tx: %w", err)
	}
	stmt, err := tx.Prepare(`
INSERT OR REPLACE INTO chunks
  (id, file_path, start_line, end_line, chunk_type, symbol_name, language, content, token_count, hash, embedding)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
`)
	if err != nil {
		_ = tx.Rollback()
		return fmt.Errorf("prepare insert: %w", err)
	}
	defer stmt.Close()

	for i, chunk := range chunks {
		emb := embeddings[i]
		if len(emb) != s.dims {
			_ = tx.Rollback()
			return fmt.Errorf("embedding dims mismatch: expected %d got %d", s.dims, len(emb))
		}
		blob := encodeEmbedding(emb)
		if _, err := stmt.Exec(
			chunk.ID,
			chunk.FilePath,
			chunk.StartLine,
			chunk.EndLine,
			chunk.ChunkType,
			chunk.SymbolName,
			chunk.Language,
			chunk.Content,
			chunk.TokenCount,
			chunk.Hash,
			blob,
		); err != nil {
			_ = tx.Rollback()
			return fmt.Errorf("insert chunk %s: %w", chunk.FilePath, err)
		}
	}

	if err := tx.Commit(); err != nil {
		return fmt.Errorf("commit: %w", err)
	}
	return nil
}

func (s *SQLiteVectorStore) Search(queryEmbedding []float32, topK int) ([]*SearchResult, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	rows, err := s.db.Query(`SELECT id, file_path, start_line, end_line, chunk_type, symbol_name, language, content, token_count, hash, embedding FROM chunks`)
	if err != nil {
		return nil, fmt.Errorf("select embeddings: %w", err)
	}
	defer rows.Close()

	results := make([]*SearchResult, 0, topK)

	for rows.Next() {
		var (
			id         string
			filePath   string
			startLine  int
			endLine    int
			chunkType  string
			symbolName string
			language   string
			content    string
			tokenCount int
			hash       string
			blob       []byte
		)
		if err := rows.Scan(&id, &filePath, &startLine, &endLine, &chunkType, &symbolName, &language, &content, &tokenCount, &hash, &blob); err != nil {
			return nil, fmt.Errorf("scan chunk: %w", err)
		}
		vec, err := decodeEmbedding(blob, s.dims)
		if err != nil {
			return nil, fmt.Errorf("decode embedding: %w", err)
		}
		score := cosineSimilarity(queryEmbedding, vec)
		chunk := &Chunk{
			ID:         id,
			FilePath:   filePath,
			StartLine:  startLine,
			EndLine:    endLine,
			ChunkType:  chunkType,
			SymbolName: symbolName,
			Language:   language,
			Content:    content,
			TokenCount: tokenCount,
			Hash:       hash,
		}
		results = append(results, &SearchResult{
			Chunk:  chunk,
			Score:  score,
			Source: "rag",
		})
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate rows: %w", err)
	}

	// Sort topK manually (small N expected)
	if len(results) > 1 {
		sort.Slice(results, func(i, j int) bool {
			return results[i].Score > results[j].Score
		})
	}
	if topK > len(results) {
		topK = len(results)
	}
	return results[:topK], nil
}

func (s *SQLiteVectorStore) Delete(filePath string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	_, err := s.db.Exec(`DELETE FROM chunks WHERE file_path = ?`, filePath)
	if err != nil {
		return fmt.Errorf("delete %s: %w", filePath, err)
	}
	return nil
}

func (s *SQLiteVectorStore) Count() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	var count int
	_ = s.db.QueryRow(`SELECT COUNT(*) FROM chunks`).Scan(&count)
	return count
}

func (s *SQLiteVectorStore) Clear() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, err := s.db.Exec(`DELETE FROM chunks`); err != nil {
		return fmt.Errorf("clear chunks: %w", err)
	}
	return nil
}

func encodeEmbedding(vec []float32) []byte {
	buf := make([]byte, len(vec)*4)
	for i, v := range vec {
		binary.LittleEndian.PutUint32(buf[i*4:], mathFloat32bits(v))
	}
	return buf
}

func decodeEmbedding(data []byte, dims int) ([]float32, error) {
	if len(data) != dims*4 {
		return nil, fmt.Errorf("embedding length mismatch: want %d bytes got %d", dims*4, len(data))
	}
	vec := make([]float32, dims)
	for i := 0; i < dims; i++ {
		vec[i] = mathFloat32frombits(binary.LittleEndian.Uint32(data[i*4:]))
	}
	return vec, nil
}

// Tiny wrappers to avoid importing math in the hot path.
func mathFloat32bits(f float32) uint32     { return math.Float32bits(f) }
func mathFloat32frombits(b uint32) float32 { return math.Float32frombits(b) }
