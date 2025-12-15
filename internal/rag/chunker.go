package rag

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"path/filepath"
	"strings"
)

// Chunker splits code into searchable chunks
type Chunker interface {
	ChunkFile(filePath string, content string) ([]*Chunk, error)
	Language() string
}

// GoChunker implements AST-based chunking for Go files
type GoChunker struct{}

func NewGoChunker() *GoChunker {
	return &GoChunker{}
}

func (c *GoChunker) Language() string {
	return "go"
}

func (c *GoChunker) ChunkFile(filePath string, content string) ([]*Chunk, error) {
	fset := token.NewFileSet()
	node, err := parser.ParseFile(fset, filePath, content, parser.ParseComments)
	if err != nil {
		// Fallback to generic chunking if parsing fails
		return c.genericChunk(filePath, content), nil
	}

	var chunks []*Chunk

	// Extract package-level declarations
	ast.Inspect(node, func(n ast.Node) bool {
		switch decl := n.(type) {
		case *ast.FuncDecl:
			// Function or method
			start := fset.Position(decl.Pos()).Line
			end := fset.Position(decl.End()).Line
			funcContent := extractLines(content, start, end)

			chunkType := "function"
			symbolName := decl.Name.Name

			// Check if it's a method
			if decl.Recv != nil {
				chunkType = "method"
				// Get receiver type
				if len(decl.Recv.List) > 0 {
					recvType := exprToString(decl.Recv.List[0].Type)
					symbolName = recvType + "." + decl.Name.Name
				}
			}

			subChunks := splitLargeChunk(filePath, funcContent, chunkType, symbolName, "go", start, end)
			chunks = append(chunks, subChunks...)

		case *ast.GenDecl:
			// Type, const, var declarations
			if decl.Tok == token.TYPE {
				// Type declarations (structs, interfaces, etc.)
				for _, spec := range decl.Specs {
					if typeSpec, ok := spec.(*ast.TypeSpec); ok {
						start := fset.Position(typeSpec.Pos()).Line
						end := fset.Position(typeSpec.End()).Line
						typeContent := extractLines(content, start, end)

						chunkType := "type"
						if _, isStruct := typeSpec.Type.(*ast.StructType); isStruct {
							chunkType = "struct"
						} else if _, isInterface := typeSpec.Type.(*ast.InterfaceType); isInterface {
							chunkType = "interface"
						}

						subChunks := splitLargeChunk(filePath, typeContent, chunkType, typeSpec.Name.Name, "go", start, end)
						chunks = append(chunks, subChunks...)
					}
				}
			}
		}
		return true
	})

	// If no chunks were extracted, fallback to generic
	if len(chunks) == 0 {
		return c.genericChunk(filePath, content), nil
	}

	return chunks, nil
}

// genericChunk uses sliding window for files that can't be parsed
func (c *GoChunker) genericChunk(filePath string, content string) []*Chunk {
	lines := strings.Split(content, "\n")
	var chunks []*Chunk

	// Sliding window: 50 lines per chunk, 10 line overlap
	chunkSize := 50
	overlap := 10
	stride := chunkSize - overlap

	for i := 0; i < len(lines); i += stride {
		end := i + chunkSize
		if end > len(lines) {
			end = len(lines)
		}

		chunkLines := lines[i:end]
		chunkContent := strings.Join(chunkLines, "\n")

		if len(strings.TrimSpace(chunkContent)) < 50 {
			// Skip very small chunks
			continue
		}

		chunk := NewChunk(filePath, chunkContent, "block", "", "go", i+1, end)
		chunks = append(chunks, chunk)

		if end >= len(lines) {
			break
		}
	}

	return chunks
}

// PythonChunker implements lightweight chunking for Python files using indentation blocks.
// It captures decorators with their def/class and handles nested defs by indentation.
type PythonChunker struct{}

func NewPythonChunker() *PythonChunker {
	return &PythonChunker{}
}

func (c *PythonChunker) Language() string {
	return "python"
}

func (c *PythonChunker) ChunkFile(filePath string, content string) ([]*Chunk, error) {
	lines := strings.Split(content, "\n")
	var chunks []*Chunk

	for i := 0; i < len(lines); i++ {
		line := lines[i]
		trimmed := strings.TrimSpace(line)
		if trimmed == "" {
			continue
		}

		var chunkType, symbolName string
		switch {
		case strings.HasPrefix(trimmed, "class "):
			chunkType = "class"
			symbolName = extractPythonName(trimmed[len("class "):])
		case strings.HasPrefix(trimmed, "def "):
			// treat indented defs as methods
			if leadingIndent(line) > 0 {
				chunkType = "method"
			} else {
				chunkType = "function"
			}
			symbolName = extractPythonName(trimmed[len("def "):])
		default:
			continue
		}

		// Include decorators immediately above with same indent
		baseIndent := leadingIndent(line)
		startIdx := i
		for d := i - 1; d >= 0; d-- {
			decTrim := strings.TrimSpace(lines[d])
			if decTrim == "" {
				break
			}
			if strings.HasPrefix(decTrim, "@") && leadingIndent(lines[d]) == baseIndent {
				startIdx = d
				continue
			}
			break
		}

		// Use indentation to find end of block
		endIdx := i + 1
		for j := i + 1; j < len(lines); j++ {
			next := lines[j]
			trimNext := strings.TrimSpace(next)
			if trimNext == "" || strings.HasPrefix(trimNext, "#") {
				continue
			}
			if leadingIndent(next) <= baseIndent && !strings.HasPrefix(trimNext, "@") {
				break
			}
			endIdx = j + 1
		}

		chunkContent := strings.Join(lines[startIdx:endIdx], "\n")
		// Skip tiny blocks
		if len(strings.TrimSpace(chunkContent)) < 20 {
			continue
		}

		// Split large chunks to avoid exceeding embedding model context limit
		subChunks := splitLargeChunk(filePath, chunkContent, chunkType, symbolName, "python", startIdx+1, endIdx)
		chunks = append(chunks, subChunks...)
		i = endIdx - 1 // continue after this block
	}

	if len(chunks) == 0 {
		return genericSlidingChunks(filePath, content, "python"), nil
	}
	return chunks, nil
}

// splitLargeChunk splits oversized chunks into smaller pieces
// Max chunk size is ~4000 characters (roughly 1000 tokens) to stay well below embedding model limits
func splitLargeChunk(filePath, content, chunkType, symbolName, language string, start, end int) []*Chunk {
	const maxChunkSize = 4000
	const overlapLines = 10

	// If chunk is small enough, return as-is
	if len(content) <= maxChunkSize {
		return []*Chunk{NewChunk(filePath, content, chunkType, symbolName, language, start, end)}
	}

	// Split into smaller overlapping chunks
	lines := strings.Split(content, "\n")
	var chunks []*Chunk

	// Calculate lines per chunk (~100 lines assuming ~40 chars per line)
	linesPerChunk := maxChunkSize / 40
	if linesPerChunk > 100 {
		linesPerChunk = 100
	}

	partNum := 1
	for i := 0; i < len(lines); i += (linesPerChunk - overlapLines) {
		endIdx := i + linesPerChunk
		if endIdx > len(lines) {
			endIdx = len(lines)
		}

		subContent := strings.Join(lines[i:endIdx], "\n")
		if len(strings.TrimSpace(subContent)) < 20 {
			continue
		}

		// Create chunk with part indicator in symbol name
		partSymbol := symbolName
		if partNum > 1 || endIdx < len(lines) {
			partSymbol = fmt.Sprintf("%s_part%d", symbolName, partNum)
		}

		chunk := NewChunk(filePath, subContent, chunkType, partSymbol, language, start+i, start+endIdx-1)
		chunks = append(chunks, chunk)
		partNum++

		if endIdx >= len(lines) {
			break
		}
	}

	return chunks
}

func extractPythonName(signature string) string {
	// signature examples: "Foo(Bar):", "foo(bar):"
	sig := strings.TrimSpace(signature)
	sig = strings.TrimSuffix(sig, ":")
	if idx := strings.Index(sig, "("); idx >= 0 {
		sig = sig[:idx]
	}
	return strings.TrimSpace(sig)
}

func leadingIndent(s string) int {
	return len(s) - len(strings.TrimLeft(s, " \t"))
}

func genericSlidingChunks(filePath, content, lang string) []*Chunk {
	lines := strings.Split(content, "\n")
	var chunks []*Chunk

	chunkSize := 50
	overlap := 10
	stride := chunkSize - overlap

	for i := 0; i < len(lines); i += stride {
		end := i + chunkSize
		if end > len(lines) {
			end = len(lines)
		}

		chunkLines := lines[i:end]
		chunkContent := strings.Join(chunkLines, "\n")

		if len(strings.TrimSpace(chunkContent)) < 50 {
			continue
		}

		chunk := NewChunk(filePath, chunkContent, "block", "", lang, i+1, end)
		chunks = append(chunks, chunk)

		if end >= len(lines) {
			break
		}
	}

	return chunks
}

// Helper functions

func extractLines(content string, startLine, endLine int) string {
	lines := strings.Split(content, "\n")
	if startLine < 1 || startLine > len(lines) {
		return ""
	}
	if endLine > len(lines) {
		endLine = len(lines)
	}

	return strings.Join(lines[startLine-1:endLine], "\n")
}

func exprToString(expr ast.Expr) string {
	switch e := expr.(type) {
	case *ast.Ident:
		return e.Name
	case *ast.StarExpr:
		return "*" + exprToString(e.X)
	case *ast.SelectorExpr:
		return exprToString(e.X) + "." + e.Sel.Name
	default:
		return ""
	}
}

// ChunkerFactory creates appropriate chunker based on file extension
func ChunkerFactory(filePath string) Chunker {
	ext := filepath.Ext(filePath)
	switch ext {
	case ".go":
		return NewGoChunker()
	case ".py":
		return NewPythonChunker()
	default:
		// TODO: Add JS/TS chunkers
		return NewGoChunker() // Fallback for now
	}
}
