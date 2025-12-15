package retrieval

import (
	"regexp"
	"strings"
)

// QueryType indicates how to retrieve context
type QueryType string

const (
	StructuralQuery QueryType = "structural" // Use indexer only (exact symbols)
	SemanticQuery   QueryType = "semantic"   // Use RAG only (conceptual)
	HybridQuery     QueryType = "hybrid"     // Use both
)

// QueryAnalyzer determines which retrieval method to use
type QueryAnalyzer struct {
	symbolPattern  *regexp.Regexp
	conceptWords   []string
	behaviorWords  []string
	callGraphWords []string
}

func NewQueryAnalyzer() *QueryAnalyzer {
	return &QueryAnalyzer{
		// Pattern for symbol names (PascalCase, camelCase)
		symbolPattern: regexp.MustCompile(`[A-Z][a-zA-Z0-9]*`),

		// Words indicating conceptual/semantic queries
		conceptWords: []string{"how", "where", "why", "find", "search", "explain", "show me"},

		// Words indicating behavior/functionality
		behaviorWords: []string{"handle", "process", "validate", "check", "manage", "create", "update", "delete"},

		// Words indicating call graph queries
		callGraphWords: []string{"calls", "called by", "uses", "used by", "depends on", "imports"},
	}
}

// Classify determines the query type
func (qa *QueryAnalyzer) Classify(query string) QueryType {
	query = strings.ToLower(query)

	hasSymbol := qa.hasSymbols(query)
	hasFilePath := strings.Contains(query, "/") || strings.Contains(query, ".go") || strings.Contains(query, ".py")
	hasCallGraph := qa.containsAny(query, qa.callGraphWords)

	hasConcept := qa.containsAny(query, qa.conceptWords)
	hasBehavior := qa.containsAny(query, qa.behaviorWords)

	// Structural: explicit symbols, file paths, or call graph queries
	if (hasSymbol || hasFilePath || hasCallGraph) && !hasConcept && !hasBehavior {
		return StructuralQuery
	}

	// Semantic: conceptual or behavioral without specific symbols
	if (hasConcept || hasBehavior) && !hasSymbol && !hasFilePath {
		return SemanticQuery
	}

	// Hybrid: mix of both or default
	return HybridQuery
}

func (qa *QueryAnalyzer) hasSymbols(query string) bool {
	// Check for PascalCase or camelCase patterns
	matches := qa.symbolPattern.FindAllString(query, -1)

	// Filter out common English words that happen to be capitalized
	commonWords := map[string]bool{
		"I": true, "A": true, "The": true, "This": true, "That": true,
		"What": true, "How": true, "Where": true, "Why": true,
	}

	for _, match := range matches {
		if !commonWords[match] && len(match) > 1 {
			return true
		}
	}

	return false
}

func (qa *QueryAnalyzer) containsAny(text string, words []string) bool {
	for _, word := range words {
		if strings.Contains(text, word) {
			return true
		}
	}
	return false
}
