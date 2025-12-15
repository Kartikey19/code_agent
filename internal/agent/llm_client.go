package agent

import (
	"context"
	"fmt"
)

// Message represents a chat message
type Message struct {
	Role    string // "user", "assistant", "system"
	Content string
}

// LLMResponse represents the response from an LLM
type LLMResponse struct {
	Content      string
	Provider     string
	Model        string
	TokensUsed   int
	FinishReason string
}

// LLMClient is the interface that all LLM providers must implement
type LLMClient interface {
	// Chat sends a chat request and returns the response
	Chat(ctx context.Context, messages []Message) (*LLMResponse, error)

	// GetProvider returns the provider name (e.g., "claude", "gemini", "openai")
	GetProvider() string

	// GetModel returns the model being used
	GetModel() string

	// SupportsStreaming returns true if the client supports streaming responses
	SupportsStreaming() bool
}

// LLMConfig holds configuration for LLM clients
type LLMConfig struct {
	Provider string // "claude", "gemini", "openai", "ollama"
	APIKey   string
	Model    string
	BaseURL  string // For custom endpoints (e.g., Ollama)
}

// NewLLMClient creates a new LLM client based on the provider
func NewLLMClient(config LLMConfig) (LLMClient, error) {
	switch config.Provider {
	case "claude":
		return NewClaudeClient(config)
	case "gemini":
		return NewGeminiClient(config)
	case "openai":
		return NewOpenAIClient(config)
	case "ollama":
		return NewOllamaClient(config)
	default:
		return nil, fmt.Errorf("unsupported provider: %s", config.Provider)
	}
}
