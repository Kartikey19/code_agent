package agent

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

// OllamaClient implements LLMClient for local Ollama models
type OllamaClient struct {
	model   string
	baseURL string
	client  *http.Client
}

// NewOllamaClient creates a new Ollama client
func NewOllamaClient(config LLMConfig) (*OllamaClient, error) {
	model := config.Model
	if model == "" {
		model = "llama3.3" // Default local model
	}

	baseURL := config.BaseURL
	if baseURL == "" {
		baseURL = "http://localhost:11434"
	}

	return &OllamaClient{
		model:   model,
		baseURL: baseURL,
		client:  &http.Client{},
	}, nil
}

type ollamaRequest struct {
	Model    string          `json:"model"`
	Messages []ollamaMessage `json:"messages"`
	Stream   bool            `json:"stream"`
}

type ollamaMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ollamaResponse struct {
	Model     string `json:"model"`
	CreatedAt string `json:"created_at"`
	Message   struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	} `json:"message"`
	Done bool `json:"done"`
}

// Chat sends a chat request to Ollama
func (o *OllamaClient) Chat(ctx context.Context, messages []Message) (*LLMResponse, error) {
	var ollamaMessages []ollamaMessage
	for _, msg := range messages {
		ollamaMessages = append(ollamaMessages, ollamaMessage{
			Role:    msg.Role,
			Content: msg.Content,
		})
	}

	reqBody := ollamaRequest{
		Model:    o.model,
		Messages: ollamaMessages,
		Stream:   false,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", o.baseURL+"/api/chat", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")

	resp, err := o.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("API request failed with status %d: %s", resp.StatusCode, string(body))
	}

	var ollamaResp ollamaResponse
	if err := json.Unmarshal(body, &ollamaResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	return &LLMResponse{
		Content:      ollamaResp.Message.Content,
		Provider:     "ollama",
		Model:        ollamaResp.Model,
		TokensUsed:   0, // Ollama doesn't return token counts in basic mode
		FinishReason: "stop",
	}, nil
}

func (o *OllamaClient) GetProvider() string {
	return "ollama"
}

func (o *OllamaClient) GetModel() string {
	return o.model
}

func (o *OllamaClient) SupportsStreaming() bool {
	return false
}
