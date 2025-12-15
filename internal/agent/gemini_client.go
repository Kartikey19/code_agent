package agent

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

// GeminiClient implements LLMClient for Google's Gemini API
type GeminiClient struct {
	apiKey  string
	model   string
	baseURL string
	client  *http.Client
}

// NewGeminiClient creates a new Gemini API client
func NewGeminiClient(config LLMConfig) (*GeminiClient, error) {
	if config.APIKey == "" {
		return nil, fmt.Errorf("Gemini API key is required")
	}

	model := config.Model
	if model == "" {
		model = "gemini-2.0-flash-exp" // Latest model
	}

	baseURL := config.BaseURL
	if baseURL == "" {
		baseURL = "https://generativelanguage.googleapis.com/v1beta"
	}

	return &GeminiClient{
		apiKey:  config.APIKey,
		model:   model,
		baseURL: baseURL,
		client:  &http.Client{},
	}, nil
}

type geminiRequest struct {
	Contents []geminiContent `json:"contents"`
	SystemInstruction *geminiContent `json:"systemInstruction,omitempty"`
}

type geminiContent struct {
	Role  string        `json:"role,omitempty"`
	Parts []geminiPart  `json:"parts"`
}

type geminiPart struct {
	Text string `json:"text"`
}

type geminiResponse struct {
	Candidates []struct {
		Content struct {
			Parts []struct {
				Text string `json:"text"`
			} `json:"parts"`
			Role string `json:"role"`
		} `json:"content"`
		FinishReason string `json:"finishReason"`
	} `json:"candidates"`
	UsageMetadata struct {
		PromptTokenCount     int `json:"promptTokenCount"`
		CandidatesTokenCount int `json:"candidatesTokenCount"`
		TotalTokenCount      int `json:"totalTokenCount"`
	} `json:"usageMetadata"`
}

// Chat sends a chat request to Gemini API
func (g *GeminiClient) Chat(ctx context.Context, messages []Message) (*LLMResponse, error) {
	var systemPrompt *geminiContent
	var contents []geminiContent

	for _, msg := range messages {
		if msg.Role == "system" {
			systemPrompt = &geminiContent{
				Parts: []geminiPart{{Text: msg.Content}},
			}
		} else {
			// Gemini uses "model" instead of "assistant"
			role := msg.Role
			if role == "assistant" {
				role = "model"
			}
			contents = append(contents, geminiContent{
				Role:  role,
				Parts: []geminiPart{{Text: msg.Content}},
			})
		}
	}

	reqBody := geminiRequest{
		Contents:          contents,
		SystemInstruction: systemPrompt,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	url := fmt.Sprintf("%s/models/%s:generateContent?key=%s", g.baseURL, g.model, g.apiKey)
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")

	resp, err := g.client.Do(req)
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

	var geminiResp geminiResponse
	if err := json.Unmarshal(body, &geminiResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	var content string
	var finishReason string
	if len(geminiResp.Candidates) > 0 {
		candidate := geminiResp.Candidates[0]
		if len(candidate.Content.Parts) > 0 {
			content = candidate.Content.Parts[0].Text
		}
		finishReason = candidate.FinishReason
	}

	return &LLMResponse{
		Content:      content,
		Provider:     "gemini",
		Model:        g.model,
		TokensUsed:   geminiResp.UsageMetadata.TotalTokenCount,
		FinishReason: finishReason,
	}, nil
}

func (g *GeminiClient) GetProvider() string {
	return "gemini"
}

func (g *GeminiClient) GetModel() string {
	return g.model
}

func (g *GeminiClient) SupportsStreaming() bool {
	return false
}
