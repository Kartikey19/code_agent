package agent

import (
	"context"
	"fmt"

	"github.com/yourorg/agent/internal/indexer"
)

// CodingAgent is the main agent that orchestrates task planning and execution
type CodingAgent struct {
	llmClient   LLMClient
	indexer     *indexer.Indexer
	taskManager *TaskManager
	projectPath string
}

// AgentConfig holds configuration for creating a coding agent
type AgentConfig struct {
	ProjectPath string
	LLMConfig   LLMConfig
}

// NewCodingAgent creates a new coding agent
func NewCodingAgent(config AgentConfig) (*CodingAgent, error) {
	// Create LLM client
	llmClient, err := NewLLMClient(config.LLMConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create LLM client: %w", err)
	}

	// Create indexer
	idx := indexer.NewIndexer()
	idx.RegisterParser(indexer.NewGoParser())
	idx.RegisterParser(indexer.NewPythonParser())

	return &CodingAgent{
		llmClient:   llmClient,
		indexer:     idx,
		taskManager: NewTaskManager(),
		projectPath: config.ProjectPath,
	}, nil
}

// PlanTask takes a user prompt and generates a task breakdown
func (a *CodingAgent) PlanTask(ctx context.Context, userPrompt string) (*TaskBreakdown, error) {
	// Step 1: Index the project (or use cache)
	fmt.Println("Indexing project...")
	projIdx, err := a.indexer.IndexProject(a.projectPath)
	if err != nil {
		return nil, fmt.Errorf("failed to index project: %w", err)
	}

	// Step 2: Fetch relevant context
	fmt.Println("Fetching relevant context...")
	contextFetcher := indexer.NewContextFetcher(projIdx)
	projectContext := contextFetcher.FetchContext(userPrompt, 10)

	// Format context for LLM
	contextStr := indexer.FormatContext(projectContext)

	// Step 3: Generate task breakdown prompt
	taskPrompt := a.taskManager.GenerateTaskPrompt(userPrompt, contextStr)

	// Step 4: Send to LLM
	fmt.Printf("Generating task breakdown using %s (%s)...\n",
		a.llmClient.GetProvider(), a.llmClient.GetModel())

	messages := []Message{
		{
			Role:    "system",
			Content: "You are an expert coding assistant that helps break down development tasks into actionable steps.",
		},
		{
			Role:    "user",
			Content: taskPrompt,
		},
	}

	response, err := a.llmClient.Chat(ctx, messages)
	if err != nil {
		return nil, fmt.Errorf("failed to get LLM response: %w", err)
	}

	// Step 5: Parse tasks from LLM response
	fmt.Println("Parsing task breakdown...")
	breakdown, err := a.taskManager.ParseTasksFromLLM(response.Content)
	if err != nil {
		return nil, fmt.Errorf("failed to parse tasks: %w", err)
	}

	breakdown.UserPrompt = userPrompt
	breakdown.Summary = fmt.Sprintf("Task breakdown for: %s", userPrompt)

	return breakdown, nil
}

// Chat sends a message to the LLM with project context
func (a *CodingAgent) Chat(ctx context.Context, userMessage string, includeContext bool) (*LLMResponse, error) {
	messages := []Message{
		{
			Role:    "user",
			Content: userMessage,
		},
	}

	// If context is requested, fetch and prepend it
	if includeContext {
		projIdx, err := a.indexer.IndexProject(a.projectPath)
		if err != nil {
			return nil, fmt.Errorf("failed to index project: %w", err)
		}

		contextFetcher := indexer.NewContextFetcher(projIdx)
		projectContext := contextFetcher.FetchContext(userMessage, 10)
		contextStr := indexer.FormatContext(projectContext)

		// Prepend context to the message
		messages[0].Content = fmt.Sprintf("PROJECT CONTEXT:\n%s\n\nUSER QUESTION:\n%s",
			contextStr, userMessage)
	}

	return a.llmClient.Chat(ctx, messages)
}

// GetProjectSummary returns a summary of the indexed project
func (a *CodingAgent) GetProjectSummary(ctx context.Context) (string, error) {
	projIdx, err := a.indexer.IndexProject(a.projectPath)
	if err != nil {
		return "", fmt.Errorf("failed to index project: %w", err)
	}

	summ := indexer.NewSummarizer()
	return summ.GenerateProjectOverview(projIdx), nil
}

// SearchCode searches for code symbols in the project
func (a *CodingAgent) SearchCode(ctx context.Context, query string) ([]indexer.SearchResult, error) {
	projIdx, err := a.indexer.IndexProject(a.projectPath)
	if err != nil {
		return nil, fmt.Errorf("failed to index project: %w", err)
	}

	search := indexer.NewSearchEngine(projIdx)
	return search.SearchSymbol(query), nil
}

// ExplainCode asks the LLM to explain a specific code symbol
func (a *CodingAgent) ExplainCode(ctx context.Context, symbolName string) (*LLMResponse, error) {
	// Search for the symbol
	results, err := a.SearchCode(ctx, symbolName)
	if err != nil {
		return nil, err
	}

	if len(results) == 0 {
		return nil, fmt.Errorf("symbol '%s' not found", symbolName)
	}

	// Use the first result
	result := results[0]

	prompt := fmt.Sprintf(`Please explain this code:

Symbol: %s
Type: %s
Location: %s:%d
Signature: %s
Documentation: %s

Provide a clear explanation of what this code does, its purpose, and how it's used.`,
		result.Name, result.Type, result.FilePath, result.Line,
		result.Signature, result.Doc)

	messages := []Message{
		{
			Role:    "system",
			Content: "You are an expert code reviewer and educator.",
		},
		{
			Role:    "user",
			Content: prompt,
		},
	}

	return a.llmClient.Chat(ctx, messages)
}
