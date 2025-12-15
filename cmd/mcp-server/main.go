package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/yourorg/agent/internal/agent"
	"github.com/yourorg/agent/internal/indexer"
	"github.com/yourorg/agent/internal/rag"
	"github.com/yourorg/agent/internal/retrieval"
)

// MCP Server for Code Indexer
// Exposes indexer functionality via Model Context Protocol

type MCPServer struct {
	indexer       *indexer.Indexer
	cache         map[string]*indexer.ProjectIndex
	ragIndexers   map[string]*rag.RAGIndexer
	queryAnalyzer *retrieval.QueryAnalyzer
	useHybrid     bool // Enable hybrid search
}

func NewMCPServer() *MCPServer {
	idx := indexer.NewIndexer()
	idx.RegisterParser(indexer.NewGoParser())
	idx.RegisterParser(indexer.NewPythonParser())

	return &MCPServer{
		indexer:       idx,
		cache:         make(map[string]*indexer.ProjectIndex),
		ragIndexers:   make(map[string]*rag.RAGIndexer),
		queryAnalyzer: retrieval.NewQueryAnalyzer(),
		useHybrid:     true, // Enable hybrid search by default
	}
}

// MCP Protocol types
type JSONRPCRequest struct {
	JSONRPC string                 `json:"jsonrpc"`
	ID      interface{}            `json:"id,omitempty"`
	Method  string                 `json:"method"`
	Params  map[string]interface{} `json:"params,omitempty"`
}

type JSONRPCResponse struct {
	JSONRPC string      `json:"jsonrpc"`
	ID      interface{} `json:"id,omitempty"`
	Result  interface{} `json:"result,omitempty"`
	Error   *RPCError   `json:"error,omitempty"`
}

type RPCError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

type InitializeResult struct {
	ProtocolVersion string       `json:"protocolVersion"`
	Capabilities    Capabilities `json:"capabilities"`
	ServerInfo      ServerInfo   `json:"serverInfo"`
}

type Capabilities struct {
	Tools *ToolsCapability `json:"tools"`
}

type ToolsCapability struct {
	ListChanged bool `json:"listChanged,omitempty"`
}

type ServerInfo struct {
	Name    string `json:"name"`
	Version string `json:"version"`
}

type ListToolsResult struct {
	Tools []Tool `json:"tools"`
}

type CallToolResult struct {
	Content []ContentBlock `json:"content"`
	IsError bool           `json:"isError,omitempty"`
}

// Tool definitions for MCP
type Tool struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	InputSchema map[string]interface{} `json:"inputSchema"`
}

type ContentBlock struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

// GetTools returns available tools
func (s *MCPServer) GetTools() []Tool {
	return []Tool{
		{
			Name:        "get_project_context",
			Description: "Get relevant code context for a task from an indexed project. Returns file paths, symbols, and project structure relevant to the task.",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"project_path": map[string]interface{}{
						"type":        "string",
						"description": "Absolute path to the project directory",
					},
					"task": map[string]interface{}{
						"type":        "string",
						"description": "Description of the task or bug to get context for",
					},
					"max_results": map[string]interface{}{
						"type":        "integer",
						"description": "Maximum number of results to return (default: 10)",
						"default":     10,
					},
				},
				"required": []string{"project_path", "task"},
			},
		},
		{
			Name:        "search_code",
			Description: "Search for symbols (functions, classes, types) in an indexed project",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"project_path": map[string]interface{}{
						"type":        "string",
						"description": "Absolute path to the project directory",
					},
					"query": map[string]interface{}{
						"type":        "string",
						"description": "Symbol name to search for",
					},
				},
				"required": []string{"project_path", "query"},
			},
		},
		{
			Name:        "get_project_structure",
			Description: "Get the hierarchical structure of a project",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"project_path": map[string]interface{}{
						"type":        "string",
						"description": "Absolute path to the project directory",
					},
					"depth": map[string]interface{}{
						"type":        "integer",
						"description": "Maximum depth to traverse (default: 3)",
						"default":     3,
					},
				},
				"required": []string{"project_path"},
			},
		},
		{
			Name:        "get_call_graph",
			Description: "Get the call graph for a function (who calls it and what it calls)",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"project_path": map[string]interface{}{
						"type":        "string",
						"description": "Absolute path to the project directory",
					},
					"function_name": map[string]interface{}{
						"type":        "string",
						"description": "Name of the function",
					},
					"direction": map[string]interface{}{
						"type":        "string",
						"description": "Direction: 'callers', 'callees', or 'both' (default: 'both')",
						"enum":        []string{"callers", "callees", "both"},
						"default":     "both",
					},
				},
				"required": []string{"project_path", "function_name"},
			},
		},
		{
			Name:        "run_agent_task",
			Description: "Plan and execute a coding task (same behavior as `indexer agent run`). Returns checklist and execution log.",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"project_path": map[string]interface{}{
						"type":        "string",
						"description": "Absolute path to the project directory",
					},
					"task": map[string]interface{}{
						"type":        "string",
						"description": "Description of the task to execute",
					},
					"provider": map[string]interface{}{
						"type":        "string",
						"description": "LLM provider (claude, gemini, openai, ollama)",
						"default":     "claude",
					},
					"model": map[string]interface{}{
						"type":        "string",
						"description": "Optional model name for provider",
					},
					"api_key": map[string]interface{}{
						"type":        "string",
						"description": "API key (falls back to environment variable)",
					},
					"dry_run": map[string]interface{}{
						"type":        "boolean",
						"description": "If true, don't modify files or run commands",
						"default":     true,
					},
					"max_iterations": map[string]interface{}{
						"type":        "integer",
						"description": "Max action iterations per task",
						"default":     20,
					},
					"max_context": map[string]interface{}{
						"type":        "integer",
						"description": "Max context results per task",
						"default":     8,
					},
				},
				"required": []string{"project_path", "task"},
			},
		},
	}
}

// ExecuteTool executes a tool and returns the result
func (s *MCPServer) ExecuteTool(toolName string, arguments map[string]interface{}) (*CallToolResult, error) {
	switch toolName {
	case "get_project_context":
		return s.getProjectContext(arguments)
	case "search_code":
		return s.searchCode(arguments)
	case "get_project_structure":
		return s.getProjectStructure(arguments)
	case "get_call_graph":
		return s.getCallGraph(arguments)
	case "run_agent_task":
		return s.runAgentTask(arguments)
	default:
		return nil, fmt.Errorf("unknown tool: %s", toolName)
	}
}

func (s *MCPServer) getProjectIndex(projectPath string) (*indexer.ProjectIndex, error) {
	if idx, ok := s.cache[projectPath]; ok {
		return idx, nil
	}

	idx, err := s.indexer.IndexProject(projectPath)
	if err != nil {
		return nil, err
	}

	s.cache[projectPath] = idx
	return idx, nil
}

func (s *MCPServer) getOrCreateRAGIndexer(projectPath string) (*rag.RAGIndexer, error) {
	if idx, ok := s.ragIndexers[projectPath]; ok {
		return idx, nil
	}

	embedder := rag.NewOllamaEmbedder("nomic-embed-text")
	dbPath := filepath.Join(projectPath, ".index", "rag_vectors.db")
	store, err := rag.NewSQLiteVectorStore(dbPath, embedder.Dimension())
	if err != nil {
		return nil, fmt.Errorf("create sqlite vector store: %w", err)
	}
	idx := rag.NewRAGIndexer(embedder, store)
	s.ragIndexers[projectPath] = idx
	return idx, nil
}

// ensureRAGIndexed ensures RAG index exists for project (auto-index if needed)
func (s *MCPServer) ensureRAGIndexed(projectPath string) error {
	ragIndexer, err := s.getOrCreateRAGIndexer(projectPath)
	if err != nil {
		return err
	}

	if ragIndexer.Stats().TotalChunks > 0 {
		return nil
	}

	// Auto-index the project
	log.Printf("Auto-indexing project for RAG: %s", projectPath)
	if err := ragIndexer.IndexProject(projectPath); err != nil {
		return fmt.Errorf("failed to RAG index project: %w", err)
	}

	log.Printf("RAG indexing complete: %d chunks", ragIndexer.Stats().TotalChunks)
	return nil
}

func (s *MCPServer) getProjectContext(args map[string]interface{}) (*CallToolResult, error) {
	projectPath := args["project_path"].(string)
	task := args["task"].(string)
	maxResults := 10
	if mr, ok := args["max_results"].(float64); ok {
		maxResults = int(mr)
	}

	log.Printf("getProjectContext called: project=%s, task=%s, useHybrid=%v", projectPath, task, s.useHybrid)

	// Get structural index
	idx, err := s.getProjectIndex(projectPath)
	if err != nil {
		return &CallToolResult{
			Content: []ContentBlock{{Type: "text", Text: fmt.Sprintf("Error indexing project: %v", err)}},
			IsError: true,
		}, nil
	}

	var formatted string

	if s.useHybrid {
		// Always try hybrid search - run both and merge
		if err := s.ensureRAGIndexed(projectPath); err != nil {
			// RAG not available, fall back to structural only
			log.Printf("RAG not available, using structural only: %v", err)
			fetcher := indexer.NewContextFetcher(idx)
			ctx := fetcher.FetchContext(task, maxResults)
			formatted = indexer.FormatContext(ctx)
		} else {
			// Hybrid search: run both and merge
			ragIndexer, _ := s.getOrCreateRAGIndexer(projectPath)
			log.Printf("Hybrid context search: project=%s query=\"%s\"", projectPath, task)
			formatted = s.hybridSearch(idx, ragIndexer, task, maxResults)
		}
	} else {
		// Hybrid disabled, use structural only
		fetcher := indexer.NewContextFetcher(idx)
		ctx := fetcher.FetchContext(task, maxResults)
		formatted = indexer.FormatContext(ctx)
	}

	return &CallToolResult{
		Content: []ContentBlock{{Type: "text", Text: formatted}},
	}, nil
}

// hybridSearch combines structural and semantic search
func (s *MCPServer) hybridSearch(idx *indexer.ProjectIndex, ragIndexer *rag.RAGIndexer, query string, maxResults int) string {
	// Get structural results
	fetcher := indexer.NewContextFetcher(idx)
	structuralCtx := fetcher.FetchContext(query, maxResults)
	log.Printf("Hybrid structural results: %d modules", len(structuralCtx.RelevantModules))

	// Get semantic results from RAG
	ragResults, err := ragIndexer.Search(query, maxResults)
	if err != nil {
		log.Printf("RAG search failed: %v", err)
		return indexer.FormatContext(structuralCtx)
	}
	log.Printf("Hybrid RAG results: %d chunks", len(ragResults))

	// Extract file paths from structural context
	structuralFiles := make([]string, 0)
	for _, mod := range structuralCtx.RelevantModules {
		structuralFiles = append(structuralFiles, mod)
	}

	// Merge results
	merger := retrieval.NewResultMerger(50000) // 50k token budget
	hybridResult := merger.Merge(ragResults, structuralFiles)

	// Format hybrid results
	var result strings.Builder
	result.WriteString(fmt.Sprintf("=== Hybrid Search Results (%s + %s) ===\n\n",
		hybridResult.QueryType, fmt.Sprintf("Found %d files", len(hybridResult.Files))))

	result.WriteString(fmt.Sprintf("Sources: %d from indexer, %d from RAG\n\n",
		hybridResult.Sources.Indexer, hybridResult.Sources.RAG))

	for i, file := range hybridResult.Files {
		if i >= maxResults {
			break
		}

		result.WriteString(fmt.Sprintf("%d. %s (Relevance: %.2f, Source: %s)\n",
			i+1, file.Path, file.Relevance, file.Source))

		// Show highlights
		if len(file.Highlights) > 0 {
			result.WriteString("   Relevant sections:\n")
			for _, highlight := range file.Highlights {
				result.WriteString(fmt.Sprintf("   - Lines %d-%d\n", highlight.Start, highlight.End))
			}
		}

		// Show chunks if available
		if len(file.Chunks) > 0 {
			result.WriteString(fmt.Sprintf("   Found %d relevant code segments\n", len(file.Chunks)))
		}

		result.WriteString("\n")
	}

	result.WriteString(fmt.Sprintf("\nTotal tokens: %d\n", hybridResult.TotalTokens))

	return result.String()
}

func (s *MCPServer) searchCode(args map[string]interface{}) (*CallToolResult, error) {
	projectPath := args["project_path"].(string)
	query := args["query"].(string)

	idx, err := s.getProjectIndex(projectPath)
	if err != nil {
		return &CallToolResult{
			Content: []ContentBlock{{Type: "text", Text: fmt.Sprintf("Error indexing project: %v", err)}},
			IsError: true,
		}, nil
	}

	// Always run structural search
	search := indexer.NewSearchEngine(idx)
	structuralResults := search.SearchSymbol(query)

	var text strings.Builder
	text.WriteString(fmt.Sprintf("=== Search Results for '%s' ===\n\n", query))

	if s.useHybrid {
		// Try to add RAG results
		if err := s.ensureRAGIndexed(projectPath); err == nil {
			// RAG available, get semantic results
			ragIndexer, _ := s.getOrCreateRAGIndexer(projectPath)
			ragResults, err := ragIndexer.Search(query, 10)
			if err == nil && len(ragResults) > 0 {
				// Show combined results
				text.WriteString(fmt.Sprintf("Structural: %d results | Semantic: %d results\n\n", len(structuralResults), len(ragResults)))

				text.WriteString("Structural Matches:\n")
				for i, result := range structuralResults {
					if i >= 5 {
						text.WriteString(fmt.Sprintf("   ... and %d more\n", len(structuralResults)-5))
						break
					}
					text.WriteString(indexer.FormatSearchResult(result) + "\n")
				}

				text.WriteString("\nSemantic Matches:\n")
				for i, result := range ragResults {
					if i >= 5 {
						text.WriteString(fmt.Sprintf("   ... and %d more\n", len(ragResults)-5))
						break
					}
					text.WriteString(fmt.Sprintf("  [Score: %.3f] %s:%d-%d %s\n",
						result.Score, result.Chunk.FilePath, result.Chunk.StartLine,
						result.Chunk.EndLine, result.Chunk.SymbolName))
				}

				return &CallToolResult{
					Content: []ContentBlock{{Type: "text", Text: text.String()}},
				}, nil
			}
		}
	}

	// Structural only (hybrid disabled or RAG not available)
	text.WriteString(fmt.Sprintf("Found %d results:\n\n", len(structuralResults)))
	for _, result := range structuralResults {
		text.WriteString(indexer.FormatSearchResult(result) + "\n")
	}

	return &CallToolResult{
		Content: []ContentBlock{{Type: "text", Text: text.String()}},
	}, nil
}

func (s *MCPServer) getProjectStructure(args map[string]interface{}) (*CallToolResult, error) {
	projectPath := args["project_path"].(string)
	depth := 3
	if d, ok := args["depth"].(float64); ok {
		depth = int(d)
	}

	idx, err := s.getProjectIndex(projectPath)
	if err != nil {
		return &CallToolResult{
			Content: []ContentBlock{{Type: "text", Text: fmt.Sprintf("Error indexing project: %v", err)}},
			IsError: true,
		}, nil
	}

	summ := indexer.NewSummarizer()
	tree := summ.GenerateStructureTree(idx, depth)

	return &CallToolResult{
		Content: []ContentBlock{{Type: "text", Text: tree}},
	}, nil
}

func (s *MCPServer) getCallGraph(args map[string]interface{}) (*CallToolResult, error) {
	projectPath := args["project_path"].(string)
	functionName := args["function_name"].(string)
	direction := "both"
	if d, ok := args["direction"].(string); ok {
		direction = d
	}

	idx, err := s.getProjectIndex(projectPath)
	if err != nil {
		return &CallToolResult{
			Content: []ContentBlock{{Type: "text", Text: fmt.Sprintf("Error indexing project: %v", err)}},
			IsError: true,
		}, nil
	}

	search := indexer.NewSearchEngine(idx)
	results := search.SearchByCallGraph(functionName, direction)

	text := fmt.Sprintf("Call graph for '%s' (%s):\n\n", functionName, direction)
	for _, fn := range results {
		text += fmt.Sprintf("  - %s\n", fn)
	}
	text += fmt.Sprintf("\nTotal: %d functions\n", len(results))

	return &CallToolResult{
		Content: []ContentBlock{{Type: "text", Text: text}},
	}, nil
}

func (s *MCPServer) runAgentTask(args map[string]interface{}) (*CallToolResult, error) {
	projectPath := args["project_path"].(string)
	task := args["task"].(string)
	provider := getStringArg(args, "provider", "claude")
	model := getStringArg(args, "model", "")
	apiKey := getStringArg(args, "api_key", "")
	dryRun := getBoolArg(args, "dry_run", true)
	maxIterations := getIntArg(args, "max_iterations", 20)
	maxContext := getIntArg(args, "max_context", 8)

	if apiKey == "" {
		switch provider {
		case "claude":
			apiKey = os.Getenv("CLAUDE_API_KEY")
		case "gemini":
			apiKey = os.Getenv("GEMINI_API_KEY")
		case "openai":
			apiKey = os.Getenv("OPENAI_API_KEY")
		}
	}

	agentConfig := agent.AgentConfig{
		ProjectPath: projectPath,
		LLMConfig: agent.LLMConfig{
			Provider: provider,
			Model:    model,
			APIKey:   apiKey,
		},
	}

	codingAgent, err := agent.NewCodingAgent(agentConfig)
	if err != nil {
		return nil, err
	}

	runResult, err := codingAgent.Run(context.Background(), task, agent.RunOptions{
		DryRun:            dryRun,
		MaxIterations:     maxIterations,
		MaxContextResults: maxContext,
	})
	if err != nil {
		return nil, err
	}

	tm := agent.NewTaskManager()
	checklist := tm.FormatAsChecklist(runResult.Plan)
	execSummary := formatExecutionLog(runResult.Executions)

	return &CallToolResult{
		Content: []ContentBlock{
			{Type: "text", Text: checklist + "\n\n" + execSummary},
		},
	}, nil
}

func getStringArg(args map[string]interface{}, key, def string) string {
	if v, ok := args[key].(string); ok {
		return v
	}
	return def
}

func getBoolArg(args map[string]interface{}, key string, def bool) bool {
	if v, ok := args[key].(bool); ok {
		return v
	}
	return def
}

func getIntArg(args map[string]interface{}, key string, def int) int {
	if v, ok := args[key]; ok {
		switch t := v.(type) {
		case float64:
			return int(t)
		case int:
			return t
		case string:
			if parsed, err := strconv.Atoi(t); err == nil {
				return parsed
			}
		}
	}
	return def
}

func formatExecutionLog(exec []agent.TaskExecution) string {
	var b strings.Builder
	b.WriteString("Execution log:\n")
	for _, e := range exec {
		status := "pending"
		switch {
		case e.Completed:
			status = "done"
		case e.Failed:
			status = "failed"
		}
		b.WriteString(fmt.Sprintf("- %s [%s]\n", e.Task.Description, status))
		for i, act := range e.Actions {
			res := e.Results[i]
			out := strings.TrimSpace(res.Output)
			if len(out) > 160 {
				out = out[:160] + "..."
			}
			b.WriteString(fmt.Sprintf("  • %s %s -> %t\n", act.Type, act.Path, res.Success))
			if out != "" {
				b.WriteString("    " + out + "\n")
			}
			if res.Error != "" {
				b.WriteString("    error: " + res.Error + "\n")
			}
		}
		if e.FailureMsg != "" {
			b.WriteString("  failure: " + e.FailureMsg + "\n")
		}
	}
	return b.String()
}

func main() {
	logFile, err := os.OpenFile("/tmp/mcp-server.log", os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0666)
	if err == nil {
		log.SetOutput(logFile)
		defer logFile.Close()
	}

	log.Println("MCP Server starting...")

	server := NewMCPServer()

	scanner := bufio.NewScanner(os.Stdin)
	encoder := json.NewEncoder(os.Stdout)

	for scanner.Scan() {
		line := scanner.Text()

		// Skip empty lines
		if len(strings.TrimSpace(line)) == 0 {
			continue
		}

		log.Printf("Received: %s", line)

		var req JSONRPCRequest
		if err := json.Unmarshal([]byte(line), &req); err != nil {
			log.Printf("Error decoding request: %v", err)
			continue
		}

		var resp JSONRPCResponse
		resp.JSONRPC = "2.0"
		resp.ID = req.ID

		switch req.Method {
		case "initialize":
			log.Println("Handling initialize")
			// Use the protocol version from the client request
			clientVersion := "2024-11-05" // default
			if params, ok := req.Params["protocolVersion"].(string); ok {
				clientVersion = params
				log.Printf("Client protocol version: %s", clientVersion)
			}
			resp.Result = InitializeResult{
				ProtocolVersion: clientVersion,
				Capabilities: Capabilities{
					Tools: &ToolsCapability{},
				},
				ServerInfo: ServerInfo{
					Name:    "code-indexer",
					Version: "1.0.0",
				},
			}

		case "notifications/initialized":
			log.Println("Handling notifications/initialized")
			// Notifications don't get responses in JSON-RPC
			continue

		case "tools/list":
			log.Println("Handling tools/list")
			resp.Result = ListToolsResult{
				Tools: server.GetTools(),
			}

		case "tools/call":
			log.Println("Handling tools/call")
			toolName, ok := req.Params["name"].(string)
			if !ok {
				resp.Error = &RPCError{Code: -32602, Message: "Invalid tool name"}
				break
			}

			arguments, ok := req.Params["arguments"].(map[string]interface{})
			if !ok {
				arguments = make(map[string]interface{})
			}

			log.Printf("Executing tool: %s with args: %v", toolName, arguments)
			result, err := server.ExecuteTool(toolName, arguments)
			if err != nil {
				log.Printf("Tool execution error: %v", err)
				resp.Error = &RPCError{Code: -32603, Message: err.Error()}
			} else {
				log.Printf("Tool execution successful: %s", toolName)
				resp.Result = result
			}

		default:
			// Check if it's a notification (no response needed)
			if strings.HasPrefix(req.Method, "notifications/") {
				log.Printf("Ignoring notification: %s", req.Method)
				continue
			}

			log.Printf("Unknown method: %s", req.Method)
			resp.Error = &RPCError{Code: -32601, Message: fmt.Sprintf("Method not found: %s", req.Method)}
		}

		if err := encoder.Encode(resp); err != nil {
			log.Printf("Error encoding response: %v", err)
			break
		}
		log.Printf("Sent response for method: %s", req.Method)
	}

	if err := scanner.Err(); err != nil {
		log.Printf("Scanner error: %v", err)
	}

	log.Println("MCP Server shutting down")
}
