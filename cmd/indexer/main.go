package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"

	"github.com/yourorg/agent/internal/agent"
	"github.com/yourorg/agent/internal/indexer"
	"github.com/yourorg/agent/internal/rag"
)

const usage = `Memory Indexer & Coding Agent - Universal AI Coding Assistant

Usage:
  indexer <command> [options]

INDEXER COMMANDS:
  index <path>              Index a project and create searchable memory
  search <query>            Search for symbols in the indexed project
  structure <path>          Show project structure tree
  callgraph <function>      Show call graph for a function
  imports <module>          Show import relationships for a module
  info <symbol>             Get detailed information about a symbol
  fetch_context <task>      Get relevant context for a task/prompt

AGENT COMMANDS:
  agent plan <task>         Generate task breakdown for a coding task
  agent chat <message>      Chat with AI using project context
  agent explain <symbol>    Get AI explanation of a code symbol

RAG COMMANDS:
  rag index <path>          Build semantic RAG index for a project
  rag search <query>        Perform semantic search
  rag status                Show RAG index statistics

Options:
  -path string              Path to project (default ".")
  -json                     Output in JSON format
  -depth int                Tree depth for structure (default 3)
  -refresh                  Force refresh index (ignore cache)
  -max-results int          Maximum results for fetch_context (default 10)
  -provider string          LLM provider: claude, gemini, openai, ollama (default "claude")
  -model string             Model name (provider-specific)
  -api-key string           API key (or use env: CLAUDE_API_KEY, GEMINI_API_KEY, OPENAI_API_KEY)

Examples:
  # Index a project
  indexer index /path/to/myproject

  # Generate task breakdown with Claude
  indexer agent plan "fix the secondary_phone bug in patient API"

  # Use Gemini for task planning
  indexer agent plan "add payment processing" -provider=gemini -api-key=$GEMINI_API_KEY

  # Chat with context
  indexer agent chat "how does authentication work?" -path=/path/to/project

  # Explain a symbol
  indexer agent explain UserModel -provider=openai
`

func main() {
	if len(os.Args) < 2 {
		fmt.Print(usage)
		os.Exit(1)
	}

	command := os.Args[1]

	switch command {
	case "index":
		cmdIndex()
	case "search":
		cmdSearch()
	case "structure":
		cmdStructure()
	case "callgraph":
		cmdCallGraph()
	case "imports":
		cmdImports()
	case "info":
		cmdInfo()
	case "fetch_context":
		cmdFetchContext()
	case "agent":
		cmdAgent()
	case "rag":
		cmdRAG()
	case "help", "-h", "--help":
		fmt.Print(usage)
	default:
		fmt.Printf("Unknown command: %s\n\n", command)
		fmt.Print(usage)
		os.Exit(1)
	}
}

func cmdIndex() {
	fs := flag.NewFlagSet("index", flag.ExitOnError)
	projectPath := fs.String("path", ".", "Path to the project to index")
	jsonOutput := fs.Bool("json", false, "Output in JSON format")
	refresh := fs.Bool("refresh", false, "Force refresh (ignore cache)")
	fs.Parse(os.Args[2:])

	// Get path from args or flag
	if fs.NArg() > 0 {
		*projectPath = fs.Arg(0)
	}

	absPath, err := filepath.Abs(*projectPath)
	if err != nil {
		log.Fatalf("Failed to resolve path: %v", err)
	}

	if _, err := os.Stat(absPath); err != nil {
		log.Fatalf("Project path does not exist: %v", err)
	}

	if *refresh {
		fmt.Println("Force refresh enabled - ignoring cache")
	}

	fmt.Printf("Indexing project: %s\n", absPath)

	idx := indexer.NewIndexer()
	idx.RegisterParser(indexer.NewGoParser())
	idx.RegisterParser(indexer.NewPythonParser())

	// Set cache based on refresh flag
	idx.SetCacheEnabled(!*refresh)

	projIdx, err := idx.IndexProject(absPath)
	if err != nil {
		log.Fatalf("Indexing failed: %v", err)
	}

	if *jsonOutput {
		data, _ := json.MarshalIndent(projIdx, "", "  ")
		fmt.Println(string(data))
	} else {
		summ := indexer.NewSummarizer()
		overview := summ.GenerateProjectOverview(projIdx)
		fmt.Println(overview)
		fmt.Printf("\n✓ Indexed %d modules, %d symbols\n", len(projIdx.Modules), len(projIdx.SymbolTable))
	}
}

func cmdSearch() {
	fs := flag.NewFlagSet("search", flag.ExitOnError)
	projectPath := fs.String("path", ".", "Path to the indexed project")
	searchType := fs.String("type", "symbol", "Search type: symbol, doc")
	jsonOutput := fs.Bool("json", false, "Output in JSON format")
	fs.Parse(os.Args[2:])

	if fs.NArg() < 1 {
		log.Fatal("Usage: indexer search <query>")
	}

	query := fs.Arg(0)
	absPath, _ := filepath.Abs(*projectPath)

	idx := indexer.NewIndexer()
	idx.RegisterParser(indexer.NewGoParser())
	idx.RegisterParser(indexer.NewPythonParser())

	projIdx, err := idx.IndexProject(absPath)
	if err != nil {
		log.Fatalf("Failed to load index: %v", err)
	}

	searchEngine := indexer.NewSearchEngine(projIdx)

	var results []indexer.SearchResult
	switch *searchType {
	case "symbol":
		results = searchEngine.SearchSymbol(query)
	case "doc":
		results = searchEngine.SearchDocumentation(query)
	default:
		results = searchEngine.SearchSymbol(query)
	}

	if *jsonOutput {
		data, _ := json.MarshalIndent(results, "", "  ")
		fmt.Println(string(data))
	} else {
		fmt.Printf("Found %d results for '%s':\n\n", len(results), query)
		for _, result := range results {
			fmt.Println(indexer.FormatSearchResult(result))
		}
	}
}

func cmdStructure() {
	fs := flag.NewFlagSet("structure", flag.ExitOnError)
	projectPath := fs.String("path", ".", "Path to the project")
	depth := fs.Int("depth", 3, "Maximum tree depth")
	fs.Parse(os.Args[2:])

	if fs.NArg() > 0 {
		*projectPath = fs.Arg(0)
	}

	absPath, _ := filepath.Abs(*projectPath)

	idx := indexer.NewIndexer()
	idx.RegisterParser(indexer.NewGoParser())
	idx.RegisterParser(indexer.NewPythonParser())

	projIdx, err := idx.IndexProject(absPath)
	if err != nil {
		log.Fatalf("Failed to load index: %v", err)
	}

	summ := indexer.NewSummarizer()
	tree := summ.GenerateStructureTree(projIdx, *depth)
	fmt.Println(tree)
}

func cmdCallGraph() {
	fs := flag.NewFlagSet("callgraph", flag.ExitOnError)
	projectPath := fs.String("path", ".", "Path to the project")
	direction := fs.String("dir", "both", "Direction: callers, callees, both")
	fs.Parse(os.Args[2:])

	if fs.NArg() < 1 {
		log.Fatal("Usage: indexer callgraph <function>")
	}

	functionName := fs.Arg(0)
	absPath, _ := filepath.Abs(*projectPath)

	idx := indexer.NewIndexer()
	idx.RegisterParser(indexer.NewGoParser())
	idx.RegisterParser(indexer.NewPythonParser())

	projIdx, err := idx.IndexProject(absPath)
	if err != nil {
		log.Fatalf("Failed to load index: %v", err)
	}

	searchEngine := indexer.NewSearchEngine(projIdx)
	results := searchEngine.SearchByCallGraph(functionName, *direction)

	fmt.Printf("Call graph for '%s' (%s):\n\n", functionName, *direction)
	for _, fn := range results {
		fmt.Printf("  - %s\n", fn)
	}
	fmt.Printf("\nTotal: %d functions\n", len(results))
}

func cmdImports() {
	fs := flag.NewFlagSet("imports", flag.ExitOnError)
	projectPath := fs.String("path", ".", "Path to the project")
	direction := fs.String("dir", "both", "Direction: imports, imported_by, both")
	fs.Parse(os.Args[2:])

	if fs.NArg() < 1 {
		log.Fatal("Usage: indexer imports <module>")
	}

	moduleName := fs.Arg(0)
	absPath, _ := filepath.Abs(*projectPath)

	idx := indexer.NewIndexer()
	idx.RegisterParser(indexer.NewGoParser())
	idx.RegisterParser(indexer.NewPythonParser())

	projIdx, err := idx.IndexProject(absPath)
	if err != nil {
		log.Fatalf("Failed to load index: %v", err)
	}

	searchEngine := indexer.NewSearchEngine(projIdx)
	results := searchEngine.SearchImports(moduleName, *direction)

	fmt.Printf("Import graph for '%s' (%s):\n\n", moduleName, *direction)
	for _, mod := range results {
		fmt.Printf("  - %s\n", mod)
	}
	fmt.Printf("\nTotal: %d modules\n", len(results))
}

func cmdInfo() {
	fs := flag.NewFlagSet("info", flag.ExitOnError)
	projectPath := fs.String("path", ".", "Path to the project")
	jsonOutput := fs.Bool("json", false, "Output in JSON format")
	fs.Parse(os.Args[2:])

	if fs.NArg() < 1 {
		log.Fatal("Usage: indexer info <symbol>")
	}

	symbolName := fs.Arg(0)
	absPath, _ := filepath.Abs(*projectPath)

	idx := indexer.NewIndexer()
	idx.RegisterParser(indexer.NewGoParser())
	idx.RegisterParser(indexer.NewPythonParser())

	projIdx, err := idx.IndexProject(absPath)
	if err != nil {
		log.Fatalf("Failed to load index: %v", err)
	}

	searchEngine := indexer.NewSearchEngine(projIdx)
	result := searchEngine.GetSymbolDetails(symbolName)

	if result == nil {
		fmt.Printf("Symbol '%s' not found\n", symbolName)
		os.Exit(1)
	}

	if *jsonOutput {
		data, _ := json.MarshalIndent(result, "", "  ")
		fmt.Println(string(data))
	} else {
		fmt.Println(indexer.FormatSearchResult(*result))
	}
}

func cmdFetchContext() {
	fs := flag.NewFlagSet("fetch_context", flag.ExitOnError)
	projectPath := fs.String("path", ".", "Path to the project")
	maxResults := fs.Int("max-results", 10, "Maximum number of results")
	jsonOutput := fs.Bool("json", false, "Output in JSON format")
	refresh := fs.Bool("refresh", false, "Force refresh (ignore cache)")
	fs.Parse(os.Args[2:])

	if fs.NArg() < 1 {
		log.Fatal("Usage: indexer fetch_context \"<task description>\"")
	}

	task := fs.Arg(0)
	absPath, _ := filepath.Abs(*projectPath)

	fmt.Printf("Fetching context for: %s\n", task)

	idx := indexer.NewIndexer()
	idx.RegisterParser(indexer.NewGoParser())
	idx.RegisterParser(indexer.NewPythonParser())

	// Set cache based on refresh flag
	idx.SetCacheEnabled(!*refresh)

	projIdx, err := idx.IndexProject(absPath)
	if err != nil {
		log.Fatalf("Failed to load index: %v", err)
	}

	// Create context fetcher
	fetcher := indexer.NewContextFetcher(projIdx)

	// Fetch context
	ctx := fetcher.FetchContext(task, *maxResults)

	if *jsonOutput {
		data, _ := json.MarshalIndent(ctx, "", "  ")
		fmt.Println(string(data))
	} else {
		fmt.Println(indexer.FormatContext(ctx))
	}
}

func cmdAgent() {
	if len(os.Args) < 3 {
		log.Fatal("Usage: indexer agent <subcommand> [options]\nSubcommands: plan, chat, explain, run")
	}

	subcommand := os.Args[2]

	switch subcommand {
	case "plan":
		cmdAgentPlan()
	case "chat":
		cmdAgentChat()
	case "explain":
		cmdAgentExplain()
	case "run":
		cmdAgentRun()
	default:
		log.Fatalf("Unknown agent subcommand: %s\nAvailable: plan, chat, explain, run", subcommand)
	}
}

func cmdAgentPlan() {
	fs := flag.NewFlagSet("agent plan", flag.ExitOnError)
	projectPath := fs.String("path", ".", "Path to the project")
	provider := fs.String("provider", "claude", "LLM provider (claude, gemini, openai, ollama)")
	model := fs.String("model", "", "Model name (provider-specific)")
	apiKey := fs.String("api-key", "", "API key (or use environment variable)")
	jsonOutput := fs.Bool("json", false, "Output in JSON format")
	fs.Parse(os.Args[3:])

	if fs.NArg() < 1 {
		log.Fatal("Usage: indexer agent plan \"<task description>\"")
	}

	task := fs.Arg(0)
	absPath, _ := filepath.Abs(*projectPath)

	// Get API key from environment if not provided
	if *apiKey == "" {
		switch *provider {
		case "claude":
			*apiKey = os.Getenv("CLAUDE_API_KEY")
		case "gemini":
			*apiKey = os.Getenv("GEMINI_API_KEY")
		case "openai":
			*apiKey = os.Getenv("OPENAI_API_KEY")
		}
	}

	// Create agent
	agentConfig := agent.AgentConfig{
		ProjectPath: absPath,
		LLMConfig: agent.LLMConfig{
			Provider: *provider,
			APIKey:   *apiKey,
			Model:    *model,
		},
	}

	codingAgent, err := agent.NewCodingAgent(agentConfig)
	if err != nil {
		log.Fatalf("Failed to create agent: %v", err)
	}

	// Generate task breakdown
	fmt.Printf("\n=== Coding Agent: Task Planner ===\n")
	fmt.Printf("Provider: %s\n", *provider)
	fmt.Printf("Task: %s\n\n", task)

	breakdown, err := codingAgent.PlanTask(context.Background(), task)
	if err != nil {
		log.Fatalf("Failed to plan task: %v", err)
	}

	// Output
	if *jsonOutput {
		tm := agent.NewTaskManager()
		jsonStr, _ := tm.FormatAsJSON(breakdown)
		fmt.Println(jsonStr)
	} else {
		tm := agent.NewTaskManager()
		fmt.Println(tm.FormatAsChecklist(breakdown))
	}
}

func cmdAgentChat() {
	fs := flag.NewFlagSet("agent chat", flag.ExitOnError)
	projectPath := fs.String("path", ".", "Path to the project")
	provider := fs.String("provider", "claude", "LLM provider (claude, gemini, openai, ollama)")
	model := fs.String("model", "", "Model name (provider-specific)")
	apiKey := fs.String("api-key", "", "API key (or use environment variable)")
	noContext := fs.Bool("no-context", false, "Don't include project context")
	fs.Parse(os.Args[3:])

	if fs.NArg() < 1 {
		log.Fatal("Usage: indexer agent chat \"<message>\"")
	}

	message := fs.Arg(0)
	absPath, _ := filepath.Abs(*projectPath)

	// Get API key from environment if not provided
	if *apiKey == "" {
		switch *provider {
		case "claude":
			*apiKey = os.Getenv("CLAUDE_API_KEY")
		case "gemini":
			*apiKey = os.Getenv("GEMINI_API_KEY")
		case "openai":
			*apiKey = os.Getenv("OPENAI_API_KEY")
		}
	}

	// Create agent
	agentConfig := agent.AgentConfig{
		ProjectPath: absPath,
		LLMConfig: agent.LLMConfig{
			Provider: *provider,
			APIKey:   *apiKey,
			Model:    *model,
		},
	}

	codingAgent, err := agent.NewCodingAgent(agentConfig)
	if err != nil {
		log.Fatalf("Failed to create agent: %v", err)
	}

	// Chat
	fmt.Printf("\n=== Coding Agent: Chat ===\n")
	fmt.Printf("Provider: %s\n\n", *provider)

	response, err := codingAgent.Chat(context.Background(), message, !*noContext)
	if err != nil {
		log.Fatalf("Chat failed: %v", err)
	}

	fmt.Println(response.Content)
	fmt.Printf("\n[Tokens: %d | Model: %s]\n", response.TokensUsed, response.Model)
}

func cmdAgentExplain() {
	fs := flag.NewFlagSet("agent explain", flag.ExitOnError)
	projectPath := fs.String("path", ".", "Path to the project")
	provider := fs.String("provider", "claude", "LLM provider (claude, gemini, openai, ollama)")
	model := fs.String("model", "", "Model name (provider-specific)")
	apiKey := fs.String("api-key", "", "API key (or use environment variable)")
	fs.Parse(os.Args[3:])

	if fs.NArg() < 1 {
		log.Fatal("Usage: indexer agent explain <symbol>")
	}

	symbolName := fs.Arg(0)
	absPath, _ := filepath.Abs(*projectPath)

	// Get API key from environment if not provided
	if *apiKey == "" {
		switch *provider {
		case "claude":
			*apiKey = os.Getenv("CLAUDE_API_KEY")
		case "gemini":
			*apiKey = os.Getenv("GEMINI_API_KEY")
		case "openai":
			*apiKey = os.Getenv("OPENAI_API_KEY")
		}
	}

	// Create agent
	agentConfig := agent.AgentConfig{
		ProjectPath: absPath,
		LLMConfig: agent.LLMConfig{
			Provider: *provider,
			APIKey:   *apiKey,
			Model:    *model,
		},
	}

	codingAgent, err := agent.NewCodingAgent(agentConfig)
	if err != nil {
		log.Fatalf("Failed to create agent: %v", err)
	}

	// Explain
	fmt.Printf("\n=== Coding Agent: Code Explanation ===\n")
	fmt.Printf("Symbol: %s\n", symbolName)
	fmt.Printf("Provider: %s\n\n", *provider)

	response, err := codingAgent.ExplainCode(context.Background(), symbolName)
	if err != nil {
		log.Fatalf("Explanation failed: %v", err)
	}

	fmt.Println(response.Content)
	fmt.Printf("\n[Tokens: %d | Model: %s]\n", response.TokensUsed, response.Model)
}

func cmdAgentRun() {
	fs := flag.NewFlagSet("agent run", flag.ExitOnError)
	projectPath := fs.String("path", ".", "Path to the project")
	provider := fs.String("provider", "claude", "LLM provider (claude, gemini, openai, ollama)")
	model := fs.String("model", "", "Model name (provider-specific)")
	apiKey := fs.String("api-key", "", "API key (or use environment variable)")
	dryRun := fs.Bool("dry-run", false, "If true, do not modify files or run commands")
	maxIterations := fs.Int("max-iterations", 20, "Max action iterations per task")
	maxContext := fs.Int("max-context", 8, "Max context results per task")
	fs.Parse(os.Args[3:])

	if fs.NArg() < 1 {
		log.Fatal("Usage: indexer agent run \"<task description>\"")
	}

	task := fs.Arg(0)
	absPath, _ := filepath.Abs(*projectPath)

	if *apiKey == "" {
		switch *provider {
		case "claude":
			*apiKey = os.Getenv("CLAUDE_API_KEY")
		case "gemini":
			*apiKey = os.Getenv("GEMINI_API_KEY")
		case "openai":
			*apiKey = os.Getenv("OPENAI_API_KEY")
		}
	}

	agentConfig := agent.AgentConfig{
		ProjectPath: absPath,
		LLMConfig: agent.LLMConfig{
			Provider: *provider,
			APIKey:   *apiKey,
			Model:    *model,
		},
	}

	codingAgent, err := agent.NewCodingAgent(agentConfig)
	if err != nil {
		log.Fatalf("Failed to create agent: %v", err)
	}

	fmt.Printf("\n=== Coding Agent: Autonomous Run ===\n")
	fmt.Printf("Provider: %s | Dry-run: %v\n", *provider, *dryRun)
	fmt.Printf("Task: %s\n\n", task)

	result, err := codingAgent.Run(context.Background(), task, agent.RunOptions{
		DryRun:            *dryRun,
		MaxIterations:     *maxIterations,
		MaxContextResults: *maxContext,
	})
	if err != nil {
		log.Fatalf("Agent run failed: %v", err)
	}

	// Print updated plan with statuses
	tm := agent.NewTaskManager()
	fmt.Println(tm.FormatAsChecklist(result.Plan))

	// Print execution log
	fmt.Println("\nExecution details:")
	for _, exec := range result.Executions {
		status := "pending"
		switch {
		case exec.Completed:
			status = "done"
		case exec.Failed:
			status = "failed"
		}
		fmt.Printf("\n- Task %d: %s [%s]\n", exec.Task.ID, exec.Task.Description, status)
		for i, act := range exec.Actions {
			res := exec.Results[i]
			out := strings.TrimSpace(res.Output)
			if len(out) > 160 {
				out = out[:160] + "..."
			}
			fmt.Printf("  • %s %s -> %t\n", act.Type, act.Path, res.Success)
			if out != "" {
				fmt.Printf("    %s\n", out)
			}
			if res.Error != "" {
				fmt.Printf("    error: %s\n", res.Error)
			}
		}
		if exec.FailureMsg != "" {
			fmt.Printf("  failure: %s\n", exec.FailureMsg)
		}
	}
}

// RAG Commands
func newRAGIndexer(projectPath string) *rag.RAGIndexer {
	embedder := rag.NewOllamaEmbedder("nomic-embed-text")
	dbPath := filepath.Join(projectPath, ".index", "rag_vectors.db")
	vectorStore, err := rag.NewSQLiteVectorStore(dbPath, embedder.Dimension())
	if err != nil {
		log.Fatalf("Failed to create SQLite vector store: %v", err)
	}

	return rag.NewRAGIndexer(embedder, vectorStore)
}

func cmdRAG() {
	if len(os.Args) < 3 {
		log.Fatal("Usage: indexer rag <subcommand> [options]\nSubcommands: index, search, status")
	}

	subcommand := os.Args[2]

	switch subcommand {
	case "index":
		cmdRAGIndex()
	case "search":
		cmdRAGSearch()
	case "status":
		cmdRAGStatus()
	default:
		log.Fatalf("Unknown rag subcommand: %s\nAvailable: index, search, status", subcommand)
	}
}

func cmdRAGIndex() {
	fs := flag.NewFlagSet("rag index", flag.ExitOnError)
	projectPath := fs.String("path", ".", "Path to the project to index")
	fs.Parse(os.Args[3:])

	absPath, _ := filepath.Abs(*projectPath)

	fmt.Printf("\n=== RAG Indexer ===\n")
	fmt.Printf("Building semantic index for: %s\n\n", absPath)

	indexer := newRAGIndexer(absPath)

	err := indexer.IndexProject(absPath)
	if err != nil {
		log.Fatalf("Failed to index project: %v", err)
	}

	stats := indexer.Stats()
	fmt.Printf("\nIndex Statistics:\n")
	fmt.Printf("  Files:    %d\n", stats.TotalFiles)
	fmt.Printf("  Chunks:   %d\n", stats.TotalChunks)
	fmt.Printf("  Model:    %s\n", stats.EmbeddingModel)
	fmt.Printf("  Dims:     %d\n", stats.Dimensions)
}

func cmdRAGSearch() {
	fs := flag.NewFlagSet("rag search", flag.ExitOnError)
	topK := fs.Int("top-k", 10, "Number of results to return")
	jsonOutput := fs.Bool("json", false, "Output in JSON format")
	projectPath := fs.String("path", ".", "Path to the project to search")
	fs.Parse(os.Args[3:])

	if fs.NArg() < 1 {
		log.Fatal("Usage: indexer rag search \"<query>\"")
	}

	query := fs.Arg(0)
	absPath, _ := filepath.Abs(*projectPath)

	indexer := newRAGIndexer(absPath)

	if indexer.Stats().TotalChunks == 0 {
		log.Fatal("RAG index is empty. Please run 'indexer rag index <path>' first.")
	}

	fmt.Printf("\n=== RAG Search ===\n")
	fmt.Printf("Query: %s\n\n", query)

	results, err := indexer.Search(query, *topK)
	if err != nil {
		log.Fatalf("Search failed: %v", err)
	}

	if *jsonOutput {
		jsonData, _ := json.MarshalIndent(results, "", "  ")
		fmt.Println(string(jsonData))
		return
	}

	fmt.Printf("Found %d results:\n\n", len(results))
	for i, result := range results {
		fmt.Printf("%d. [Score: %.3f] %s\n", i+1, result.Score, result.Chunk.FilePath)
		fmt.Printf("   Lines %d-%d: %s\n", result.Chunk.StartLine, result.Chunk.EndLine, result.Chunk.SymbolName)
		fmt.Printf("   Type: %s | Language: %s\n", result.Chunk.ChunkType, result.Chunk.Language)

		// Show snippet
		lines := strings.Split(result.Chunk.Content, "\n")
		previewLines := 3
		if len(lines) > previewLines {
			fmt.Printf("   Preview: %s\n", strings.Join(lines[:previewLines], "\n            "))
			fmt.Printf("            ... (%d more lines)\n", len(lines)-previewLines)
		} else {
			fmt.Printf("   Preview: %s\n", strings.Join(lines, "\n            "))
		}
		fmt.Println()
	}
}

func cmdRAGStatus() {
	fs := flag.NewFlagSet("rag status", flag.ExitOnError)
	projectPath := fs.String("path", ".", "Path to the project")
	fs.Parse(os.Args[3:])

	absPath, _ := filepath.Abs(*projectPath)

	indexer := newRAGIndexer(absPath)
	stats := indexer.Stats()

	fmt.Printf("\n=== RAG Index Status ===\n\n")
	fmt.Printf("Total Files:     %d\n", stats.TotalFiles)
	fmt.Printf("Total Chunks:    %d\n", stats.TotalChunks)
	fmt.Printf("Embedding Model: %s\n", stats.EmbeddingModel)
	fmt.Printf("Dimensions:      %d\n", stats.Dimensions)

	if stats.LastUpdated != "" {
		fmt.Printf("Last Updated:    %s\n", stats.LastUpdated)
	}

	if stats.TotalChunks == 0 {
		fmt.Printf("\n⚠ Index is empty. Run 'indexer rag index <path>' to build the index.\n")
	} else {
		fmt.Printf("\n✓ Index is ready\n")
	}
}
