package agent

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/yourorg/agent/internal/indexer"
)

// Executor is responsible for carrying out actions produced by the agent brain.
type Executor struct {
	projectRoot string
	index       *indexer.ProjectIndex
	dryRun      bool
	blocklist   []string
}

// ExecutorConfig configures an Executor instance.
type ExecutorConfig struct {
	ProjectRoot string
	Index       *indexer.ProjectIndex
	DryRun      bool
	Blocklist   []string
}

// NewExecutor creates a new executor with sensible defaults.
func NewExecutor(cfg ExecutorConfig) *Executor {
	blocked := cfg.Blocklist
	if len(blocked) == 0 {
		blocked = []string{".env", "id_rsa", "id_dsa", "secrets", "config.yml", "config.yaml"}
	}

	return &Executor{
		projectRoot: cfg.ProjectRoot,
		index:       cfg.Index,
		dryRun:      cfg.DryRun,
		blocklist:   blocked,
	}
}

// Execute runs a single action and returns the result.
func (e *Executor) Execute(ctx context.Context, action Action) ActionResult {
	start := time.Now()

	switch action.Type {
	case ActionReadFile:
		content, err := os.ReadFile(e.abs(action.Path))
		if err != nil {
			return e.result(false, "", err, start)
		}
		return e.result(true, string(content), nil, start)

	case ActionCreateFile:
		if err := e.checkPath(action.Path); err != nil {
			return e.result(false, "", err, start)
		}
		if e.dryRun {
			return e.result(true, fmt.Sprintf("[dry-run] would create %s", action.Path), nil, start)
		}
		if err := os.MkdirAll(filepath.Dir(e.abs(action.Path)), 0o755); err != nil {
			return e.result(false, "", err, start)
		}
		if err := os.WriteFile(e.abs(action.Path), []byte(action.Content), 0o644); err != nil {
			return e.result(false, "", err, start)
		}
		return e.result(true, fmt.Sprintf("created %s", action.Path), nil, start, action.Path)

	case ActionEditFile:
		if err := e.checkPath(action.Path); err != nil {
			return e.result(false, "", err, start)
		}
		if len(action.Edits) == 0 {
			return e.result(false, "", fmt.Errorf("no edits provided"), start)
		}
		absPath := e.abs(action.Path)
		data, err := os.ReadFile(absPath)
		if err != nil {
			return e.result(false, "", err, start)
		}
		content := string(data)
		for _, edit := range action.Edits {
			if !strings.Contains(content, edit.OldText) {
				return e.result(false, "", fmt.Errorf("old_text not found in %s", action.Path), start)
			}
			content = strings.Replace(content, edit.OldText, edit.NewText, 1)
		}
		if e.dryRun {
			return e.result(true, fmt.Sprintf("[dry-run] would edit %s", action.Path), nil, start)
		}
		if err := os.WriteFile(absPath, []byte(content), 0o644); err != nil {
			return e.result(false, "", err, start)
		}
		return e.result(true, fmt.Sprintf("edited %s", action.Path), nil, start, action.Path)

	case ActionDeleteFile:
		if err := e.checkPath(action.Path); err != nil {
			return e.result(false, "", err, start)
		}
		if e.dryRun {
			return e.result(true, fmt.Sprintf("[dry-run] would delete %s", action.Path), nil, start)
		}
		if err := os.Remove(e.abs(action.Path)); err != nil {
			return e.result(false, "", err, start)
		}
		return e.result(true, fmt.Sprintf("deleted %s", action.Path), nil, start, action.Path)

	case ActionRunCommand:
		workdir := action.Workdir
		if workdir == "" {
			workdir = e.projectRoot
		}
		timeout := time.Duration(action.Timeout) * time.Second
		if timeout == 0 {
			timeout = 5 * time.Minute
		}
		runCtx, cancel := context.WithTimeout(ctx, timeout)
		defer cancel()

		if e.dryRun {
			return e.result(true, fmt.Sprintf("[dry-run] would run '%s' (cwd=%s)", action.Command, workdir), nil, start)
		}

		cmd := exec.CommandContext(runCtx, "bash", "-c", action.Command)
		cmd.Dir = workdir
		output, err := cmd.CombinedOutput()
		if err != nil {
			return e.result(false, string(output), err, start)
		}
		return e.result(true, string(output), nil, start)

	case ActionSearch:
		if e.index == nil {
			return e.result(false, "", fmt.Errorf("search unavailable: index is nil"), start)
		}
		engine := indexer.NewSearchEngine(e.index)
		results := engine.SearchSymbol(action.Query)
		var b strings.Builder
		for _, r := range results {
			b.WriteString(indexer.FormatSearchResult(r))
			b.WriteString("\n")
		}
		return e.result(true, b.String(), nil, start)

	case ActionAskUser:
		// Ask_user is a no-op for automation; bubble up the question.
		return e.result(false, action.Question, fmt.Errorf("user input required"), start)

	case ActionComplete:
		return e.result(true, action.Summary, nil, start)

	case ActionFail:
		return e.result(false, "", fmt.Errorf("%s", action.Reason), start)

	default:
		return e.result(false, "", fmt.Errorf("unsupported action type: %s", action.Type), start)
	}
}

func (e *Executor) abs(path string) string {
	if filepath.IsAbs(path) {
		return path
	}
	return filepath.Join(e.projectRoot, path)
}

func (e *Executor) checkPath(path string) error {
	abs := e.abs(path)
	if !strings.HasPrefix(abs, filepath.Clean(e.projectRoot)) {
		return fmt.Errorf("path %s escapes project root", path)
	}
	for _, blocked := range e.blocklist {
		if strings.Contains(abs, blocked) {
			return fmt.Errorf("path %s is blocked", path)
		}
	}
	return nil
}

func (e *Executor) result(success bool, output string, err error, start time.Time, changed ...string) ActionResult {
	res := ActionResult{
		Success:      success,
		Output:       output,
		Duration:     time.Since(start),
		FilesChanged: changed,
	}
	if err != nil {
		res.Error = err.Error()
	}
	return res
}
